# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FSDP actor update for multiview MeSD and Direct RKL."""

from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm

from ...protocol import DataProto, batch_collate
from ...trainer.mesd_algos import (
    MESD_PRIVILEGED_VIEW_NAMES,
    apply_mesd_advantage_reweighting,
    build_mesd_answer_anchored_avsd_gap,
    build_mesd_multiview_direction_token_loss,
)
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .dp_actor import DataParallelPPOActor

try:
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelMeSDActor"]


class DataParallelMeSDActor(DataParallelPPOActor):
    """Train sampled student trajectories against detached multiview EMA targets."""

    def __init__(
        self,
        config,
        actor_module,
        actor_optimizer=None,
        teacher_module: Optional[torch.nn.Module] = None,
    ):
        super().__init__(config=config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        self.teacher_module = teacher_module

    def _get_teacher_module(self):
        return self.teacher_module if self.teacher_module is not None else self.actor_module

    def _ema_teacher_from_actor(self, update_rate: float) -> None:
        """Update the separate MeSD teacher from the post-step actor shards."""

        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise RuntimeError("MeSD EMA requires a separate teacher module.")
        rate = float(update_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"MeSD teacher EMA rate must be in [0, 1], got {rate}.")

        teacher_params = list(self.teacher_module.parameters())
        actor_params = list(self.actor_module.parameters())
        teacher_buffers = list(self.teacher_module.buffers())
        actor_buffers = list(self.actor_module.buffers())
        if len(teacher_params) != len(actor_params):
            raise RuntimeError(
                f"MeSD teacher/actor parameter layouts differ: {len(teacher_params)} != {len(actor_params)}."
            )
        if len(teacher_buffers) != len(actor_buffers):
            raise RuntimeError(
                f"MeSD teacher/actor buffer layouts differ: {len(teacher_buffers)} != {len(actor_buffers)}."
            )

        with torch.no_grad():
            for index, (teacher_param, actor_param) in enumerate(zip(teacher_params, actor_params)):
                if tuple(teacher_param.shape) != tuple(actor_param.shape):
                    raise RuntimeError(
                        f"MeSD teacher/actor parameter {index} shape mismatch: "
                        f"{tuple(teacher_param.shape)} != {tuple(actor_param.shape)}."
                    )
                actor_data = actor_param.data.to(
                    device=teacher_param.device,
                    dtype=teacher_param.dtype,
                )
                teacher_param.data.mul_(1.0 - rate).add_(actor_data, alpha=rate)

            for index, (teacher_buffer, actor_buffer) in enumerate(zip(teacher_buffers, actor_buffers)):
                if tuple(teacher_buffer.shape) != tuple(actor_buffer.shape):
                    raise RuntimeError(
                        f"MeSD teacher/actor buffer {index} shape mismatch: "
                        f"{tuple(teacher_buffer.shape)} != {tuple(actor_buffer.shape)}."
                    )
                actor_data = actor_buffer.data.to(
                    device=teacher_buffer.device,
                    dtype=teacher_buffer.dtype,
                )
                if torch.is_floating_point(teacher_buffer):
                    teacher_buffer.data.mul_(1.0 - rate).add_(actor_data, alpha=rate)
                else:
                    teacher_buffer.data.copy_(actor_data)

    @staticmethod
    def _compute_mesd_policy_loss(
        *,
        old_log_probs: torch.Tensor,
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        clip_ratio_low: float,
        clip_ratio_high: float,
        sequence_valid_mask: Optional[torch.Tensor] = None,
        direction_token_loss: Optional[torch.Tensor] = None,
        direction_mask: Optional[torch.Tensor] = None,
        direction_coef: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """MeSD token-GRPO with optional per-sequence direction replacement.

        This intentionally has no PPO dual-clip branch and ignores the actor's
        generic loss type/averaging settings.  Those defaults belong to the
        original RLSD path and must not silently change MeSD's objective.
        """

        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        clipped_ratio = torch.clamp(
            ratio,
            min=1.0 - clip_ratio_low,
            max=1.0 + clip_ratio_high,
        )
        unclipped_loss = -advantages * ratio
        clipped_loss = -advantages * clipped_ratio
        token_loss = torch.maximum(unclipped_loss, clipped_loss)
        sequence_denom = response_mask.sum(dim=-1) + 1.0e-8
        pg_sequence_loss = (token_loss * response_mask).sum(dim=-1) / sequence_denom

        batch_size = response_mask.size(0)
        if sequence_valid_mask is None:
            valid = torch.ones(batch_size, dtype=response_mask.dtype, device=response_mask.device)
        else:
            if tuple(sequence_valid_mask.shape) != (batch_size,):
                raise ValueError("sequence_valid_mask must have shape [batch].")
            valid = sequence_valid_mask.to(device=response_mask.device, dtype=response_mask.dtype)

        route = torch.zeros_like(valid)
        direction_sequence_loss = torch.zeros_like(pg_sequence_loss)
        if direction_token_loss is not None or direction_mask is not None:
            if direction_token_loss is None or direction_mask is None:
                raise ValueError("direction_token_loss and direction_mask must be provided together.")
            if tuple(direction_token_loss.shape) != tuple(response_mask.shape):
                raise ValueError("direction_token_loss must match response_mask.")
            if tuple(direction_mask.shape) != (batch_size,):
                raise ValueError("direction_mask must have shape [batch].")
            if not torch.isfinite(direction_token_loss).all():
                raise FloatingPointError("direction_token_loss contains a non-finite value.")
            if not torch.isfinite(torch.as_tensor(float(direction_coef))):
                raise ValueError("direction_coef must be finite.")
            if float(direction_coef) < 0.0:
                raise ValueError("direction_coef must be non-negative.")
            route = direction_mask.to(device=response_mask.device, dtype=response_mask.dtype) * valid
            direction_sequence_loss = (direction_token_loss * response_mask).sum(dim=-1) / sequence_denom

        route_bool = route > 0.5
        selected_sequence_loss = torch.where(
            route_bool,
            float(direction_coef) * direction_sequence_loss,
            pg_sequence_loss,
        )
        combined_sequence_loss = torch.where(
            valid > 0.5,
            selected_sequence_loss,
            torch.zeros_like(selected_sequence_loss),
        )
        # Keep a local batch mean, then apply DP-global valid-sequence scaling
        # at the call site.  This preserves hard routing under micro-batching.
        policy_loss = combined_sequence_loss.mean()

        upper_clipped = ((ratio > (1.0 + clip_ratio_high)) & (advantages > 0)).to(response_mask.dtype)
        lower_clipped = ((ratio < (1.0 - clip_ratio_low)) & (advantages < 0)).to(response_mask.dtype)
        metric_mask = response_mask * valid.unsqueeze(-1)
        valid_denom = valid.sum().clamp_min(1.0)
        route_denom = route.sum().clamp_min(1.0)
        pg_kept = valid * (1.0 - route)
        pg_kept_denom = pg_kept.sum().clamp_min(1.0)
        metrics = {
            "ppo_kl": VF.masked_mean(-log_ratio.detach(), metric_mask).item(),
            "entropy_loss": (((-log_probs.detach() * response_mask).sum(dim=-1) / sequence_denom) * valid)
            .sum()
            .div(valid_denom)
            .item(),
            "pg_clipfrac_higher": VF.masked_mean(upper_clipped, metric_mask).item(),
            "pg_clipfrac_lower": VF.masked_mean(lower_clipped, metric_mask).item(),
            "mesd_pg_sequence_loss": float((pg_sequence_loss.detach() * pg_kept).sum().div(pg_kept_denom).item()),
            "mesd_direction_sequence_loss": float(
                (direction_sequence_loss.detach() * route).sum().div(route_denom).item()
            ),
            "mesd_direction_route_fraction": float(route.sum().div(valid_denom).detach().item()),
        }
        return policy_loss, metrics

    def _forward_micro_batch_logits(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        module: Optional[torch.nn.Module] = None,
    ) -> torch.Tensor:
        """Return sliced logits aligned with `responses` for top-k distillation."""
        forward_module = module if module is not None else self.actor_module
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:
            position_ids = position_ids.transpose(0, 1)

        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
            else:
                pad_size = 0

            output = forward_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits_rmpad = output.logits.squeeze(0)
            logits_rmpad.div_(temperature)

            if self.config.ulysses_size > 1:
                logits_rmpad = gather_outputs_and_unpad(logits_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            full_logits = pad_input(hidden_states=logits_rmpad, indices=indices, batch=batch_size, seqlen=seqlen)
            logits = full_logits[:, -response_length - 1 : -1, :]
        else:
            output = forward_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]

        # A response slice otherwise retains the full [B, S, V] backing
        # storage.  Materializing the short response here is essential when
        # keeping A/T/S (and optionally an unprivileged EMA) logits together.
        return logits.contiguous()

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        """Direct RKL replacement for confirmed failures; MeSD credit for other rows.

        This path keeps the standard PPO/GRPO policy loss and injects Teacher
        guidance through token-level advantage reweighting on the same sampled
        responses.
        """
        self.actor_module.train()
        teacher_module = self._get_teacher_module()
        if teacher_module is not self.actor_module:
            teacher_module.eval()
        temperature = data.meta_info["temperature"]
        mesd_config = data.meta_info["mesd_config"]
        mesd_enabled = bool(mesd_config.get("mesd_enabled", False))
        mesd_teacher_enabled = bool(mesd_config.get("mesd_teacher_enabled", False))
        mesd_teacher_aggregation = str(mesd_config.get("mesd_teacher_aggregation", "anchored_avsd"))
        mesd_multiview_enabled = mesd_enabled and mesd_teacher_aggregation == "anchored_avsd"
        mesd_direction_mode = str(mesd_config.get("mesd_failure_direction_mode", "direct_rkl"))
        mesd_direction_enabled = (
            mesd_multiview_enabled
            and mesd_teacher_enabled
            and (mesd_direction_mode == "direct_rkl")
            and (float(mesd_config.get("mesd_failure_direction_gamma", 0.0)) > 0.0)
        )
        select_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "old_log_probs",
            "advantages",
        ]
        if mesd_teacher_enabled:
            for view_name in MESD_PRIVILEGED_VIEW_NAMES:
                select_keys.extend(
                    [
                        f"mesd_teacher_{view_name}_input_ids",
                        f"mesd_teacher_{view_name}_attention_mask",
                        f"mesd_teacher_{view_name}_position_ids",
                    ]
                )
            select_keys.append("mesd_view_availability")
        if mesd_teacher_enabled:
            select_keys.append("mesd_target_mask")
        select_keys.append("mesd_group_valid_mask")
        if mesd_direction_enabled:
            select_keys.append("mesd_direction_mask")
        mini_batches = data.select(select_keys, ["multi_modal_inputs"]).split(self.config.global_batch_size_per_device)
        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)
            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)
                total_sequence_count = None
                total_sequence_count = (
                    mini_batch.batch["mesd_group_valid_mask"]
                    .to(device=mini_batch.batch["response_mask"].device, dtype=torch.float32)
                    .sum()
                )
                if total_sequence_count.ndim != 0:
                    raise ValueError("mesd_group_valid_mask must reduce to a scalar sequence count.")
                dist.all_reduce(total_sequence_count, op=dist.ReduceOp.SUM)
                if total_sequence_count.item() <= 0:
                    append_to_dict(metrics, {"mesd/all_invalid_minibatch_skip": 1.0})
                    continue
                append_to_dict(metrics, {"mesd/all_invalid_minibatch_skip": 0.0})
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)
                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]
                    sequence_valid_mask = model_inputs.get("mesd_group_valid_mask")
                    direction_token_loss = None
                    direction_mask = None
                    if mesd_teacher_enabled:
                        if teacher_module is self.actor_module:
                            raise RuntimeError("MeSD requires a separate EMA teacher model.")
                        student_logits = self._forward_micro_batch_logits(
                            model_inputs, temperature=temperature, module=self.actor_module
                        )
                        log_probs = self.log_probs_from_logits(
                            logits=student_logits, labels=micro_batch.batch["responses"]
                        )
                        raw_view_availability = micro_batch.batch["mesd_view_availability"].to(dtype=torch.bool)
                        expected_view_shape = (response_mask.size(0), len(MESD_PRIVILEGED_VIEW_NAMES))
                        if tuple(raw_view_availability.shape) != expected_view_shape:
                            raise ValueError(
                                f"mesd_view_availability must have shape {expected_view_shape}, got {tuple(raw_view_availability.shape)}."
                            )
                        target_mask = model_inputs["mesd_target_mask"].to(
                            device=raw_view_availability.device, dtype=torch.bool
                        )
                        effective_view_availability = raw_view_availability & target_mask.unsqueeze(-1)
                        teacher_module.eval()
                        teacher_logits_by_view: dict[str, torch.Tensor] = {}
                        teacher_forward_by_view: dict[str, float] = {}
                        with torch.no_grad():
                            for view_index, view_name in enumerate(MESD_PRIVILEGED_VIEW_NAMES):
                                active_view_flag = (
                                    effective_view_availability[:, view_index].any().to(dtype=torch.int32)
                                )
                                dist.all_reduce(active_view_flag, op=dist.ReduceOp.MAX)
                                has_active_view = bool(active_view_flag.item())
                                teacher_forward_by_view[view_name] = float(has_active_view)
                                if not has_active_view:
                                    teacher_logits_by_view[view_name] = student_logits.detach()
                                    continue
                                teacher_mb = {
                                    "input_ids": micro_batch.batch[f"mesd_teacher_{view_name}_input_ids"],
                                    "attention_mask": micro_batch.batch[
                                        f"mesd_teacher_{view_name}_attention_mask"
                                    ],
                                    "position_ids": micro_batch.batch[f"mesd_teacher_{view_name}_position_ids"],
                                    "responses": micro_batch.batch["responses"],
                                }
                                if "multi_modal_inputs" in micro_batch.non_tensor_batch:
                                    teacher_mb["multi_modal_inputs"] = micro_batch.non_tensor_batch[
                                        "multi_modal_inputs"
                                    ]
                                teacher_logits_by_view[view_name] = self._forward_micro_batch_logits(
                                    teacher_mb, temperature=temperature, module=teacher_module
                                )
                        target_response_mask = response_mask * target_mask.to(response_mask.dtype).unsqueeze(-1)
                        teacher_gap, avsd_metrics = build_mesd_answer_anchored_avsd_gap(
                            student_logits=student_logits,
                            answer_teacher_logits=teacher_logits_by_view["answer"],
                            temporal_teacher_logits=teacher_logits_by_view["temporal"],
                            spatial_teacher_logits=teacher_logits_by_view["spatial"],
                            sampled_token_ids=micro_batch.batch["responses"],
                            view_availability=effective_view_availability,
                            response_mask=target_response_mask,
                            top_k=int(mesd_config.get("mesd_topk", 16)),
                            evidence_alpha=float(mesd_config.get("mesd_avsd_evidence_alpha", 1.0)),
                            epsilon=float(mesd_config.get("mesd_avsd_epsilon", 1e-06)),
                        )
                        advantages_for_pg, _, mesd_metrics = apply_mesd_advantage_reweighting(
                            advantages=advantages,
                            teacher_gap=teacher_gap,
                            response_mask=response_mask,
                            mixing_lambda=float(mesd_config.get("mesd_lambda", 0.0)),
                            weight_clip=mesd_config.get("mesd_weight_clip", 0.2),
                            target_mask=model_inputs["mesd_target_mask"],
                        )
                        if mesd_direction_enabled:
                            local_direction_mask = micro_batch.batch["mesd_direction_mask"].to(
                                device=response_mask.device, dtype=torch.bool
                            )
                            active_direction_flag = local_direction_mask.any().to(dtype=torch.int32)
                            dist.all_reduce(active_direction_flag, op=dist.ReduceOp.MAX)
                            has_global_direction = bool(active_direction_flag.item())
                            if has_global_direction:
                                direction_token_loss, direction_metrics = build_mesd_multiview_direction_token_loss(
                                    student_logits=student_logits,
                                    answer_teacher_logits=teacher_logits_by_view["answer"],
                                    temporal_teacher_logits=teacher_logits_by_view["temporal"],
                                    spatial_teacher_logits=teacher_logits_by_view["spatial"],
                                    sampled_token_ids=micro_batch.batch["responses"],
                                    view_availability=effective_view_availability,
                                    response_mask=response_mask,
                                    direction_mask=local_direction_mask,
                                    top_k=int(mesd_config.get("mesd_failure_direction_topk", 32)),
                                    evidence_alpha=float(mesd_config.get("mesd_avsd_evidence_alpha", 1.0)),
                                    avsd_epsilon=float(mesd_config.get("mesd_avsd_epsilon", 1e-06)),
                                    target_epsilon=float(mesd_config.get("mesd_failure_target_epsilon", 1e-08)),
                                )
                                direction_mask = local_direction_mask
                                mesd_metrics.update(direction_metrics)
                            mesd_metrics["mesd/direction_global_active"] = float(has_global_direction)
                        target_denom = target_response_mask.sum().clamp(min=1.0)
                        mesd_metrics.update(avsd_metrics)
                        mesd_metrics.update(
                            {
                                "mesd/teacher_forward": float(any(teacher_forward_by_view.values())),
                                "mesd/teacher_view_forward_count": float(
                                    sum(teacher_forward_by_view.values())
                                ),
                                "mesd/teacher_forward_answer": teacher_forward_by_view["answer"],
                                "mesd/teacher_forward_temporal": teacher_forward_by_view["temporal"],
                                "mesd/teacher_forward_spatial": teacher_forward_by_view["spatial"],
                                "mesd/teacher_gap_mean": float(
                                    (teacher_gap.detach() * target_response_mask).sum().div(target_denom).item()
                                ),
                            }
                        )
                        del teacher_logits_by_view
                    else:
                        log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                        advantages_for_pg = advantages
                        mesd_metrics = {
                            "mesd/teacher_forward": 0.0,
                            "mesd/weight_mean": 1.0,
                            "mesd/effective_token_fraction": 0.0,
                        }
                    pg_loss, pg_metrics = self._compute_mesd_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages_for_pg,
                        response_mask=response_mask,
                        clip_ratio_low=float(mesd_config.get("mesd_policy_clip_low", 0.2)),
                        clip_ratio_high=float(mesd_config.get("mesd_policy_clip_high", 0.2)),
                        sequence_valid_mask=sequence_valid_mask,
                        direction_token_loss=direction_token_loss,
                        direction_mask=direction_mask,
                        direction_coef=float(mesd_config.get("mesd_failure_direction_gamma", 0.0)),
                    )
                    loss = pg_loss
                    assert total_sequence_count is not None
                    loss = loss * float(response_mask.size(0)) * self.world_size / total_sequence_count.clamp_min(1.0)
                    loss.backward()
                    batch_metrics = {
                        (f"mesd/{k.removeprefix('mesd_')}" if k.startswith("mesd_") else f"actor/{k}"): v
                        for k, v in pg_metrics.items()
                    }
                    batch_metrics["actor/pg_loss"] = pg_loss.detach().item()
                    batch_metrics.update(mesd_metrics)
                    append_to_dict(metrics, batch_metrics)
                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})
        global_step = mesd_config.get("global_step", 0)
        update_interval = int(mesd_config.get("mesd_teacher_update_interval", 1))
        if mesd_teacher_enabled and global_step > 0 and (global_step % update_interval == 0):
            self._ema_teacher_from_actor(update_rate=float(mesd_config.get("mesd_teacher_update_rate", 0.01)))
            append_to_dict(metrics, {"mesd/teacher_ema_applied": 1.0})
        else:
            append_to_dict(metrics, {"mesd/teacher_ema_applied": 0.0})
        return metrics
