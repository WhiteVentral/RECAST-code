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
"""STGR training with multiview MeSD and confirmed-failure Direct RKL."""

from typing import Any, Optional

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm

from ..protocol import DataProto
from ..utils.py_functional import timer
from .core_algos import compute_grpo_advantage
from .metrics import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from .ray_trainer import RayTrainerBase
from .mesd_algos import (
    MESD_PRIVILEGED_VIEW_NAMES,
    build_mesd_privileged_views,
    compute_mesd_mixing_lambda,
    mesd_teacher_enabled,
)
from .mesd_prompts import (
    construct_mesd_multiview_teacher_prompts,
)


def _concat_prompt_and_response(
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    prompt_position_ids: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate prompt and response following the rollout engine's position-id logic."""
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.cat([prompt_attention_mask, response_mask], dim=1)

    response_length = response_ids.size(1)
    batch_size = prompt_ids.size(0)
    delta_position_id = torch.arange(1, response_length + 1, device=prompt_position_ids.device)
    delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
    if prompt_position_ids.ndim == 3:
        delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
            batch_size, prompt_position_ids.size(1), -1
        )
    response_position_ids = prompt_position_ids[..., -1:] + delta_position_id
    position_ids = torch.cat([prompt_position_ids, response_position_ids], dim=-1)
    return input_ids, attention_mask, position_ids


_MESD_OUTCOME_FIELDS = (
    "outcome_success",
    "outcome_failure",
    "outcome_unknown",
    "outcome_row_valid",
    "outcome_rkl_eligible",
    "outcome_task_eligible",
)


def _attach_mesd_outcome_tensors(
    batch: DataProto,
    reward_metrics: dict[str, list[float]],
    *,
    rollout_n: int,
) -> dict[str, float]:
    """Attach absolute S/F/U routing without changing the fixed batch shape.

    ``uid`` identifies one prompt's rollout group after length balancing.  A
    system-invalid group is retained for logical-resume/FSDP parity, but its
    GRPO and direction losses are masked later.  A malformed model response is
    an unknown outcome, not a system-invalid row.
    """

    batch_size = len(batch)
    device = batch.batch["response_mask"].device
    dtype = batch.batch["response_mask"].dtype
    missing = [name for name in _MESD_OUTCOME_FIELDS if name not in reward_metrics]
    if missing:
        raise KeyError(
            f"MeSD direction control requires absolute outcome metrics from the reward function; missing {missing}."
        )

    values: dict[str, np.ndarray] = {}
    for name in _MESD_OUTCOME_FIELDS:
        raw = np.asarray(reward_metrics[name], dtype=np.float32)
        if raw.shape != (batch_size,):
            raise ValueError(f"Reward metric {name!r} must have {batch_size} rows, got {raw.shape}.")
        if not np.isfinite(raw).all():
            raise ValueError(f"Reward metric {name!r} contains a non-finite value.")
        if not np.logical_or(
            np.isclose(raw, 0.0, atol=1.0e-6, rtol=0.0),
            np.isclose(raw, 1.0, atol=1.0e-6, rtol=0.0),
        ).all():
            raise ValueError(f"Reward metric {name!r} must be binary per row.")
        values[name] = raw

    vision_skip = np.asarray(reward_metrics.get("vision_skip", [0.0] * batch_size), dtype=np.float32)
    if vision_skip.shape != (batch_size,) or not np.isfinite(vision_skip).all():
        raise ValueError("Reward metric 'vision_skip' must be a finite vector of batch length.")

    status_sum = values["outcome_success"] + values["outcome_failure"] + values["outcome_unknown"]
    if not np.allclose(status_sum, 1.0, atol=1.0e-6, rtol=0.0):
        raise ValueError("MeSD outcome_success/failure/unknown must be one-hot per row.")
    if bool(
        np.logical_and(
            values["outcome_rkl_eligible"] > 0.5,
            values["outcome_failure"] <= 0.5,
        ).any()
    ):
        raise ValueError("MeSD RKL eligibility must be a subset of confirmed failure.")
    if bool(
        np.logical_and(
            values["outcome_row_valid"] <= 0.5,
            values["outcome_unknown"] <= 0.5,
        ).any()
    ):
        raise ValueError("A system-invalid MeSD row must abstain with outcome_unknown=1.")

    row_system_valid = (values["outcome_row_valid"] > 0.5) & (vision_skip <= 0.5)
    uids = np.asarray(batch.non_tensor_batch["uid"], dtype=object)
    if uids.shape != (batch_size,):
        raise ValueError("MeSD uid vector must have one entry per rollout row.")
    group_valid = np.zeros(batch_size, dtype=bool)
    for uid in dict.fromkeys(uids.tolist()):
        indices = np.flatnonzero(uids == uid)
        if indices.size != int(rollout_n):
            raise ValueError(f"MeSD rollout group {uid!r} has {indices.size} rows; expected {rollout_n}.")
        # Media/ground-truth validity is a prompt-level contract.  Do not let a
        # partially invalid group enter GRPO normalization.
        group_valid[indices] = bool(row_system_valid[indices].all())

    tensor_names = {
        "outcome_success": "mesd_outcome_success_mask",
        "outcome_failure": "mesd_outcome_failure_mask",
        "outcome_unknown": "mesd_outcome_unknown_mask",
        "outcome_rkl_eligible": "mesd_rkl_eligible_mask",
        "outcome_task_eligible": "mesd_outcome_task_eligible_mask",
    }
    for source, destination in tensor_names.items():
        batch.batch[destination] = torch.as_tensor(values[source] > 0.5, device=device, dtype=dtype)
    batch.batch["mesd_group_valid_mask"] = torch.as_tensor(group_valid, device=device, dtype=dtype)

    valid_denom = max(float(group_valid.sum()), 1.0)
    return {
        "mesd/group_valid_fraction": float(group_valid.mean()),
        "mesd/outcome_success_fraction": float(
            ((values["outcome_success"] > 0.5) & group_valid).sum() / valid_denom
        ),
        "mesd/outcome_failure_fraction": float(
            ((values["outcome_failure"] > 0.5) & group_valid).sum() / valid_denom
        ),
        "mesd/outcome_unknown_fraction": float(
            ((values["outcome_unknown"] > 0.5) & group_valid).sum() / valid_denom
        ),
        "mesd/rkl_candidate_fraction": float(
            ((values["outcome_rkl_eligible"] > 0.5) & group_valid).sum() / valid_denom
        ),
    }


class RayMeSDTrainer(RayTrainerBase):
    """Direct RKL training with synchronous rewards and an EMA teacher."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.mesd_config = self.config.mesd
        self.config.worker.actor.mesd_enabled = True
        self.config.worker.actor.freeze_teacher_model = True
        if self.config.algorithm.adv_estimator != "grpo":
            raise ValueError("Direct RKL requires algorithm.adv_estimator=grpo.")
        if self.config.worker.rollout.n <= 1:
            raise ValueError("Direct RKL requires worker.rollout.n > 1.")
        mesd_aggregation = self.mesd_config.mesd_teacher_aggregation
        if not self.config.algorithm.disable_kl or self.config.algorithm.use_kl_loss:
            raise ValueError("MeSD parity requires algorithm.disable_kl=true and algorithm.use_kl_loss=false.")
        if self.config.worker.actor.ppo_epochs != 1:
            raise ValueError("MeSD parity requires worker.actor.ppo_epochs=1.")
        if self.config.algorithm.grpo_eps != 0.0001:
            raise ValueError("MeSD parity requires algorithm.grpo_eps=1e-4.")
        if self.config.worker.actor.global_batch_size != self.config.data.rollout_batch_size:
            raise ValueError(
                "MeSD parity requires one optimizer update per rollout step: worker.actor.global_batch_size must equal data.rollout_batch_size."
            )
        if self.config.algorithm.online_filtering:
            raise ValueError("The released MeSD baseline does not use online filtering.")
        if self.config.worker.ref.offload.offload_params:
            raise ValueError(
                "MeSD's same-step teacher forward and EMA update require worker.ref.offload.offload_params=false in the current worker implementation."
            )
        if self.mesd_config.mesd_reweighting_topk_gamma != 0.0:
            raise ValueError(
                "anchored_avsd already constructs a common-support teacher distribution, so mesd.mesd_reweighting_topk_gamma must be 0."
            )
        if self.mesd_config.mesd_avsd_use_token_roles:
            raise ValueError(
                "mesd.mesd_avsd_use_token_roles=true is not implemented in the minimal anchored_avsd training path yet."
            )
        if self.config.worker.actor.dynamic_batching:
            raise ValueError(
                "anchored_avsd currently requires worker.actor.dynamic_batching=false so every FSDP rank executes an identical A/T/S teacher-forward schedule."
            )
        if self.config.algorithm.pipeline_reward:
            print(
                "[MeSD] Disabling algorithm.pipeline_reward so Open-o3 reward calls receive the current step metadata."
            )
            self.config.algorithm.pipeline_reward = False
        print("[MeSD] Mode: Direct RKL with multiview EMA replay")
        print(f"[MeSD]   - teacher aggregation: {mesd_aggregation}")
        print(f"[MeSD]   - teacher EMA rate: {self.mesd_config.mesd_teacher_update_rate}")
        print(f"[MeSD]   - teacher EMA interval: {self.mesd_config.mesd_teacher_update_interval}")
        print(
            f"[MeSD]   - top-k/gamma: {self.mesd_config.mesd_reweighting_topk}/{self.mesd_config.mesd_reweighting_topk_gamma}"
        )
        print(f"[MeSD]   - initial lambda: {self.mesd_config.mesd_reweighting_lambda}")
        print(
            f"[MeSD]   - anneal/cutoff step: {self.mesd_config.mesd_reweighting_anneal_steps}/{self.mesd_config.mesd_teacher_disable_after_step}"
        )
        print(
            f"[MeSD]   - failure direction: {self.mesd_config.mesd_failure_direction_mode} (gamma={self.mesd_config.mesd_failure_direction_gamma}, warmup={self.mesd_config.mesd_failure_direction_warmup_steps}, top-k={self.mesd_config.mesd_failure_direction_topk})"
        )
        print(f"[MeSD]   - GRPO epsilon: {self.config.algorithm.grpo_eps}")
        print("[MeSD]   - policy loss: token ratios, clip 0.2/0.2, sequence mean, no dual clip")
        return

    @staticmethod
    def _non_tensor_row(batch: DataProto, index: int) -> dict[str, Any]:
        return {key: values[index] for key, values in batch.non_tensor_batch.items() if len(values) > index}

    def _augment_batch_with_mesd(
        self, batch: DataProto, metrics: dict[str, Any], timing_raw: dict[str, Any]
    ) -> DataProto:
        """Attach same-completion teacher views for the configured MeSD mode."""
        aggregation = self.mesd_config.mesd_teacher_aggregation
        teacher_active = mesd_teacher_enabled(
            disable_after_step=self.mesd_config.mesd_teacher_disable_after_step, step=self.global_step
        )
        effective_lambda = compute_mesd_mixing_lambda(
            initial_lambda=self.mesd_config.mesd_reweighting_lambda,
            anneal_steps=self.mesd_config.mesd_reweighting_anneal_steps,
            step=self.global_step,
        )
        if not teacher_active:
            effective_lambda = 0.0
        response_mask = batch.batch["response_mask"]
        device, dtype = (response_mask.device, response_mask.dtype)
        group_valid = batch.batch.get("mesd_group_valid_mask")
        if group_valid is None:
            group_valid = torch.ones(len(batch), dtype=dtype, device=device)
            batch.batch["mesd_group_valid_mask"] = group_valid
        group_valid_values = group_valid.detach().cpu().numpy() > 0.5
        direction_values = np.zeros(len(batch), dtype=np.float32)
        teacher_usable_values = np.zeros(len(batch), dtype=np.float32)
        batch.batch["mesd_direction_mask"] = torch.zeros(len(batch), dtype=dtype, device=device)
        batch.batch["mesd_teacher_usable_mask"] = torch.zeros(len(batch), dtype=dtype, device=device)
        if teacher_active:
            raw_prompts = batch.non_tensor_batch.get("raw_prompt")
            if raw_prompts is None:
                raise ValueError(
                    "MeSD requires dataset raw_prompt strings so hindsight can be injected without applying the chat template twice."
                )
            non_tensor_rows = [self._non_tensor_row(batch, index) for index in range(len(batch))]
            privileged_views = [build_mesd_privileged_views(row) for row in non_tensor_rows]
            with timer("mesd_teacher_prompt", timing_raw):
                teacher_gen = construct_mesd_multiview_teacher_prompts(
                    student_raw_prompts=raw_prompts,
                    privileged_infos=privileged_views,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    max_prompt_length=self.mesd_config.mesd_teacher_max_prompt_length,
                    multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                    image_min_pixels=self.config.data.image_min_pixels,
                    image_max_pixels=self.config.data.image_max_pixels,
                    video_min_pixels=self.config.data.video_min_pixels,
                    video_max_pixels=self.config.data.video_max_pixels,
                    video_fps=self.config.data.video_fps,
                    video_max_frames=self.config.data.video_max_frames,
                    cache_keys=batch.non_tensor_batch.get("uid"),
                )
            view_names = MESD_PRIVILEGED_VIEW_NAMES
            view_available = teacher_gen.batch["view_available"].to(
                device=batch.batch["response_mask"].device, dtype=torch.bool
            )
            if view_available.shape != (len(batch), len(view_names)):
                raise ValueError(
                    f"anchored_avsd prompt construction must return view_available with shape {(len(batch), len(view_names))}, got {tuple(view_available.shape)}."
                )
            for view_index, view_name in enumerate(view_names):
                teacher_input_ids, teacher_attention_mask, teacher_position_ids = _concat_prompt_and_response(
                    prompt_ids=teacher_gen.batch["input_ids"][:, view_index],
                    prompt_attention_mask=teacher_gen.batch["attention_mask"][:, view_index],
                    prompt_position_ids=teacher_gen.batch["position_ids"][:, view_index],
                    response_ids=batch.batch["responses"],
                    response_mask=batch.batch["response_mask"],
                )
                prefix = f"mesd_teacher_{view_name}"
                batch.batch[f"{prefix}_input_ids"] = teacher_input_ids
                batch.batch[f"{prefix}_attention_mask"] = teacher_attention_mask
                batch.batch[f"{prefix}_position_ids"] = teacher_position_ids
                metrics[f"mesd/view_{view_name}_coverage"] = float(
                    view_available[:, view_index].float().mean().item()
                )
            batch.batch["mesd_view_availability"] = view_available
            has_privileged_view = view_available.any(dim=1).detach().cpu().numpy()
            metrics["mesd/no_view_skip_fraction"] = float(1.0 - has_privileged_view.mean())
            vision_failed = batch.non_tensor_batch.get("vision_read_failed")
            if vision_failed is None:
                vision_available = np.ones(len(batch), dtype=bool)
            else:
                vision_available = np.array([not bool(value) for value in vision_failed], dtype=bool)
            target_values = np.logical_and(
                np.logical_and(vision_available, has_privileged_view), group_valid_values
            ).astype(np.float32)
            batch.batch["mesd_target_mask"] = torch.as_tensor(
                target_values, dtype=batch.batch["response_mask"].dtype, device=batch.batch["response_mask"].device
            )
            metrics["mesd/vision_skip_fraction"] = float(1.0 - vision_available.mean())
            metrics["mesd/target_active_fraction"] = float(target_values.mean())
            tasks = [str(row.get("task", "")) for row in non_tensor_rows]
            view_values = view_available.detach().cpu().numpy().astype(bool)
            for index, task in enumerate(tasks):
                if task == "General video QA MCQ":
                    teacher_usable_values[index] = float(view_values[index, 0])
                elif task == "temporal QA":
                    teacher_usable_values[index] = float(view_values[index, 1])
                elif task == "temporal QA (MCQ)":
                    teacher_usable_values[index] = float(view_values[index, 0] and view_values[index, 1])
            rkl_candidates = batch.batch.get("mesd_rkl_eligible_mask")
            if rkl_candidates is None:
                rkl_candidate_values = np.zeros(len(batch), dtype=bool)
            else:
                rkl_candidate_values = rkl_candidates.detach().cpu().numpy() > 0.5
            direction_values = (
                rkl_candidate_values & group_valid_values & vision_available & (teacher_usable_values > 0.5)
            ).astype(np.float32)
            batch.batch["mesd_teacher_usable_mask"] = torch.as_tensor(
                teacher_usable_values, dtype=dtype, device=device
            )
            batch.batch["mesd_direction_mask"] = torch.as_tensor(direction_values, dtype=dtype, device=device)
        direction_mode = self.mesd_config.mesd_failure_direction_mode
        warmup_steps = self.mesd_config.mesd_failure_direction_warmup_steps
        if not teacher_active:
            direction_gamma = 0.0
            direction_values.fill(0.0)
            batch.batch["mesd_direction_mask"] = torch.zeros(len(batch), dtype=dtype, device=device)
        elif warmup_steps > 0:
            direction_gamma = self.mesd_config.mesd_failure_direction_gamma * min(
                float(self.global_step) / float(warmup_steps), 1.0
            )
        else:
            direction_gamma = self.mesd_config.mesd_failure_direction_gamma
        batch.meta_info["loss_type"] = "mesd"
        batch.meta_info["mesd_config"] = {
            "mesd_enabled": True,
            "mesd_teacher_enabled": teacher_active,
            "mesd_teacher_aggregation": aggregation,
            "mesd_avsd_evidence_alpha": self.mesd_config.mesd_avsd_evidence_alpha,
            "mesd_avsd_epsilon": self.mesd_config.mesd_avsd_epsilon,
            "mesd_avsd_use_token_roles": self.mesd_config.mesd_avsd_use_token_roles,
            "mesd_lambda": effective_lambda,
            "mesd_topk": self.mesd_config.mesd_reweighting_topk,
            "mesd_topk_gamma": self.mesd_config.mesd_reweighting_topk_gamma,
            "mesd_include_sampled_token": self.mesd_config.mesd_include_sampled_token_in_topk,
            "mesd_weight_clip": self.mesd_config.mesd_reweighting_weight_clip,
            "mesd_teacher_update_rate": self.mesd_config.mesd_teacher_update_rate,
            "mesd_teacher_update_interval": self.mesd_config.mesd_teacher_update_interval,
            "mesd_policy_clip_low": self.mesd_config.mesd_policy_clip_low,
            "mesd_policy_clip_high": self.mesd_config.mesd_policy_clip_high,
            "mesd_failure_direction_mode": direction_mode,
            "mesd_failure_direction_gamma": direction_gamma,
            "mesd_failure_direction_topk": self.mesd_config.mesd_failure_direction_topk,
            "mesd_failure_target_epsilon": self.mesd_config.mesd_failure_target_epsilon,
            "global_step": self.global_step,
        }
        metrics["mesd/effective_lambda"] = effective_lambda
        metrics["mesd/teacher_enabled"] = float(teacher_active)
        metrics["mesd/pure_grpo"] = float(not teacher_active)
        metrics["mesd/direction_gamma"] = float(direction_gamma)
        metrics["mesd/teacher_usable_fraction"] = float(teacher_usable_values.mean())
        metrics["mesd/direction_active_fraction"] = float(direction_values.mean())
        return batch

    @staticmethod
    def _compute_mesd_advantage(batch: DataProto, *, eps: float) -> DataProto:
        """Compute released TRL/MeSD GRPO normalization with epsilon 1e-4."""
        advantages, returns = compute_grpo_advantage(
            token_level_rewards=batch.batch["token_level_rewards"],
            response_mask=batch.batch["response_mask"],
            index=batch.non_tensor_batch["uid"],
            eps=eps,
        )
        group_valid = batch.batch.get("mesd_group_valid_mask")
        if group_valid is not None:
            valid_tokens = group_valid.to(device=advantages.device, dtype=advantages.dtype).unsqueeze(-1)
            advantages = advantages * valid_tokens
            returns = returns * valid_tokens
        batch.batch["advantages"] = advantages
        batch.batch["returns"] = returns
        return batch

    def fit(self):
        self.logger = self._create_logger()
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None
        self._load_checkpoint()
        main_tqdm.update(self.global_step)
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return
        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1
            metrics, timing_raw = ({}, {})
            with timer("step", timing_raw):
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()
                self._balance_batch(batch, metrics=metrics)
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                with timer("reward", timing_raw):
                    reward_batch = batch.select(
                        batch_keys=["responses", "response_mask"],
                        non_tensor_batch_keys=[k for k in batch.non_tensor_batch if k != "multi_modal_data"],
                    )
                    reward_batch.meta_info.update(
                        {
                            "global_step": int(self.global_step),
                            "total_training_steps": int(self.training_steps),
                            "step_percent": float(self.global_step / max(self.training_steps, 1)),
                        }
                    )
                    reward_ref = self.reward_fn.compute_reward.remote(reward_batch)
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)
                with timer("adv", timing_raw):
                    reward_tensor, reward_metrics = ray.get(reward_ref)
                    batch.batch["token_level_scores"] = reward_tensor
                    if reward_metrics is None:
                        raise RuntimeError("MeSD requires synchronous reward metrics for absolute outcome routing.")
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()})
                    metrics.update(
                        _attach_mesd_outcome_tensors(batch, reward_metrics, rollout_n=self.config.worker.rollout.n)
                    )
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                    batch = self._compute_mesd_advantage(batch, eps=self.config.algorithm.grpo_eps)
                batch = self._augment_batch_with_mesd(batch, metrics, timing_raw)
                all_invalid = bool(
                    "mesd_group_valid_mask" in batch.batch and (batch.batch["mesd_group_valid_mask"].sum().item() == 0)
                )
                if all_invalid:
                    metrics["mesd/all_invalid_update_skip"] = 1.0
                else:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)
                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)
                    metrics["mesd/all_invalid_update_skip"] = 0.0
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and (self.global_step % self.config.trainer.val_freq == 0)
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)
                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))
            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)
        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()

    def _create_logger(self):
        """Create logger instance."""
        from ..utils.logger import Tracker

        return Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
