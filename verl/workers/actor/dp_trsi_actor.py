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
"""FSDP actor for Video-t-RSI: frozen Route-A teacher scoring and the RSI objective.

Teacher scoring (``compute_teacher_log_probs``) runs before the policy update
as a separate gradient-free pass.  For every evidence view and both frozen
snapshots (updated ``theta+`` and, when the correction is active, anchor
``theta-``) it evaluates float32 logits only at the positions that predict the
realized response tokens, builds the full-vocabulary teacher distribution of
that view and reads it at the realized token.  View distributions are mixed
in probability space.  Every FSDP rank executes the same sequence of teacher
forwards (global view flags are all-reduced), which keeps the collectives of
sharded modules aligned; errors found on one rank are all-reduced as well, so
every rank raises together instead of waiting in a collective.

The policy update (``update_policy``) optimizes
``task_coef * L_task + sd_coef * L_SD`` (plus ``ref_kl_coef * L_KL`` when the
optional KL to the initialization model is enabled; see
``verl.trainer.trsi.losses``), plus ``route_c_coef * L_C`` on the steps where
the Route C inner loop (``rsi.inner_loop=route_c``, package v6) set teacher
distributions at branch positions (``verl.trainer.trsi.route_c``).  For that
loop, teacher scoring also reads the teacher at the student's top candidates.
MeSD's success/failure routing and failure-only Direct RKL are intentionally
absent.  An advantage-sign reweighting exists only as ``rsi.sd_mode=reweight``
(package v5): the driver then scales the GRPO advantages by the teacher's
reward-anchored chunk factors before this update, which trains ``L_task``
alone (``sd_coef = 0``) and is unchanged.
"""

import math
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm

from ...protocol import DataProto, batch_collate
from ...trainer.trsi.losses import compute_rsi_policy_loss
from ...trainer.trsi.route_c import route_c_policy_loss
from ...trainer.trsi.teacher import RouteATeacherRule, mix_view_log_probs, route_a_token_log_probs
from ...trainer.trsi.torch_utils import (
    lm_head_logits_at,
    padded_response_logit_positions,
    rmpad_response_logit_positions,
)
from ...utils.py_functional import append_to_dict
from ..trsi_snapshots import reshard_fsdp_root
from .dp_actor import DataParallelPPOActor


try:
    from flash_attn.bert_padding import index_first_axis, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelTRSIActor", "TRSI_VIEW_NAMES", "reshard_fsdp_root"]

TRSI_VIEW_NAMES = ("answer", "temporal", "spatial")
"""Evidence views A, AT and AS (MeSD's prompt builder order)."""


def _global_max_flag(value: bool, device: torch.device) -> bool:
    flag = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def _raise_on_every_rank(error: Optional[str], device: torch.device, what: str) -> None:
    """Raise on all ranks together when any rank failed, instead of leaving the others in a collective."""

    if _global_max_flag(error is not None, device):
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        raise RuntimeError(f"Video-t-RSI {what} failed (rank {rank}: {error or 'error reported by another rank'}).")


def _global_sum(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().to(dtype=torch.float32).clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


class DataParallelTRSIActor(DataParallelPPOActor):
    def __init__(
        self,
        config,
        actor_module: torch.nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        updated_teacher: Optional[torch.nn.Module] = None,
        anchor_teacher: Optional[torch.nn.Module] = None,
    ):
        super().__init__(config=config, actor_module=actor_module, actor_optimizer=actor_optimizer)
        self.updated_teacher = updated_teacher
        self.anchor_teacher = anchor_teacher

    # ------------------------------------------------------------------
    # Teacher scoring
    # ------------------------------------------------------------------
    def _teacher_response_logits(
        self,
        module: torch.nn.Module,
        micro_batch: dict[str, Any],
        *,
        fp32: bool,
        vocab_chunk: int,
    ) -> list[torch.Tensor]:
        """Per-row ``[L_i, V]`` logits predicting the realized response tokens."""

        if self.config.ulysses_size > 1:
            raise NotImplementedError("Video-t-RSI teacher scoring currently requires worker.actor.ulysses_size=1.")
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        lengths = micro_batch["response_mask"].sum(dim=-1).long()
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
            positions, counts = rmpad_response_logit_positions(attention_mask, response_length, lengths)
            with lm_head_logits_at(module, positions, fp32=fp32, vocab_chunk=vocab_chunk):
                output = module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )
            reshard_fsdp_root(module)
            logits = output.logits.squeeze(0)
            return list(torch.split(logits, counts, dim=0))

        positions = padded_response_logit_positions(seqlen, response_length, device=input_ids.device)
        with lm_head_logits_at(module, positions, fp32=fp32, vocab_chunk=vocab_chunk):
            output = module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
        reshard_fsdp_root(module)
        logits = output.logits
        return [logits[row, : int(lengths[row])] for row in range(batch_size)]

    @torch.no_grad()
    def compute_teacher_log_probs(self, data: DataProto) -> dict[str, torch.Tensor]:
        """Return realized-token teacher log probabilities and per-row diagnostics."""

        if self.updated_teacher is None or self.anchor_teacher is None:
            raise RuntimeError("Video-t-RSI requires separate frozen updated and anchor teacher modules.")
        self.updated_teacher.eval()
        self.anchor_teacher.eval()
        meta = data.meta_info["trsi_teacher"]
        rule = RouteATeacherRule(
            alpha=float(meta["alpha"]),
            correction_clip=None if meta["correction_clip"] is None else float(meta["correction_clip"]),
            base_mix=float(meta["base_mix"]),
            temperature=float(meta["temperature"]),
            use_anchor=bool(meta["use_anchor"]),
        )
        views = tuple(meta["views"])
        fp32 = bool(meta.get("fp32_logits", True))
        vocab_chunk = int(meta.get("vocab_chunk", 16384))
        position_chunk = int(meta.get("position_chunk", 256))
        # Route C (package v6): also read the teacher at the student's top-k candidates (0: not this step).
        candidates = int(meta["candidates"])
        if candidates < 0:
            raise ValueError(f"trsi_teacher candidates must be >= 0, got {candidates}.")

        select_keys = ["responses", "response_mask", "trsi_view_mask"]
        if candidates:
            select_keys.append("trsi_candidate_ids")
        for view in views:
            select_keys.extend(
                [f"trsi_teacher_{view}_input_ids", f"trsi_teacher_{view}_attention_mask", f"trsi_teacher_{view}_position_ids"]
            )
        selected = data.select(select_keys, ["multi_modal_inputs"])
        micro_batches = selected.split(self.config.micro_batch_size_per_device_for_experience)
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Video-t-RSI teacher", position=1)

        outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
        for micro_batch in micro_batches:
            responses = micro_batch.batch["responses"]
            batch_size, response_length = responses.shape
            lengths = micro_batch.batch["response_mask"].sum(dim=-1).long()
            view_mask = micro_batch.batch["trsi_view_mask"].to(torch.bool)
            if tuple(view_mask.shape) != (batch_size, len(views)):
                raise ValueError(f"trsi_view_mask must have shape {(batch_size, len(views))}.")
            device = responses.device
            # A layout error on one rank must stop every rank before the first collective of this micro-batch.
            layout_error = None
            candidate_ids = micro_batch.batch["trsi_candidate_ids"] if candidates else None
            if candidate_ids is not None and tuple(candidate_ids.shape) != (batch_size, response_length, candidates):
                layout_error = (
                    f"trsi_candidate_ids has shape {tuple(candidate_ids.shape)}, "
                    f"expected {(batch_size, response_length, candidates)}"
                )
            if layout_error is None and self.config.padding_free:
                for view in views:
                    try:
                        rmpad_response_logit_positions(
                            micro_batch.batch[f"trsi_teacher_{view}_attention_mask"], response_length, lengths
                        )
                    except ValueError as error:
                        layout_error = f"view {view}: {error}"
                        break
            _raise_on_every_rank(layout_error, device, "teacher prompt layout check")
            local_error: Optional[str] = None
            view_log_q = torch.zeros(batch_size, len(views), response_length, dtype=torch.float32, device=device)
            view_log_base = torch.zeros_like(view_log_q)
            view_log_candidates = (
                torch.zeros(batch_size, len(views), response_length, candidates, dtype=torch.float32, device=device)
                if candidates
                else None
            )
            sums = {
                name: torch.zeros(batch_size, dtype=torch.float32, device=device)
                for name in ("kl", "clip", "entropy", "positions")
            }
            max_ratio = torch.zeros(batch_size, dtype=torch.float32, device=device)
            for view_index, view in enumerate(views):
                # Identical forward schedules on every FSDP rank.
                if not _global_max_flag(bool(view_mask[:, view_index].any().item()), device):
                    continue
                view_batch = {
                    "input_ids": micro_batch.batch[f"trsi_teacher_{view}_input_ids"],
                    "attention_mask": micro_batch.batch[f"trsi_teacher_{view}_attention_mask"],
                    "position_ids": micro_batch.batch[f"trsi_teacher_{view}_position_ids"],
                    "responses": responses,
                    "response_mask": micro_batch.batch["response_mask"],
                }
                if "multi_modal_inputs" in micro_batch.non_tensor_batch:
                    view_batch["multi_modal_inputs"] = micro_batch.non_tensor_batch["multi_modal_inputs"]
                updated_rows = self._teacher_response_logits(
                    self.updated_teacher, view_batch, fp32=fp32, vocab_chunk=vocab_chunk
                )
                anchor_rows = None
                if rule.correction_active:
                    anchor_rows = self._teacher_response_logits(
                        self.anchor_teacher, view_batch, fp32=fp32, vocab_chunk=vocab_chunk
                    )
                for row in range(batch_size):
                    length = int(lengths[row])
                    if local_error is not None or length == 0 or not bool(view_mask[row, view_index]):
                        continue
                    try:
                        scores = route_a_token_log_probs(
                            updated_rows[row],
                            responses[row, :length],
                            rule,
                            None if anchor_rows is None else anchor_rows[row],
                            position_chunk=position_chunk,
                            candidate_ids=None if candidate_ids is None else candidate_ids[row, :length],
                        )
                    except Exception as error:  # noqa: BLE001 - re-raised on every rank below
                        # Keep running the remaining forwards so every rank reaches the check below.
                        local_error = f"view {view}, row {row}: {type(error).__name__}: {error}"
                        continue
                    view_log_q[row, view_index, :length] = scores.log_q
                    view_log_base[row, view_index, :length] = scores.log_base
                    if view_log_candidates is not None:
                        view_log_candidates[row, view_index, :length] = scores.candidate_log_q
                    sums["kl"][row] += scores.kl_to_base.sum()
                    sums["clip"][row] += scores.clip_fraction.sum()
                    sums["entropy"][row] += scores.entropy.sum()
                    sums["positions"][row] += float(length)
                    max_ratio[row] = torch.maximum(max_ratio[row], scores.max_log_ratio.max())
                del updated_rows, anchor_rows
            _raise_on_every_rank(local_error, device, "teacher scoring")
            log_q, view_count = mix_view_log_probs(view_log_q, view_mask)
            log_base, _ = mix_view_log_probs(view_log_base, view_mask)
            denom = sums["positions"].clamp_min(1.0)
            outputs["trsi_teacher_log_probs"].append(log_q)
            outputs["trsi_teacher_base_log_probs"].append(log_base)
            outputs["trsi_teacher_view_count"].append(view_count.to(torch.float32))
            outputs["trsi_teacher_kl_to_base"].append(sums["kl"] / denom)
            outputs["trsi_teacher_clip_fraction"].append(sums["clip"] / denom)
            outputs["trsi_teacher_entropy"].append(sums["entropy"] / denom)
            outputs["trsi_teacher_max_log_ratio"].append(max_ratio)
            if view_log_candidates is not None:
                # The same arithmetic view mixture, at every candidate ([B, R, k]).
                outputs["trsi_teacher_candidate_log_probs"].append(mix_view_log_probs(view_log_candidates, view_mask)[0])
        return {name: torch.cat(values, dim=0) for name, values in outputs.items()}

    # ------------------------------------------------------------------
    # Candidate log-probabilities of the policy (Route C, package v6)
    # ------------------------------------------------------------------
    @contextmanager
    def _kept_logits(self) -> Iterator[list[torch.Tensor]]:
        """Keep the logits the base forward hands to ``log_probs_from_logits`` (already divided by the temperature):
        ``[total_nnz, V]`` padding-free, ``[B, R, V]`` padded.  Reading them adds no forward of its own."""

        if self.config.ulysses_size > 1:
            raise NotImplementedError("The Route C inner loop currently requires worker.actor.ulysses_size=1.")
        kept: list[torch.Tensor] = []
        compute = self.log_probs_from_logits

        def keep(*args, **kwargs):
            kept.append(kwargs["logits"] if "logits" in kwargs else args[0])
            return compute(*args, **kwargs)

        self.log_probs_from_logits = keep
        try:
            yield kept
        finally:
            self.log_probs_from_logits = compute
        if len(kept) != 1:
            raise RuntimeError(f"Expected one logits tensor from the policy forward, got {len(kept)}.")

    def _response_logit_index(
        self, micro_batch: dict[str, Any], rows: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """Index into the kept logits of the logit that predicts response token ``positions[i]`` of row ``rows[i]``."""

        response_length = micro_batch["responses"].size(-1)
        if not self.config.padding_free:
            return (rows, positions)
        lengths = micro_batch["response_mask"].sum(dim=-1).long()
        flat, counts = rmpad_response_logit_positions(micro_batch["attention_mask"], response_length, lengths)
        starts = torch.tensor([0] + counts[:-1], dtype=torch.long, device=flat.device).cumsum(dim=0)
        return (flat[starts[rows.to(flat.device)] + positions.to(flat.device)],)

    @torch.no_grad()
    def compute_log_prob_with_candidates(
        self, data: DataProto, *, candidates: int, position_chunk: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``compute_log_prob`` plus the policy's ``candidates`` most probable tokens at every response position.

        Returns ``(log_probs [B, R], candidate_ids [B, R, k], candidate_log_probs [B, R, k])`` (zeros after each
        response).  The candidates come from the same forward as the old log-probabilities, so the gap, the Route C
        target and the update all read one numerical policy.
        """

        if candidates <= 0 or position_chunk <= 0:
            raise ValueError(f"candidates and position_chunk must be positive, got {candidates}, {position_chunk}.")
        if self.config.dynamic_batching:
            raise NotImplementedError("Route C candidate scoring does not support worker.actor.dynamic_batching.")
        self.actor_module.eval()
        temperature = data.meta_info["temperature"]
        data = data.select(["input_ids", "attention_mask", "position_ids", "responses", "response_mask"], ["multi_modal_inputs"])
        micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs and candidates", position=1)
        outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            responses = model_inputs["responses"]
            batch_size, response_length = responses.shape
            valid = model_inputs["response_mask"].to(torch.bool)
            rows, positions = valid.nonzero(as_tuple=True)
            # A layout error on one rank must stop every rank before the forward's collectives.
            error, index = None, None
            try:
                index = self._response_logit_index(model_inputs, rows, positions)
            except ValueError as exc:
                error = str(exc)
            _raise_on_every_rank(error, responses.device, "Route C candidate layout check")
            with self._kept_logits() as kept:
                log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            logits = kept[0]
            ids = torch.zeros(batch_size, response_length, candidates, dtype=torch.long, device=responses.device)
            values = torch.zeros(batch_size, response_length, candidates, dtype=torch.float32, device=responses.device)
            for start in range(0, rows.numel(), position_chunk):
                end = min(start + position_chunk, rows.numel())
                chunk = logits[tuple(part[start:end] for part in index)].float()  # [c, V]
                top = chunk.topk(candidates, dim=-1)
                normalizer = torch.logsumexp(chunk, dim=-1, keepdim=True)
                ids[rows[start:end], positions[start:end]] = top.indices.to(ids.device)
                values[rows[start:end], positions[start:end]] = (top.values - normalizer).to(values.device)
                del chunk, top, normalizer
            del logits, kept
            outputs["log_probs"].append(log_probs)
            outputs["ids"].append(ids)
            outputs["values"].append(values)
        return tuple(torch.cat(outputs[name], dim=0) for name in ("log_probs", "ids", "values"))

    # ------------------------------------------------------------------
    # Policy update
    # ------------------------------------------------------------------
    def _forward_with_candidates(
        self, micro_batch: dict[str, Any], *, temperature: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Response log-probs ``[B, R]`` and the policy's log-probs of ``trsi_rc_candidates`` ``[B, k]`` at
        response position ``trsi_rc_position`` (Route C; rows outside ``trsi_rc_row_mask`` get zeros).

        Both come from one forward (``_kept_logits``), so the candidate log-probs and the response log-probs share
        one graph and one full-vocabulary normalization.  The candidate rows are copied out of the logits (advanced
        indexing), which leaves the logits themselves untouched for flash-attention's in-place cross-entropy
        backward.
        """

        positions = micro_batch["trsi_rc_position"].long()
        candidates = micro_batch["trsi_rc_candidates"].long()
        rows = micro_batch["trsi_rc_row_mask"].to(torch.bool)
        responses = micro_batch["responses"]
        batch_size = responses.size(0)
        # A bad branch on one rank must stop every rank before the forward's collectives, not leave them waiting.
        error, index, chosen = None, None, rows.nonzero(as_tuple=True)[0]
        if tuple(positions.shape) != (batch_size,) or candidates.ndim != 2 or candidates.size(0) != batch_size:
            error = "trsi_rc_position must be [B] and trsi_rc_candidates [B, k]"
        else:
            lengths = micro_batch["response_mask"].sum(dim=-1).long()
            if bool((positions[chosen] < 0).any()) or bool((positions[chosen] >= lengths[chosen]).any()):
                error = "every branched row needs a branch position inside its response"
            else:
                try:
                    index = self._response_logit_index(micro_batch, chosen, positions[chosen])
                except ValueError as exc:
                    error = str(exc)
        _raise_on_every_rank(error, responses.device, "Route C branch check")
        with self._kept_logits() as kept:
            log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
        logits = kept[0]
        selected = logits[index]  # [n, V], a copy
        selected = torch.log_softmax(selected.float(), dim=-1).gather(-1, candidates[chosen].to(selected.device))
        out = torch.zeros(batch_size, candidates.size(1), dtype=selected.dtype, device=selected.device)
        return log_probs, out.index_put((chosen.to(selected.device),), selected)

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()
        temperature = data.meta_info["temperature"]
        loss_cfg = data.meta_info["trsi_loss"]
        task_coef = float(loss_cfg["task_coef"])
        sd_coef = float(loss_cfg["sd_coef"])
        task_clip = (float(loss_cfg["task_clip_low"]), float(loss_cfg["task_clip_high"]))
        sd_clip = (float(loss_cfg["sd_clip_low"]), float(loss_cfg["sd_clip_high"]))
        # Keys absent from older drivers mean the original behaviour (sequence mean, no rollout correction).
        task_loss_agg = str(loss_cfg.get("task_loss_agg", "seq_mean_token_mean"))
        use_rollout_is = bool(loss_cfg.get("rollout_is", False))
        ref_kl_coef = float(loss_cfg.get("ref_kl_coef", 0.0))
        # Package v6: the Route C term (0 on steps without branch targets).
        route_c_coef = float(loss_cfg["route_c_coef"])
        if not (math.isfinite(route_c_coef) and route_c_coef >= 0.0):
            raise ValueError(f"route_c_coef must be finite and non-negative, got {route_c_coef}.")

        select_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "responses",
            "response_mask",
            "old_log_probs",
            "advantages",
            "trsi_task_row_mask",
        ]
        if sd_coef > 0.0:
            select_keys.extend(["trsi_sd_advantages", "trsi_sd_token_weights", "trsi_sd_row_mask"])
        if use_rollout_is:
            select_keys.append("trsi_rollout_is_weights")
        if ref_kl_coef > 0.0:
            select_keys.append("ref_log_probs")
        if route_c_coef > 0.0:
            select_keys.extend(["trsi_rc_position", "trsi_rc_candidates", "trsi_rc_target", "trsi_rc_row_mask"])
        mini_batches = data.select(select_keys, ["multi_modal_inputs"]).split(self.config.global_batch_size_per_device)
        metrics = defaultdict(list)
        # A rollout batch of k mini-batches makes up to k optimizer steps (rollout_batch_size > global_batch_size).
        applied_steps = 0
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)
            for mini_batch in mini_batches:
                task_rows = _global_sum(mini_batch.batch["trsi_task_row_mask"].sum())
                sd_rows = (
                    _global_sum(mini_batch.batch["trsi_sd_row_mask"].sum())
                    if sd_coef > 0.0
                    else torch.zeros((), dtype=torch.float32)
                )
                rc_rows = (
                    _global_sum(mini_batch.batch["trsi_rc_row_mask"].sum())
                    if route_c_coef > 0.0
                    else torch.zeros((), dtype=torch.float32)
                )
                effective_task = task_coef > 0.0 and float(task_rows.item()) > 0.0
                effective_sd = sd_coef > 0.0 and float(sd_rows.item()) > 0.0
                effective_rc = route_c_coef > 0.0 and float(rc_rows.item()) > 0.0
                task_tokens = None
                if (effective_task and task_loss_agg == "token_mean") or (
                    ref_kl_coef > 0.0 and (effective_task or effective_sd or effective_rc)
                ):
                    # Response tokens of the valid rows of this optimizer mini-batch on all ranks (the token-mean
                    # task loss and the KL penalty share the valid rows of the GRPO groups; a mini-batch with Route C
                    # rows alone has none, and its KL term is 0).
                    row_mask = mini_batch.batch["trsi_task_row_mask"].to(torch.float32)
                    task_tokens = float(
                        _global_sum(
                            (mini_batch.batch["response_mask"].to(torch.float32) * row_mask.unsqueeze(-1)).sum()
                        ).item()
                    )
                if not (effective_task or effective_sd or effective_rc):
                    append_to_dict(metrics, {"trsi/empty_minibatch_skip": 1.0})
                    continue
                append_to_dict(metrics, {"trsi/empty_minibatch_skip": 0.0})
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)
                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    if effective_rc:  # the same forward on every rank (a global decision)
                        log_probs, rc_log_probs = self._forward_with_candidates(model_inputs, temperature=temperature)
                    else:
                        log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                    result = compute_rsi_policy_loss(
                        log_probs=log_probs,
                        old_log_probs=model_inputs["old_log_probs"],
                        response_mask=model_inputs["response_mask"],
                        task_advantages=model_inputs["advantages"],
                        task_row_mask=model_inputs["trsi_task_row_mask"],
                        sd_advantages=model_inputs.get("trsi_sd_advantages"),
                        sd_token_weights=model_inputs.get("trsi_sd_token_weights"),
                        sd_row_mask=model_inputs.get("trsi_sd_row_mask"),
                        task_coef=task_coef if effective_task else 0.0,
                        sd_coef=sd_coef if effective_sd else 0.0,
                        task_global_rows=float(task_rows.item()),
                        sd_global_rows=float(sd_rows.item()),
                        world_size=self.world_size,
                        task_clip=task_clip,
                        sd_clip=sd_clip,
                        task_loss_agg=task_loss_agg,
                        task_global_tokens=task_tokens,
                        rollout_is_weights=model_inputs.get("trsi_rollout_is_weights") if use_rollout_is else None,
                        ref_kl_coef=ref_kl_coef,
                        ref_log_probs=model_inputs.get("ref_log_probs") if ref_kl_coef > 0.0 else None,
                        kl_row_mask=model_inputs["trsi_task_row_mask"] if ref_kl_coef > 0.0 else None,
                        kl_global_tokens=task_tokens if ref_kl_coef > 0.0 else None,
                    )
                    loss = result.loss
                    rc_row_kl = None
                    if effective_rc:
                        branched = model_inputs["trsi_rc_row_mask"].to(torch.bool)
                        rc_term, rc_row_kl = route_c_policy_loss(
                            rc_log_probs[branched],
                            model_inputs["trsi_rc_target"][branched],
                            global_rows=float(rc_rows.item()),
                            world_size=self.world_size,
                        )
                        loss = loss + route_c_coef * rc_term
                        # Not raised (see compute_rsi_policy_loss): a non-finite loss skips the step on every rank.
                        result.metrics["loss_nonfinite"] = float(not bool(torch.isfinite(loss.detach()).item()))
                    loss.backward()
                    mask = model_inputs["response_mask"].to(torch.float32)
                    batch_metrics = {
                        "actor/pg_loss": float(loss.detach().item()),
                        "actor/entropy_loss": float(
                            ((-log_probs.detach() * mask).sum() / mask.sum().clamp_min(1.0)).item()
                        ),
                    }
                    if effective_task:
                        batch_metrics["trsi/task_loss"] = float(
                            result.task_row_loss.sum().item() / max(result.task_loss_count, 1.0)
                        )
                    if effective_sd:
                        valid = model_inputs["trsi_sd_row_mask"].to(torch.float32)
                        batch_metrics["trsi/sd_loss"] = float(
                            (result.sd_row_loss.sum() / valid.sum().clamp_min(1.0)).item()
                        )
                    if rc_row_kl is not None and rc_row_kl.numel():
                        batch_metrics["trsi/route_c_loss"] = float(rc_row_kl.mean().item())
                    for key, value in result.metrics.items():
                        batch_metrics[f"actor/{key}" if key == "ppo_kl" else f"trsi/{key}"] = value
                    append_to_dict(metrics, batch_metrics)
                grad_norm = self._optimizer_step()
                applied = bool(torch.isfinite(grad_norm).item())
                applied_steps += int(applied)
                append_to_dict(
                    metrics,
                    {
                        "actor/grad_norm": float(grad_norm.detach().item()),
                        "trsi/update_applied": float(applied),
                    },
                )
        # One value per rank (identical on all ranks): the number of optimizer steps this call applied.
        metrics["trsi/optimizer_steps"] = [float(applied_steps)]
        return metrics
