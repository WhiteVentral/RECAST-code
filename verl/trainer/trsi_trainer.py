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
"""Ray driver for Video-t-RSI (token-level recursive self-improvement of the teacher).

One rollout step:

1. apply the scheduled teacher-state events (anchor / updated snapshots);
2. generate ``rollout.n`` fresh responses per prompt with the current student (vLLM);
3. compute raw task rewards (original STGR-RL reward), old-policy log probabilities,
   and GRPO advantages using the separate effective rewards (v7: true length
   truncations receive zero training reward);
4. in teacher-using phases, build the A / AT / AS evidence prompts, score the
   *current* responses with the frozen teacher version at the same prefixes,
   partition responses into semantic chunks, and compute chunk credit; with
   ``rsi.sd_mode=reweight`` (package v5) the chunk credit then reweights the
   GRPO advantages (``credit.reward_anchored_factors``);
5. with ``rsi.inner_loop=route_c`` (package v6), branch every distilled row
   once at its least absorbed chunk position, sample continuations of the
   student's top candidates there and score them with the same effective training reward, and set
   a reward-tilted teacher distribution at that position
   (``verl.trainer.trsi.route_c``);
6. update the student with ``task_coef * L_task + sd_coef * L_SD`` (reweight:
   ``L_task`` with the reweighted advantages, ``sd_coef = 0``), plus
   ``route_c_coef * L_C`` at the branch positions.

The loop closes because snapshots taken from the updated student define the
next teacher version, which subsequent student training consumes (method
sections 1, 13 and 15).

Run health (``verl.trainer.trsi.health``): every step records the rollouts'
format rate, response lengths and sampled-token NLL; the collapse guard stops
a run whose rollouts stay unhealthy (``patience`` of the last ``window``
steps), and the teacher health gate keeps a degraded task phase from becoming
the next teacher.  Rollout requests get
per-step seeds, and with ``worker.rollout.calculate_log_probs`` the
vLLM/FSDP mismatch is recorded (``verl.trainer.trsi.rollout_correction``).
"""

import json
import math
import os
import time
import uuid
from dataclasses import asdict
from typing import Any, Optional

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm

from ..protocol import DataProto
from ..utils import torch_functional as VF
from ..utils.checkpoint.shards import check_rank_shards
from ..utils.py_functional import timer
from ..utils.rollout_dispatch import ROLLOUT_SEED_KEY, rollout_request_seeds
from ..utils.seqlen_balancing import (
    log_seqlen_unbalance,
    micro_batch_balanced_order,
    minibatch_group_order,
    synchronized_token_cost,
)
from .core_algos import compute_grpo_advantage
from .mesd_algos import MESD_PRIVILEGED_VIEW_NAMES, build_mesd_privileged_views
from .mesd_prompts import construct_mesd_multiview_teacher_prompts
from .mesd_trainer import _concat_prompt_and_response
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from .ray_trainer import RayTrainerBase
from .trsi import TRSI_SCHEMA_VERSION
from .trsi.chunks import ChunkingConfig, TokenTextAligner
from .trsi.credit import reward_anchored_factors, reweight_metrics
from .trsi.health import (
    CollapseGuard,
    HealthThresholds,
    PhaseHealth,
    TrainingCollapseError,
    format_steps,
    phase_degradation,
    rollout_health,
    unhealthy_reasons,
)
from .trsi.losses import reference_kl_k3
from .trsi.rollout_correction import rollout_mismatch_metrics, truncated_is_weights
from .trsi.training_reward import (
    effective_training_rewards,
    metadata_to_arrays,
    terminal_metadata,
    termination_flags,
)
from .trsi.route_c import (
    candidate_gap,
    candidate_values,
    continuation_layout,
    continuation_requests,
    continuation_responses,
    prompt_digest,
    route_c_seed_offset,
    route_c_targets,
    select_branches,
)
from .trsi.schedule import (
    EVENT_INIT_TEACHER,
    EVENT_REFRESH_TEACHER,
    EVENT_SNAPSHOT_ANCHOR,
    EVENT_SNAPSHOT_UPDATED,
    PHASE_DISTILL,
    RSISchedule,
    StepPlan,
    TeacherVersion,
)
from .trsi.targets import CreditConfig, build_sd_targets
from .trsi.teacher import RouteATeacherRule
from .trsi.verify import VerificationConfig
from .trsi_records import (
    COLLAPSE_MARKER,
    COMPLETION_MARKER,
    StepRecorder,
    combine_fingerprints,
    completion_payload,
    correction_reason,
    ids_digest,
    metric_groups,
    write_json_atomic,
)


TRSI_STATE_FILE = "trsi_state.json"
# Per-rank peak memory of the training stages (FSDPTRSIWorker.update_actor), reduced by max over the ranks.
PEAK_MEMORY_METRICS = (
    "perf/experience_peak_allocated_gb",
    "perf/update_peak_allocated_gb",
    "perf/train_peak_reserved_gb",
)
TEACHER_SHARD_PREFIXES = ("trsi_updated", "trsi_anchor")
# Settings that only change memory use, logging, sampling noise or when a run is stopped may differ on resume.
_RESUME_TOLERANT_KEYS = frozenset(
    {
        "teacher_vocab_chunk",
        "teacher_position_chunk",
        "log_chunk_examples",
        "log_chunk_interval",
        "record_state_fingerprints",
        "rollout_seed_mode",
        "health_min_format_rate",
        "health_min_response_length",
        "health_max_truncated_fraction",
        "health_ref_kl_jump_ratio",
        "health_ref_kl_jump_floor",
        "health_ref_kl_window",
        "guard_patience",
        "guard_window",
        "dump_rollouts_interval",
        "dump_rollouts_rows",
        "resume_allow_changes",
        "ref_kl_monitor",
    }
)
# Values of protocol settings added after the first release that reproduce the behaviour of a checkpoint
# written before them (its trsi_state.json does not record them).
_PROTOCOL_LEGACY_VALUES = {
    "task_loss_agg": "seq_mean_token_mean",
    "rollout_correction": "none",
    "rollout_correction_cap": 2.0,
    "teacher_health_gate": False,
    "teacher_health_window": 5,
    "teacher_health_max_format_drop": 0.1,
    "teacher_health_max_length_drop": 0.5,
    "ref_kl_coef": 0.0,
    "task_adv_std_floor": 0.0,
    # Added in package v5.
    "sd_mode": "surrogate",
    # Without effect under sd_mode=surrogate: the shipped values, so the sd_mode allowance covers the change.
    "reweight_clip": 0.2,
    "reweight_center": "row",
    "teacher_health_max_length_rise": None,
    "teacher_health_max_truncation_rise": None,
    # Added in package v6.
    "inner_loop": "none",
    # Without effect under inner_loop=none: the shipped values, so the inner_loop allowance covers the change.
    "route_c_candidates": 4,
    "route_c_continuations": 2,
    "route_c_min_gap": 0.05,
    "route_c_min_coverage": 0.5,
    "route_c_value_clip": 1.0,
    "route_c_kappa": 1.0,
    "route_c_coef": 1.0,
    # Added in package v7; changing this changes the training reward protocol.
    "zero_truncated_rewards": False,
}


def group_std_floor_scale(scores: torch.Tensor, uids: Any, *, floor: float, eps: float) -> torch.Tensor:
    """Per-row factor (std + eps) / (max(std, floor) + eps) of each row's GRPO group (torch.std, as GRPO)."""

    uids = np.asarray(uids, dtype=object)
    scale = torch.ones(len(uids), dtype=torch.float32)
    values = scores.detach().float().cpu()
    for uid in dict.fromkeys(uids.tolist()):
        rows = torch.as_tensor(np.flatnonzero(uids == uid))
        std = float(torch.std(values[rows]).item()) if len(rows) > 1 else 0.0
        scale[rows] = (std + eps) / (max(std, float(floor)) + eps)
    return scale


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def compute_group_validity(
    reward_metrics: dict[str, list[float]], uids: np.ndarray, rollout_n: int
) -> tuple[np.ndarray, dict[str, float]]:
    """Prompt-level system validity (media and ground truth usable); not task success."""

    batch_size = len(uids)
    row_valid = np.asarray(reward_metrics.get("outcome_row_valid", [1.0] * batch_size), dtype=np.float32) > 0.5
    vision_skip = np.asarray(reward_metrics.get("vision_skip", [0.0] * batch_size), dtype=np.float32) > 0.5
    if row_valid.shape != (batch_size,) or vision_skip.shape != (batch_size,):
        raise ValueError("Reward validity metrics must have one value per rollout row.")
    usable = row_valid & ~vision_skip
    group_valid = np.zeros(batch_size, dtype=bool)
    uids = np.asarray(uids, dtype=object)
    for uid in dict.fromkeys(uids.tolist()):
        indices = np.flatnonzero(uids == uid)
        if indices.size != int(rollout_n):
            raise ValueError(f"Rollout group {uid!r} has {indices.size} rows; expected {rollout_n}.")
        group_valid[indices] = bool(usable[indices].all())
    return group_valid, {"trsi/group_valid_fraction": float(group_valid.mean()) if batch_size else 0.0}


class RayVideoTRSITrainer(RayTrainerBase):
    """GRPO task learning + chunk-credit self-distillation with recursive Route-A teachers."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.rsi = self.config.rsi
        # global_batch_size divides rollout_batch_size (VideoTRSITrainingConfig.post_init): each rollout step makes
        # rollout_batch_size / global_batch_size optimizer steps; the base configs make one.
        self.schedule = RSISchedule(
            mode=self.rsi.schedule,
            task_steps=self.rsi.task_steps_per_round,
            distill_steps=self.rsi.distill_steps_per_round,
            stage_steps=self.rsi.joint_stage_steps,
            sd_coef=self.rsi.sd_coef,
            alpha=self.rsi.extrapolation_alpha if self.rsi.route == "A" else 0.0,
            alpha_decay=self.rsi.extrapolation_alpha_decay,
            alpha_min=self.rsi.extrapolation_alpha_min if self.rsi.route == "A" else 0.0,
            min_task_signal_steps=self.rsi.min_task_signal_steps,
            total_steps=self.training_steps,
            sd_mode=self.rsi.sd_mode,
        )
        self.chunk_cfg = ChunkingConfig(
            min_tokens=self.rsi.chunk_min_tokens,
            max_tokens=self.rsi.chunk_max_tokens,
            fallback_tokens=self.rsi.chunk_fallback_tokens,
        )
        self.credit_cfg = CreditConfig(
            horizon=self.rsi.credit_horizon, discount=self.rsi.credit_discount, clip=self.rsi.credit_clip
        )
        self.verify_cfg = (
            VerificationConfig(
                time_tolerance=self.rsi.verify_time_tolerance, supported_iou=self.rsi.verify_supported_iou
            )
            if self.rsi.verify_fields
            else None
        )
        self.aligner = TokenTextAligner(self.tokenizer)
        self.view_indices = [MESD_PRIVILEGED_VIEW_NAMES.index(view) for view in self.rsi.views]
        self._distill_steps_seen = 0
        self.health_thresholds = HealthThresholds(
            min_format_rate=self.rsi.health_min_format_rate,
            min_response_length=self.rsi.health_min_response_length,
            max_truncated_fraction=self.rsi.health_max_truncated_fraction,
            ref_kl_jump_ratio=self.rsi.health_ref_kl_jump_ratio,
            ref_kl_jump_floor=float(self.rsi.health_ref_kl_jump_floor),
        )
        self.guard = CollapseGuard(
            patience=int(self.rsi.guard_patience),
            window=int(self.rsi.guard_window),
            history_size=max(20, int(self.rsi.health_ref_kl_window), int(self.rsi.guard_window)),
        )
        self.phase_health = PhaseHealth()
        self._rollout_calls_in_step = 0
        # Route C (package v6): whether this step's rollout keeps the vLLM prompts, and the rows' branch details
        # for the rollout dumps.
        self._route_c_this_step = False
        self._route_c_rows: dict[int, dict[str, Any]] = {}
        self._protocol_changes: dict[str, dict[str, Any]] = {}
        self._resumed_from_step: Optional[int] = None  # set by fit() after loading a checkpoint

        rollout = self.config.worker.rollout
        print("[Video-t-RSI] token-level recursive teacher updates (Route %s)" % self.rsi.route)
        print(f"[Video-t-RSI]   schedule: {self.schedule.config_dict()}")
        if self.rsi.sd_mode == "reweight":
            credit = "G - mean_row(G)" if self.rsi.reweight_center == "row" else "G"
            print(
                "[Video-t-RSI]   distillation steps: GRPO with reward-anchored chunk reweighting, factors "
                f"clip(exp(sign(A) * ({credit})), {1.0 - float(self.rsi.reweight_clip):g}, "
                f"{1.0 + float(self.rsi.reweight_clip):g}) (rsi.sd_mode=reweight; the task reward sets every update's sign)"
            )
        else:
            print("[Video-t-RSI]   distillation steps: L_SD surrogate with the clipped chunk credit (rsi.sd_mode=surrogate)")
        if self.rsi.inner_loop == "route_c":
            print(
                "[Video-t-RSI]   inner loop: Route C at the least absorbed chunk of every distilled row "
                f"(k={self.rsi.route_c_candidates} candidates x N={self.rsi.route_c_continuations} continuations, "
                f"gap >= {self.rsi.route_c_min_gap}, coverage >= {self.rsi.route_c_min_coverage}, "
                f"kappa={self.rsi.route_c_kappa}, c_Q={self.rsi.route_c_value_clip}, coef={self.rsi.route_c_coef}; "
                "rsi.inner_loop=route_c)"
            )
        else:
            print("[Video-t-RSI]   inner loop: none (rsi.inner_loop=none, package v5)")
        print(
            f"[Video-t-RSI]   operator: alpha={self.rsi.extrapolation_alpha}, c_Delta={self.rsi.correction_clip}, "
            f"beta={self.rsi.base_mix_beta}, tau={self.rsi.teacher_temperature}, views={self.rsi.views}"
        )
        print(
            f"[Video-t-RSI]   credit: H={self.rsi.credit_horizon}, gamma={self.rsi.credit_discount}, "
            f"c={self.rsi.credit_clip}; chunks: min={self.rsi.chunk_min_tokens}, max={self.rsi.chunk_max_tokens}"
        )
        print(f"[Video-t-RSI]   teacher-scored steps in this run: {self.schedule.distill_step_count(self.training_steps)}")
        if self.schedule.distill_step_count(self.training_steps) == 0:
            print("[Video-t-RSI] WARNING: no step of this run uses the teacher; only GRPO task learning will run.")
        if float(rollout.top_p) < 1.0 or int(rollout.top_k) > 0:
            print(
                "[Video-t-RSI] NOTE: rollouts use truncated sampling "
                f"(top_p={rollout.top_p}, top_k={rollout.top_k}) while pi_old and the PPO ratio use untruncated "
                "temperature-scaled probabilities; delta and rho are therefore an approximation (method section 12)."
            )
        print(f"[Video-t-RSI]   stability: {self.stability_settings()}")
        print(f"[Video-t-RSI]   zero true-truncation training rewards: {self.rsi.zero_truncated_rewards}")
        if bool(rollout.calculate_log_probs) and float(rollout.temperature) != 1.0:
            print(
                "[Video-t-RSI] NOTE: vLLM returns log-probabilities before temperature scaling; with "
                f"temperature={rollout.temperature} the mismatch metrics and rsi.rollout_correction compare "
                "different distributions."
            )

    def stability_settings(self) -> dict[str, Any]:
        """The settings added against training collapse (printed at start and kept in the run_start record)."""

        rollout = self.config.worker.rollout
        return {
            "sd_mode": self.rsi.sd_mode,
            "reweight_clip": float(self.rsi.reweight_clip),
            "reweight_center": self.rsi.reweight_center,
            "inner_loop": self.rsi.inner_loop,
            "route_c_candidates": int(self.rsi.route_c_candidates),
            "route_c_continuations": int(self.rsi.route_c_continuations),
            "route_c_min_gap": float(self.rsi.route_c_min_gap),
            "route_c_min_coverage": float(self.rsi.route_c_min_coverage),
            "route_c_value_clip": float(self.rsi.route_c_value_clip),
            "route_c_kappa": float(self.rsi.route_c_kappa),
            "route_c_coef": float(self.rsi.route_c_coef),
            "task_loss_agg": self.rsi.task_loss_agg,
            "zero_truncated_rewards": bool(self.rsi.zero_truncated_rewards),
            "task_adv_std_floor": float(self.rsi.task_adv_std_floor),
            "rollout_seed_mode": self.rsi.rollout_seed_mode,
            "calculate_log_probs": bool(rollout.calculate_log_probs),
            "rollout_correction": self.rsi.rollout_correction,
            "rollout_correction_cap": float(self.rsi.rollout_correction_cap),
            "health_min_format_rate": self.rsi.health_min_format_rate,
            "health_min_response_length": self.rsi.health_min_response_length,
            "health_max_truncated_fraction": self.rsi.health_max_truncated_fraction,
            "health_ref_kl_jump_ratio": self.rsi.health_ref_kl_jump_ratio,
            "health_ref_kl_jump_floor": float(self.rsi.health_ref_kl_jump_floor),
            "health_ref_kl_window": int(self.rsi.health_ref_kl_window),
            "guard_patience": int(self.rsi.guard_patience),
            "guard_window": int(self.rsi.guard_window),
            "teacher_health_gate": bool(self.rsi.teacher_health_gate),
            "teacher_health_window": int(self.rsi.teacher_health_window),
            "teacher_health_max_format_drop": float(self.rsi.teacher_health_max_format_drop),
            "teacher_health_max_length_drop": float(self.rsi.teacher_health_max_length_drop),
            "teacher_health_max_length_rise": self.rsi.teacher_health_max_length_rise,
            "teacher_health_max_truncation_rise": self.rsi.teacher_health_max_truncation_rise,
            "ref_kl_coef": float(self.rsi.ref_kl_coef),
            "ref_kl_monitor": bool(self.rsi.ref_kl_monitor),
        }

    # ------------------------------------------------------------------
    # Teacher state
    # ------------------------------------------------------------------
    def _interval_degradation(self) -> Optional[str]:
        """Teacher health gate: why the rollouts of the interval since the anchor degraded (None: not degraded)."""

        if not bool(self.rsi.teacher_health_gate):
            return None
        return phase_degradation(
            self.phase_health.steps,
            window=int(self.rsi.teacher_health_window),
            max_format_drop=float(self.rsi.teacher_health_max_format_drop),
            max_length_drop=float(self.rsi.teacher_health_max_length_drop),
            max_length_rise=self.rsi.teacher_health_max_length_rise,
            max_truncation_rise=self.rsi.teacher_health_max_truncation_rise,
        )

    def _apply_teacher_events(self, plan: StepPlan, metrics: dict[str, Any]) -> None:
        for event in plan.events:
            degradation = None
            if event == EVENT_SNAPSHOT_ANCHOR:
                self.actor_rollout_ref_wg.trsi_snapshot("anchor")
                self.phase_health.reset()
            elif event == EVENT_SNAPSHOT_UPDATED:
                degradation = self._interval_degradation()
                # Degraded: the base teacher stays the student of the phase start (the anchor), Delta = 0.
                self.actor_rollout_ref_wg.trsi_snapshot("updated" if degradation is None else "updated_from_anchor")
            elif event == EVENT_INIT_TEACHER:
                self.actor_rollout_ref_wg.trsi_snapshot("both")
                self.phase_health.reset()
            elif event == EVENT_REFRESH_TEACHER:
                degradation = self._interval_degradation()
                # Order matters: the previous updated snapshot becomes the anchor first.
                self.actor_rollout_ref_wg.trsi_snapshot("anchor_from_updated")
                self.actor_rollout_ref_wg.trsi_snapshot("updated" if degradation is None else "updated_from_anchor")
                self.phase_health.reset()
            else:
                raise ValueError(f"Unknown Video-t-RSI event {event!r}.")
            version = self.schedule.apply_event(event, self.global_step, degradation=degradation)
            metrics[f"trsi/event/{event}"] = 1.0
            if degradation is not None:
                metrics["trsi/teacher_phase_degraded"] = 1.0
                print(
                    f"[Video-t-RSI] WARNING step {self.global_step}: the interval since the anchor degraded "
                    f"({degradation}); the new teacher keeps the anchor student as its base, without correction."
                )
            if version is not None:
                print(f"[Video-t-RSI] step {self.global_step}: teacher version {version.to_dict()}")

    def _teacher_rule(self, version: TeacherVersion) -> RouteATeacherRule:
        use_anchor = self.rsi.route == "A" and bool(version.correction_enabled)
        return RouteATeacherRule(
            alpha=float(version.alpha) if use_anchor else 0.0,
            correction_clip=self.rsi.correction_clip,
            base_mix=self.rsi.base_mix_beta,
            temperature=self.rsi.teacher_temperature,
            use_anchor=use_anchor,
        )

    # ------------------------------------------------------------------
    # Distillation targets
    # ------------------------------------------------------------------
    @staticmethod
    def _non_tensor_row(batch: DataProto, index: int) -> dict[str, Any]:
        return {key: values[index] for key, values in batch.non_tensor_batch.items() if len(values) > index}

    def _attach_sd_targets(
        self,
        batch: DataProto,
        plan: StepPlan,
        group_valid: np.ndarray,
        metrics: dict[str, Any],
        timing_raw,
        *,
        route_c_prompts: Optional[np.ndarray],
    ) -> None:
        """Teacher scoring and chunk credit; on Route C steps (``route_c_prompts``: the token prompts vLLM generated
        each row from) the teacher is also read at the student's candidates and the inner loop runs."""

        version = self.schedule.teacher_for_step(self.global_step)
        rule = self._teacher_rule(version)
        raw_prompts = batch.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raise ValueError("Video-t-RSI requires dataset raw_prompt strings to build evidence views.")
        rows = [self._non_tensor_row(batch, index) for index in range(len(batch))]
        with timer("trsi_teacher_prompt", timing_raw):
            privileged = [build_mesd_privileged_views(row) for row in rows]
            teacher_gen = construct_mesd_multiview_teacher_prompts(
                student_raw_prompts=raw_prompts,
                privileged_infos=privileged,
                tokenizer=self.tokenizer,
                processor=self.processor,
                max_prompt_length=self.rsi.teacher_max_prompt_length,
                multi_modal_data=batch.non_tensor_batch.get("multi_modal_data"),
                image_min_pixels=self.config.data.image_min_pixels,
                image_max_pixels=self.config.data.image_max_pixels,
                video_min_pixels=self.config.data.video_min_pixels,
                video_max_pixels=self.config.data.video_max_pixels,
                video_fps=self.config.data.video_fps,
                video_max_frames=self.config.data.video_max_frames,
                cache_keys=batch.non_tensor_batch.get("uid"),
            )
        view_available = teacher_gen.batch["view_available"].to(torch.bool)
        vision_failed = batch.non_tensor_batch.get("vision_read_failed")
        vision_ok = (
            np.ones(len(batch), dtype=bool)
            if vision_failed is None
            else np.array([not bool(value) for value in vision_failed], dtype=bool)
        )
        row_valid = torch.as_tensor(group_valid & vision_ok, dtype=torch.bool)
        view_mask = view_available[:, self.view_indices] & row_valid.unsqueeze(-1)
        teacher_keys = []
        for view, mesd_index in zip(self.rsi.views, self.view_indices):
            input_ids, attention_mask, position_ids = _concat_prompt_and_response(
                prompt_ids=teacher_gen.batch["input_ids"][:, mesd_index],
                prompt_attention_mask=teacher_gen.batch["attention_mask"][:, mesd_index],
                prompt_position_ids=teacher_gen.batch["position_ids"][:, mesd_index],
                response_ids=batch.batch["responses"],
                response_mask=batch.batch["response_mask"],
            )
            for suffix, value in (("input_ids", input_ids), ("attention_mask", attention_mask), ("position_ids", position_ids)):
                key = f"trsi_teacher_{view}_{suffix}"
                batch.batch[key] = value
                teacher_keys.append(key)
            metrics[f"trsi/view_{view}_coverage"] = float(view_mask[:, self.rsi.views.index(view)].float().mean().item())
        batch.batch["trsi_view_mask"] = view_mask.to(batch.batch["response_mask"].dtype)

        candidate_keys = ["trsi_candidate_ids"] if route_c_prompts is not None else []
        teacher_batch = batch.select(
            batch_keys=["responses", "response_mask", "trsi_view_mask", *teacher_keys, *candidate_keys],
            non_tensor_batch_keys=["uid", "multi_modal_data"],
        )
        teacher_batch.meta_info = dict(batch.meta_info)
        teacher_batch.meta_info["trsi_teacher"] = {
            **rule.to_dict(),
            "views": list(self.rsi.views),
            "fp32_logits": bool(self.rsi.teacher_fp32_logits),
            "vocab_chunk": int(self.rsi.teacher_vocab_chunk),
            "position_chunk": int(self.rsi.teacher_position_chunk),
            # Route C: read the teacher at the student's top candidates too (0: not on this step).
            "candidates": int(self.rsi.route_c_candidates) if route_c_prompts is not None else 0,
        }
        with timer("trsi_teacher_forward", timing_raw):
            teacher_out = self.actor_rollout_ref_wg.compute_trsi_teacher_log_probs(teacher_batch)
        batch.pop(batch_keys=teacher_keys)

        with timer("trsi_chunk_credit", timing_raw):
            targets = build_sd_targets(
                responses=batch.batch["responses"],
                response_mask=batch.batch["response_mask"],
                old_log_probs=batch.batch["old_log_probs"],
                teacher_log_probs=teacher_out.batch["trsi_teacher_log_probs"],
                view_count=teacher_out.batch["trsi_teacher_view_count"],
                row_valid=row_valid,
                aligner=self.aligner,
                chunk_cfg=self.chunk_cfg,
                credit_cfg=self.credit_cfg,
                rows=rows if self.verify_cfg is not None else None,
                verify_cfg=self.verify_cfg,
            )
        dtype = torch.float32
        batch.batch["trsi_sd_advantages"] = targets.advantages.to(dtype)
        batch.batch["trsi_sd_token_weights"] = targets.token_weights.to(dtype)
        batch.batch["trsi_sd_row_mask"] = targets.row_mask.to(dtype)
        metrics.update(targets.metrics)
        metrics.update(self._teacher_metrics(batch, teacher_out, targets.row_mask, rule, version))
        if self.rsi.sd_mode == "reweight":
            self._reweight_task_advantages(batch, targets, metrics)
        self._maybe_log_chunks(targets, batch)
        if route_c_prompts is not None:
            with timer("route_c", timing_raw):
                self._route_c_inner_loop(batch, targets, teacher_out, route_c_prompts, metrics, timing_raw)

    def _reweight_task_advantages(self, batch: DataProto, targets, metrics: dict[str, Any]) -> None:
        """``rsi.sd_mode=reweight``: A' = clip(exp(sign(A) (G - Gbar)), 1 - c, 1 + c) * A on every chunk token.

        ``Gbar`` is the row's token-weighted mean credit with ``rsi.reweight_center=row`` and 0 with ``none``
        (``credit.reward_anchored_factors``).  ``batch.batch["advantages"]`` (the GRPO advantages the update uses)
        is scaled in place; the factors are kept as ``trsi_reweight_factor`` for the rollout dumps.  ``returns``
        keep the GRPO values.
        """

        advantages = batch.batch["advantages"]
        factors = reward_anchored_factors(
            advantages.detach().cpu(),
            targets.chunk_credit,
            targets.chunk_ids,
            clip=float(self.rsi.reweight_clip),
            center=str(self.rsi.reweight_center),
        )
        metrics.update(reweight_metrics(factors, advantages.detach().cpu(), batch.batch["response_mask"].cpu()))
        factor = factors.factor
        batch.batch["advantages"] = advantages * factor.to(device=advantages.device, dtype=advantages.dtype)
        batch.batch["trsi_reweight_factor"] = factor.to(torch.float32)

    # ------------------------------------------------------------------
    # Route C inner loop (package v6)
    # ------------------------------------------------------------------
    def _route_c_skip_ids(self) -> list[int]:
        """Tokens the rollout's sampler never produces: the vision placeholders (``vLLMRollout``'s logit bias).

        A continuation must not start from one, since vLLM would read it as a media placeholder."""

        if self.processor is None:
            return []
        ids = []
        for name in ("image_token", "video_token"):
            if hasattr(self.processor, name):
                ids.append(int(self.processor.tokenizer.convert_tokens_to_ids(getattr(self.processor, name))))
        return ids

    @staticmethod
    def _eos_ids(batch: DataProto) -> list[int]:
        """The end-of-sequence ids of the rollout (set by the workers' ``generate_sequences``)."""

        if "eos_token_id" not in batch.meta_info:
            raise RuntimeError("The rollout batch carries no eos_token_id; the Route C inner loop needs it.")
        eos = batch.meta_info["eos_token_id"]
        if isinstance(eos, (int, np.integer)):
            return [int(eos)]
        return [int(value) for value in eos]

    def _pad_id(self, batch: DataProto) -> int:
        """Padding of the continuation rows: the tokenizer's pad id, as ``vLLMRollout`` pads the responses, or the
        first end-of-sequence id for a tokenizer without one.  The rows are masked by their lengths, so the value
        never reaches the reward or a loss."""

        pad = getattr(self.tokenizer, "pad_token_id", None)
        return int(pad) if pad is not None else self._eos_ids(batch)[0]

    def _route_c_inner_loop(
        self,
        batch: DataProto,
        targets,
        teacher_out: DataProto,
        raw_prompt_ids: np.ndarray,
        metrics: dict[str, Any],
        timing_raw: dict[str, float],
    ) -> None:
        """Branch every distilled row at its least absorbed chunk position and set the Route C targets there.

        Attaches ``trsi_rc_position`` ``[B]`` (-1: no target), ``trsi_rc_candidates`` ``[B, k]``,
        ``trsi_rc_target`` ``[B, k + 1]`` and ``trsi_rc_row_mask`` ``[B]`` for the update
        (``verl.trainer.trsi.route_c``).  The candidates and their student log-probabilities come from the
        old-log-prob forward, the teacher's from teacher scoring; the continuations sample from the same policy.
        """

        settings = self.rsi.route_c_settings()
        k, size = int(settings.candidates), len(batch)
        responses = batch.batch["responses"].cpu()
        response_mask = batch.batch["response_mask"].cpu().bool()
        candidate_ids = batch.batch["trsi_candidate_ids"].cpu().long()
        student = batch.batch["trsi_candidate_log_probs"].cpu().float()
        teacher = teacher_out.batch["trsi_teacher_candidate_log_probs"].cpu().float()
        expected = (size, responses.size(1), k)
        for name, value in (("trsi_candidate_ids", candidate_ids), ("trsi_candidate_log_probs", student), ("teacher", teacher)):
            if tuple(value.shape) != expected:
                raise ValueError(f"Route C {name} has shape {tuple(value.shape)}, expected {expected}.")
        row_mask = targets.row_mask.cpu().float()
        gap, coverage = candidate_gap(teacher, student, response_mask & row_mask.bool().unsqueeze(-1))
        branches, row_best = select_branches(
            gap,
            coverage,
            targets.chunk_ids.cpu(),
            row_mask,
            min_gap=float(settings.min_gap),
            min_coverage=float(settings.min_coverage),
        )
        position = torch.full((size,), -1, dtype=torch.long)
        chosen = torch.zeros(size, k, dtype=torch.long)
        target = torch.zeros(size, k + 1, dtype=torch.float32)
        branched = torch.zeros(size, dtype=torch.float32)
        rows = int(row_mask.gt(0).sum().item())
        finite = row_best[torch.isfinite(row_best)]
        values: dict[str, float] = {
            "trsi/route_c_rows": float(rows),
            "trsi/route_c_branches": float(len(branches)),
            "trsi/route_c_branch_fraction": float(len(branches)) / max(rows, 1),
            # The largest eligible gap of every distilled row, below the threshold too (for tuning min_gap).
            "trsi/route_c_row_gap_mean": float(finite.mean().item()) if finite.numel() else 0.0,
        }
        if branches:
            lengths = response_mask.sum(dim=-1).float()
            values["trsi/route_c_gap_mean"] = float(np.mean([branch.gap for branch in branches]))
            values["trsi/route_c_gap_max"] = float(np.max([branch.gap for branch in branches]))
            values["trsi/route_c_coverage_mean"] = float(np.mean([branch.coverage for branch in branches]))
            values["trsi/route_c_relative_position_mean"] = float(
                np.mean([branch.position / max(float(lengths[branch.row]), 1.0) for branch in branches])
            )
            result, stats = self._route_c_estimates(batch, branches, responses, candidate_ids, student, raw_prompt_ids, timing_raw)
            values.update(stats)
            values.update(result.metrics)
            for index, branch in enumerate(branches):
                row, at = branch.row, branch.position
                valid = bool(result.valid[index])
                if valid:
                    position[row] = at
                    chosen[row] = candidate_ids[row, at]
                    target[row] = result.target[index]
                    branched[row] = 1.0
                ids = [int(value) for value in candidate_ids[row, at].tolist()]
                self._route_c_rows[row] = {
                    "position": int(at),
                    "chunk": int(branch.chunk),
                    "gap": round(float(branch.gap), 6),
                    "coverage": round(float(branch.coverage), 6),
                    "candidates": ids,
                    "candidate_text": [self.tokenizer.decode([token]) for token in ids],
                    "student": [round(float(value), 6) for value in student[row, at].exp().tolist()],
                    "values": [None if math.isnan(float(value)) else round(float(value), 6) for value in result.values[index].tolist()],
                    "baseline": None if math.isnan(float(result.baseline[index])) else round(float(result.baseline[index]), 6),
                    "target": [round(float(value), 6) for value in result.target[index].tolist()] if valid else None,
                }
        values["trsi/route_c_targets"] = float(branched.sum().item())
        metrics.update(values)
        batch.batch["trsi_rc_position"] = position
        batch.batch["trsi_rc_candidates"] = chosen
        batch.batch["trsi_rc_target"] = target
        batch.batch["trsi_rc_row_mask"] = branched
        batch.pop(batch_keys=["trsi_candidate_ids", "trsi_candidate_log_probs"])

    def _route_c_estimates(self, batch, branches, responses, candidate_ids, student, raw_prompt_ids, timing_raw):
        """Continuations of every investigated candidate, their task rewards, and the targets of the branches."""

        settings = self.rsi.route_c_settings()
        k, samples = int(settings.candidates), int(settings.continuations)
        terminal, skip = self._eos_ids(batch), self._route_c_skip_ids()
        at_branches = [set(candidate_ids[branch.row, branch.position].tolist()) for branch in branches]
        stats: dict[str, float] = {
            "trsi/route_c_terminal_candidates": float(sum(len(ids & set(terminal)) for ids in at_branches)),
            "trsi/route_c_skipped_candidates": float(sum(len(ids & set(skip)) for ids in at_branches)),
        }
        base_seed = self.config.worker.rollout.seed
        requests = continuation_requests(
            branches,
            responses=responses,
            candidate_ids=candidate_ids,
            continuations=samples,
            terminal_ids=terminal,
            skip_ids=skip,
            seed_offset=route_c_seed_offset(
                0 if base_seed is None else int(base_seed), int(self.global_step), len(branches) * k * samples
            ),
        )
        stats["trsi/route_c_requests"] = float(len(requests))
        generated: list[Optional[list[int]]] = [None] * len(requests)
        generation_metadata: list[Optional[dict[str, Any]]] = [None] * len(requests)
        if requests:
            output, layout = self._generate_continuations(batch, branches, requests, raw_prompt_ids, stats, timing_raw)
            tail_ok, prompt_ok = output.batch["trsi_rc_tail_ok"], output.batch["trsi_rc_prompt_ok"]
            tokens, counts = output.batch["trsi_rc_tokens"], output.batch["trsi_rc_length"]
            wrong_tail = wrong_prompt = 0
            for slot, index in enumerate(layout):
                if index is None:
                    continue
                if not bool(tail_ok[slot]):
                    wrong_tail += 1
                elif not bool(prompt_ok[slot]):
                    wrong_prompt += 1
                else:
                    generated[index] = tokens[slot, : int(counts[slot])].tolist()
                    generation_metadata[index] = {
                        key: values[slot] for key, values in output.non_tensor_batch.items()
                        if key.startswith("rollout_")
                    }
            kept = [len(value) for value in generated if value is not None]
            # Dropped: vLLM's prompt did not end with the prefix + candidate, or its prompt part was not the prompt
            # the trainer scores the row with (then the continuation is not a sample of this prefix).
            stats["trsi/route_c_tail_mismatch_fraction"] = wrong_tail / len(requests)
            stats["trsi/route_c_prompt_mismatch_fraction"] = wrong_prompt / len(requests)
            stats["trsi/route_c_continuation_length_mean"] = float(np.mean(kept)) if kept else 0.0
        keys, scored, lengths = continuation_responses(
            branches,
            requests,
            generated,
            responses=responses,
            candidate_ids=candidate_ids,
            terminal_ids=terminal,
            skip_ids=skip,
            pad_id=self._pad_id(batch),
        )
        stats["trsi/route_c_scored"] = float(len(keys))
        value_sum = torch.zeros(len(branches), k, dtype=torch.float64)
        value_count = torch.zeros(len(branches), k, dtype=torch.float64)
        if keys:
            # Match continuation_responses exactly: terminal candidates first,
            # followed by accepted requests in request order (dispatch is reordered).
            scored_metadata = []
            metadata_keys = []
            for index, branch in enumerate(branches):
                for candidate, token in enumerate(candidate_ids[branch.row, branch.position].tolist()):
                    if int(token) in terminal and int(token) not in skip:
                        scored_metadata.append(terminal_metadata(int(token), eos_token_ids=terminal))
                        metadata_keys.append((index, candidate))
            for request, generated_tokens, metadata in zip(requests, generated, generation_metadata):
                if generated_tokens is not None:
                    scored_metadata.append(metadata or {})
                    metadata_keys.append((request.branch, request.candidate))
            if metadata_keys != keys:
                raise RuntimeError("Route C scoring metadata is not aligned with the completed responses.")
            rewards = self._score_continuations(
                batch, [branches[branch].row for branch, _ in keys], scored, lengths, timing_raw,
                termination_metadata=metadata_to_arrays(scored_metadata), metrics=stats,
            )
            value_sum, value_count = candidate_values(keys, rewards, branches=len(branches), candidates=k)
            stats["trsi/route_c_reward_mean"] = float(np.mean(rewards))
        student_at = torch.stack([student[branch.row, branch.position] for branch in branches])
        result = route_c_targets(
            student_at, value_sum, value_count, kappa=float(settings.kappa), value_clip=float(settings.value_clip)
        )
        return result, stats

    def _generate_continuations(self, batch, branches, requests, raw_prompt_ids, stats, timing_raw):
        """Wake vLLM again (same weights as this step's rollout) and sample the continuations on all ranks."""

        world_size = int(self.actor_rollout_ref_wg.world_size)
        uids = batch.non_tensor_batch["uid"]
        rows_of = [branches[request.branch].row for request in requests]
        prompts = len({uids[row] for row in rows_of})
        # Requests of one prompt on one rank (its media sent once) when there are enough prompts for every rank.
        groups = [uids[row] for row in rows_of] if prompts >= world_size else [request.branch for request in requests]
        layout, per_rank = continuation_layout(groups, [request.max_tokens + 1 for request in requests], world_size)
        stats["trsi/route_c_dispatch_rows_per_rank"] = float(per_rank)
        stats["trsi/route_c_dispatch_by_prompt"] = float(prompts >= world_size)
        requests_batch = self._continuation_batch(batch, requests, rows_of, layout, raw_prompt_ids)
        with timer("route_c_gen", timing_raw):
            self.actor_rollout_ref_wg.prepare_route_c_engine()
            output = self.actor_rollout_ref_wg.generate_trsi_continuations(requests_batch)
            self.actor_rollout_ref_wg.release_route_c_engine()
        if len(output) != len(layout):
            raise RuntimeError(f"Got {len(output)} continuation rows for {len(layout)} dispatched rows.")
        return output, layout

    def _continuation_batch(self, batch, requests, rows_of, layout, raw_prompt_ids) -> DataProto:
        """One row per dispatched continuation: the rollout prompt (token ids and media of the first row of its
        GRPO group, shared by reference so each rank receives them once), the tail, the seed, and the digest of the
        prompt the trainer scores the row with."""

        uids = batch.non_tensor_batch["uid"]
        owner_of: dict[Any, int] = {}
        for row in range(len(batch)):
            owner_of.setdefault(uids[row], row)
        prompt_tokens = batch.batch["prompts"].cpu()
        prompt_mask = batch.batch["attention_mask"][:, : prompt_tokens.size(1)].cpu().bool()
        texts = batch.non_tensor_batch.get("raw_prompt")
        media = batch.non_tensor_batch.get("multi_modal_data")
        size = len(layout)
        prompt_ids, tails = np.empty(size, dtype=object), np.empty(size, dtype=object)
        row_texts = None if texts is None else np.empty(size, dtype=object)
        row_media = None if media is None else np.empty(size, dtype=object)
        fields = {
            name: torch.zeros(size, dtype=torch.long)
            for name in ("trsi_rc_active", "trsi_rc_seed", "trsi_rc_max_tokens", "trsi_rc_prompt_digest", "trsi_rc_prompt_length")
        }
        fields["trsi_rc_media"] = torch.full((size,), -1, dtype=torch.long)
        digests: dict[int, tuple[int, int]] = {}
        for slot, index in enumerate(layout):
            if index is None:  # padding: not generated
                prompt_ids[slot], tails[slot] = [], []
                if row_texts is not None:
                    row_texts[slot] = None
                if row_media is not None:
                    row_media[slot] = None
                continue
            request = requests[index]
            owner = owner_of[uids[rows_of[index]]]
            if owner not in digests:
                ids = prompt_tokens[owner][prompt_mask[owner]].tolist()
                digests[owner] = (prompt_digest(ids), len(ids))
            prompt_ids[slot], tails[slot] = raw_prompt_ids[owner], list(request.tail)
            if row_texts is not None:
                row_texts[slot] = texts[owner]
            if row_media is not None:
                row_media[slot] = media[owner]
            fields["trsi_rc_active"][slot] = 1
            fields["trsi_rc_seed"][slot] = int(request.seed)
            fields["trsi_rc_max_tokens"][slot] = int(request.max_tokens)
            fields["trsi_rc_media"][slot] = int(owner)
            fields["trsi_rc_prompt_digest"][slot], fields["trsi_rc_prompt_length"][slot] = digests[owner]
        non_tensors = {"raw_prompt_ids": prompt_ids, "trsi_rc_tail": tails}
        if row_texts is not None:
            non_tensors["raw_prompt"] = row_texts
        if row_media is not None:
            non_tensors["multi_modal_data"] = row_media
        data = self.config.data
        meta_info = {
            "image_min_pixels": data.image_min_pixels,
            "image_max_pixels": data.image_max_pixels,
            "video_min_pixels": data.video_min_pixels,
            "video_max_pixels": data.video_max_pixels,
            "video_fps": data.video_fps,
            "video_max_frames": data.video_max_frames,
        }
        return DataProto.from_dict(tensors=fields, non_tensors=non_tensors, meta_info=meta_info)

    def _score_continuations(
        self, batch, source_rows, responses, lengths, timing_raw, *, termination_metadata=None, metrics=None
    ) -> list[float]:
        """Score the whole continued response, applying the same v7 reward policy as GRPO.

        Termination belongs to each continuation's actual generation budget,
        not to its source rollout or to the decoded total response length.
        """

        mask = VF.get_response_mask(
            response_ids=responses, eos_token_id=self._eos_ids(batch), dtype=batch.batch["response_mask"].dtype
        )
        mask = mask * (torch.arange(responses.size(1)).unsqueeze(0) < lengths.unsqueeze(-1)).to(mask.dtype)
        index = np.asarray(source_rows, dtype=np.int64)
        non_tensors = {
            key: values[index] for key, values in batch.non_tensor_batch.items()
            if key != "multi_modal_data" and not key.startswith("rollout_")
        }
        non_tensors.update(termination_metadata or {})
        reward_batch = DataProto.from_dict(
            tensors={"responses": responses, "response_mask": mask},
            non_tensors=non_tensors,
            meta_info={
                "global_step": int(self.global_step),
                "total_training_steps": int(self.training_steps),
                "step_percent": float(self.global_step / max(self.training_steps, 1)),
            },
        )
        with timer("route_c_reward", timing_raw):
            reward_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(reward_batch))
        if metrics is not None:
            metrics["trsi/route_c_raw_reward_mean"] = float(reward_tensor.sum(dim=-1).float().mean().item())
        if self.rsi.zero_truncated_rewards:
            reward_tensor, reward_stats = effective_training_rewards(reward_tensor, non_tensors)
            if metrics is not None:
                metrics.update({f"trsi/route_c_{key.replace('/', '_')}": value for key, value in reward_stats.items()})
        return [float(value) for value in reward_tensor.sum(dim=-1).tolist()]

    def _teacher_metrics(self, batch, teacher_out, row_mask, rule, version) -> dict[str, float]:
        values: dict[str, float] = {
            "trsi/teacher_version": float(version.version),
            "trsi/teacher_round": float(version.round_index),
            "trsi/teacher_alpha": float(rule.alpha),
            "trsi/teacher_correction_active": float(rule.correction_active),
            "trsi/teacher_task_signal_steps": float(version.task_signal_steps),
        }
        rows = row_mask.to(torch.bool)
        if bool(rows.any()):
            for name in ("kl_to_base", "clip_fraction", "entropy"):
                values[f"trsi/teacher_{name}"] = float(teacher_out.batch[f"trsi_teacher_{name}"][rows].mean().item())
            max_ratio = float(teacher_out.batch["trsi_teacher_max_log_ratio"][rows].max().item())
            values["trsi/teacher_max_log_ratio"] = max_ratio
            if rule.correction_active and rule.correction_clip is not None:
                values["trsi/teacher_ratio_bound_violation"] = float(max_ratio > rule.ratio_bound + 1.0e-4)
            tokens = batch.batch["response_mask"].to(torch.bool) & rows.unsqueeze(-1)
            old = batch.batch["old_log_probs"].float()
            base = teacher_out.batch["trsi_teacher_base_log_probs"].float()
            used = teacher_out.batch["trsi_teacher_log_probs"].float()
            if bool(tokens.any()):
                values["trsi/delta_base_mean"] = float((base - old)[tokens].mean().item())
                # Correction magnitude at the realized tokens: log q_used - log p_base (0 when Delta = 0).
                values["trsi/teacher_correction_mean"] = float((used - base)[tokens].mean().item())
                values["trsi/teacher_correction_abs_mean"] = float((used - base)[tokens].abs().mean().item())
        return values

    def _maybe_log_chunks(self, targets, batch: DataProto) -> None:
        count = int(self.rsi.log_chunk_examples)
        self._distill_steps_seen += 1
        if count <= 0 or (self._distill_steps_seen - 1) % int(self.rsi.log_chunk_interval) != 0:
            return
        shown = 0
        for row, chunked in enumerate(targets.chunked):
            if chunked is None or shown >= count:
                continue
            shown += 1
            print(f"[Video-t-RSI] chunks of row {row} (mode={chunked.parse_mode}, tokens={chunked.num_tokens}):")
            for chunk in chunked.chunks:
                text = chunked.text[chunk.char_start : chunk.char_end].replace("\n", "\\n")
                print(
                    f"  #{chunk.index} [{chunk.token_start},{chunk.token_end}) {chunk.role}/{chunk.status} "
                    f"D={float(targets.chunk_feedback[row, chunk.index]):+.4f} "
                    f"G={float(targets.chunk_credit[row, chunk.index]):+.4f} :: {text[:160]}"
                )

    # ------------------------------------------------------------------
    # Checkpoint state
    # ------------------------------------------------------------------
    def _rsi_protocol_dict(self) -> dict[str, Any]:
        value = asdict(self.rsi)
        return {key: item for key, item in value.items() if key not in _RESUME_TOLERANT_KEYS}

    def _protocol_differences(self, saved: Optional[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Protocol settings of this run that differ from a checkpoint's ``rsi`` state.

        A setting the checkpoint does not record (written before the setting existed) is compared with its
        legacy value, the behaviour of the code that wrote the checkpoint.
        """

        saved = dict(saved or {})
        current = self._rsi_protocol_dict()
        differences = {}
        for key in sorted(set(saved) | set(current)):
            if key in _RESUME_TOLERANT_KEYS:
                continue
            missing = object()
            old = saved.get(key, _PROTOCOL_LEGACY_VALUES.get(key, missing))
            new = current.get(key, missing)
            if old is missing or new is missing or old != new:
                differences[key] = {
                    "saved": None if old is missing else old,
                    "current": None if new is missing else new,
                    "recorded": key in saved,
                }
        return differences

    def _run_contract(self) -> dict[str, Any]:
        """Settings a resumed run must share with the checkpoint.

        Shards and batch layout (world size, batch sizes), the LR schedule
        (LambdaLR does not save its lambdas, so a different schedule would be
        applied silently), gradient clipping and PPO epochs, and the rollout
        sampling settings.  AdamW's own hyperparameters (betas, weight decay,
        eps) are restored from the optimizer checkpoint and need no check.
        """

        actor = self.config.worker.actor
        optim = actor.optim
        rollout = self.config.worker.rollout
        return {
            "training_steps": int(self.training_steps),
            "world_size": int(self.configured_world_size),
            "rollout_batch_size": int(self.config.data.rollout_batch_size),
            "rollout_n": int(rollout.n),
            "global_batch_size": int(actor.global_batch_size),
            "lr": float(optim.lr),
            "lr_scheduler_type": str(optim.lr_scheduler_type),
            "lr_warmup_ratio": float(optim.lr_warmup_ratio),
            "lr_warmup_steps": None if optim.lr_warmup_steps is None else int(optim.lr_warmup_steps),
            "min_lr_ratio": None if optim.min_lr_ratio is None else float(optim.min_lr_ratio),
            "max_grad_norm": float(actor.max_grad_norm),
            "ppo_epochs": int(actor.ppo_epochs),
            "rollout_sampling": {
                "temperature": float(rollout.temperature),
                "top_p": float(rollout.top_p),
                "top_k": int(rollout.top_k),
                "seed": None if rollout.seed is None else int(rollout.seed),
                "max_response_length": int(self.config.data.max_response_length),
            },
        }

    def _save_extra_state(self, folder_path: str) -> None:
        state = {
            "schema_version": TRSI_SCHEMA_VERSION,
            "completed_global_step": int(self.global_step),
            "rsi": self._rsi_protocol_dict(),
            "run_contract": self._run_contract(),
            "schedule": self.schedule.state_dict(),
            # Rollout health of the steps since the anchor snapshot (input of the teacher health gate).
            "phase_health": self.phase_health.state_dict(),
            # The guard's streak and recent history: a resumed run continues its window and its KL median.
            "collapse_guard": self.guard.state_dict(),
        }
        os.makedirs(folder_path, exist_ok=True)
        path = os.path.join(folder_path, TRSI_STATE_FILE)
        temporary = f"{path}.tmp-{uuid.uuid4().hex}"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)

    def _read_trsi_state(self, checkpoint_path: str) -> dict[str, Any]:
        """Load and validate ``trsi_state.json`` against this run (before any weights are loaded)."""

        path = os.path.join(checkpoint_path, TRSI_STATE_FILE)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} is missing; this is not a resumable Video-t-RSI checkpoint.")
        with open(path, encoding="utf-8") as handle:
            state = json.load(handle)
        if int(state.get("schema_version", -1)) != TRSI_SCHEMA_VERSION:
            raise ValueError("Unsupported Video-t-RSI checkpoint schema.")
        if int(state.get("completed_global_step", -1)) != int(self.global_step):
            raise ValueError("Video-t-RSI state step does not match the checkpoint directory.")
        differences = self._protocol_differences(state.get("rsi"))
        allowed = set(self.rsi.resume_allow_changes)
        refused = {key: value for key, value in differences.items() if key not in allowed}
        if refused:
            raise ValueError(
                "The Video-t-RSI protocol settings differ from the checkpoint in "
                f"{sorted(refused)}: { {key: (value['saved'], value['current']) for key, value in refused.items()} } "
                "(saved, current). Start a new run, or, to continue this checkpoint under the new settings on "
                f"purpose, add rsi.resume_allow_changes=[{','.join(sorted(refused))}]; the change is then recorded."
            )
        unused = sorted(allowed - set(differences))
        if not getattr(self, "_protocol_reported", False):
            self._protocol_reported = True  # the state is read twice per resume
            for key, value in differences.items():
                print(
                    f"[Video-t-RSI] WARNING: resuming with rsi.{key}={value['current']!r} instead of the checkpoint's "
                    f"{value['saved']!r}{'' if value['recorded'] else ' (not recorded; the earlier behaviour)'} "
                    "(rsi.resume_allow_changes)."
                )
            if unused:
                print(
                    f"[Video-t-RSI] WARNING: rsi.resume_allow_changes lists {unused}, but this run's values equal the "
                    "checkpoint's, so they do not change. That is expected when a run resumed under the allowance is "
                    "resumed again; otherwise check which config file this run reads (the launcher prints it as "
                    "'Resuming ... config ...')."
                )
        self._protocol_changes = differences
        saved_contract = state.get("run_contract")
        current = self._run_contract()
        if saved_contract is not None:
            # Keys the checkpoint does not record (written before they were added) cannot be compared.
            differing = sorted(key for key in saved_contract if saved_contract[key] != current.get(key))
            if differing:
                raise ValueError(
                    f"This run differs from the checkpoint in {differing} (world size, batch layout, LR schedule, "
                    f"gradient clipping or rollout sampling): saved={ {key: saved_contract[key] for key in differing} }, "
                    f"current={ {key: current.get(key) for key in differing} }. Resume with the configuration the "
                    "checkpoint was written with (the launcher reuses it from <run_dir>/launch/run_manifest.env)."
                )
            unchecked = sorted(set(current) - set(saved_contract))
            if unchecked and not getattr(self, "_contract_warned", False):
                self._contract_warned = True  # the state is read twice per resume
                print(
                    f"[Video-t-RSI] WARNING: the checkpoint's run contract does not record {unchecked} "
                    "(written by an earlier version); these settings were not compared."
                )
        return state

    def _verify_checkpoint(self, checkpoint_path: str) -> None:
        super()._verify_checkpoint(checkpoint_path)
        check_rank_shards(
            os.path.join(checkpoint_path, "actor"), TEACHER_SHARD_PREFIXES, self.actor_rollout_ref_wg.world_size
        )
        self._read_trsi_state(checkpoint_path)

    def _load_extra_state(self, checkpoint_path: str) -> None:
        state = self._read_trsi_state(checkpoint_path)
        self.schedule.load_state_dict(state["schedule"])
        # Absent from checkpoints written before the teacher health gate: the gate then sees the steps after the resume.
        self.phase_health.load_state_dict(state.get("phase_health"))
        self.guard.load_state_dict(state.get("collapse_guard"))
        # Only the chunk-printing cadence depends on this counter.
        self._distill_steps_seen = self.schedule.distill_step_count(self.global_step)
        print(f"[Video-t-RSI] restored teacher state: {self.schedule.current.to_dict() if self.schedule.current else None}")

    # ------------------------------------------------------------------
    # Run records
    # ------------------------------------------------------------------
    def _state_fingerprint(self) -> Optional[dict[str, Any]]:
        if not bool(self.rsi.record_state_fingerprints):
            return None
        return combine_fingerprints(self.actor_rollout_ref_wg.trsi_state_fingerprint())

    def _step_record(
        self,
        *,
        plan: StepPlan,
        metrics: dict[str, Any],
        fingerprint_before: Optional[dict[str, Any]],
        fingerprint_after: Optional[dict[str, Any]],
        data_ids: list[Any],
        has_work: bool,
        update_applied: bool,
        checkpoint_due: bool,
        timing: Optional[dict[str, float]] = None,
    ) -> dict[str, Any]:
        version = self.schedule.teacher_for_step(self.global_step) if plan.needs_teacher else None
        version_dict = None if version is None else version.to_dict()
        rule = None if version is None else self._teacher_rule(version).to_dict()
        # How the teacher is consumed at this step (step records before package v5: always the L_SD surrogate).
        sd_mode = str(self.rsi.sd_mode) if plan.needs_teacher else None
        # The inner loop of this step (step records before package v6: none).
        inner_loop = str(self.rsi.inner_loop) if plan.needs_teacher else None
        reason = correction_reason(
            needs_teacher=plan.needs_teacher, route=self.rsi.route, version=version_dict, rule=rule
        )
        groups = metric_groups(metrics)
        optimizer_steps = None
        if fingerprint_before is not None and fingerprint_after is not None:
            optimizer_steps = {
                "before": fingerprint_before["optimizer"]["step"],
                "after": fingerprint_after["optimizer"]["step"],
            }
        return {
            "kind": "step",
            "step": int(self.global_step),
            "total_steps": int(self.training_steps),
            "round": int(plan.round_index),
            "phase": plan.phase,
            "offset": int(plan.offset),
            "events": list(plan.events),
            "task_coef": float(plan.task_coef),
            "sd_coef": float(plan.sd_coef),
            "sd_mode": sd_mode,
            "reweight_clip": float(self.rsi.reweight_clip) if sd_mode == "reweight" else None,
            "reweight_center": str(self.rsi.reweight_center) if sd_mode == "reweight" else None,
            "inner_loop": inner_loop,
            "zero_truncated_rewards": bool(self.rsi.zero_truncated_rewards),
            "route_c_coef": float(self.rsi.route_c_coef) if inner_loop == "route_c" else None,
            "teacher": {
                "used": bool(plan.needs_teacher),
                "version": version_dict,
                "rule": rule,
                "correction_applied": reason == "enabled",
                "correction_reason": reason,
            },
            "metrics": groups,
            "update": {
                "has_work": bool(has_work),
                "update_applied": bool(update_applied),
                "optimizer_steps": optimizer_steps,
            },
            "fingerprint_before": fingerprint_before,
            "fingerprint_after": fingerprint_after,
            "data": {
                "prompts": len(data_ids),
                "ids_sha256": ids_digest(data_ids),
                "ids_head": [str(value) for value in data_ids[:4]],
            },
            "dispatch": {
                "world_size": int(self.actor_rollout_ref_wg.world_size),
                "sample_level": bool(metrics.get("rollout/sample_level_dispatch", 0.0) > 0.5),
                "rows_per_rank": metrics.get("rollout/rows_per_rank"),
            },
            "schedule_state": {
                "signal_steps_in_stage": int(self.schedule.signal_steps),
                "teacher_versions_built": len(self.schedule.history),
            },
            # The checkpoint itself gets its own "checkpoint" record once it is complete.
            "checkpoint_due": bool(checkpoint_due),
            # Wall-clock seconds of this step up to this record (step_s; the checkpoint save is not included)
            # and of its stages (gen, old, adv, trsi_teacher, update_actor, ...).
            "timing": {name: round(float(value), 3) for name, value in (timing or {}).items()},
        }

    @staticmethod
    def _prompt_ids(batch: DataProto) -> list[Any]:
        """Dataset ids of the step's prompts in loader order (one per GRPO group)."""

        uids = batch.non_tensor_batch["uid"].tolist()
        ids = batch.non_tensor_batch.get("id")
        first_row = {}
        for row, uid in enumerate(uids):
            first_row.setdefault(uid, row)
        return [ids[row] if ids is not None else uid for uid, row in first_row.items()]

    def _balance_batch(
        self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen"
    ) -> torch.Tensor:
        """Order the rows so that every micro-batch position carries about the same tokens on every rank.

        Every forward/backward of a micro-batch is synchronized across the ranks (FSDP collectives), so a
        pass over a rank's micro-batches costs the slowest rank at each position.  The base trainer balances
        only each rank's total.  Here the units are gcd(update, experience) micro-batch rows, so the update
        micro-batches, the experience micro-batches and, with several optimizer steps per rollout step, the
        rows of each optimizer step are balanced (``micro_batch_balanced_order``).  Advantages, rewards and
        targets are computed per uid and row, so the order changes only which rows share an optimizer step
        (and the floating-point summation order), as the base trainer's reordering already did.

        With several optimizer steps per rollout step, every optimizer step holds whole GRPO groups
        (``minibatch_group_order``): the advantages of a group then sum to zero within the step, which the
        token-mean task loss needs to leave the tokens shared by a group (the format prefix) without a net push.
        """

        actor = self.config.worker.actor
        world_size = self.actor_rollout_ref_wg.world_size
        seqlens = batch.batch["attention_mask"].view(len(batch), -1).sum(-1).tolist()
        update_rows = int(actor.micro_batch_size_per_device_for_update)
        experience_rows = int(actor.micro_batch_size_per_device_for_experience)
        steps = int(self.batch_layout.optimizer_steps_per_rollout_step)
        if steps > 1:
            order = minibatch_group_order(
                seqlens,
                batch.non_tensor_batch["uid"].tolist(),
                world_size=world_size,
                minibatches=steps,
                unit_size=math.gcd(update_rows, int(self.batch_layout.update_rows_per_rank)),
            )
        else:
            order = micro_batch_balanced_order(seqlens, world_size=world_size, unit_size=math.gcd(update_rows, experience_rows))
        global_idx = torch.tensor(order)
        batch.reorder(global_idx)
        rows_per_rank = len(order) // world_size
        partitions = [order[rank * rows_per_rank : (rank + 1) * rows_per_rank] for rank in range(world_size)]
        metrics.update(log_seqlen_unbalance(seqlen_list=seqlens, partitions=partitions, prefix=logging_prefix))
        # 1.0 = every synchronized micro-batch is as long on every rank (tokens of the mean rank / slowest rank).
        ideal = sum(seqlens) / world_size
        for name, rows in (("update", update_rows), ("experience", experience_rows)):
            cost = synchronized_token_cost(seqlens, order, world_size, rows)
            metrics[f"{logging_prefix}/{name}_sync_efficiency"] = float(ideal / cost) if cost else 1.0
        return global_idx

    def _generate_rollout(self, gen_batch: DataProto, metrics: dict[str, Any]) -> DataProto:
        """Attach per-prompt sampling seeds (``rsi.rollout_seed_mode=per_step``), then dispatch as the base trainer.

        On Route C steps the rows also keep the token prompt vLLM generated from (``trsi_raw_prompt_ids``; the
        rollout consumes ``raw_prompt_ids``): the continuations restart from it.
        """

        kept_prompts = None
        if self._route_c_this_step:
            source = gen_batch.non_tensor_batch["raw_prompt_ids"]
            prompts = np.empty(len(source), dtype=object)  # one list per row (never a 2-D array)
            for index, ids in enumerate(source):
                prompts[index] = [int(value) for value in ids]
            kept_prompts = np.repeat(prompts, int(self.config.worker.rollout.n), axis=0)

        base_seed = self.config.worker.rollout.seed
        if self.rsi.rollout_seed_mode == "per_step" and base_seed is not None:
            seeds = rollout_request_seeds(
                base_seed=int(base_seed),
                step=int(self.global_step),
                call=int(self._rollout_calls_in_step),
                num_prompts=len(gen_batch),
                samples_per_prompt=int(self.config.worker.rollout.n),
            )
            gen_batch.non_tensor_batch[ROLLOUT_SEED_KEY] = seeds
            self._rollout_calls_in_step += 1
            metrics["rollout/seed_first"] = float(seeds[0]) if len(seeds) else 0.0
        output = super()._generate_rollout(gen_batch, metrics)
        if kept_prompts is not None:
            if len(kept_prompts) != len(output):
                raise RuntimeError(f"Kept {len(kept_prompts)} rollout prompts for {len(output)} generated rows.")
            output.non_tensor_batch["trsi_raw_prompt_ids"] = kept_prompts
        return output

    # ------------------------------------------------------------------
    # Run health
    # ------------------------------------------------------------------
    def _rollout_correction(self, batch: DataProto, metrics: dict[str, Any]) -> None:
        """Mismatch metrics of vLLM's sampled-token log-probabilities and, with ``tis``, the importance weights."""

        if "rollout_log_probs" not in batch.batch.keys():
            return
        old, rollout = batch.batch["old_log_probs"], batch.batch["rollout_log_probs"]
        mask = batch.batch["response_mask"]
        cap = float(self.rsi.rollout_correction_cap)
        metrics.update(rollout_mismatch_metrics(old, rollout, mask, cap))
        if self.rsi.rollout_correction == "tis":
            batch.batch["trsi_rollout_is_weights"] = truncated_is_weights(old, rollout, mask, cap)

    def _reference_log_probs(self, batch: DataProto, metrics: dict[str, Any], timing_raw: dict[str, float]) -> DataProto:
        """log pi_ref of the sampled tokens (KL penalty and health/ref_kl), when the reference exists."""

        if not (float(self.rsi.ref_kl_coef) > 0.0 or bool(self.rsi.ref_kl_monitor)):
            return batch
        with timer("ref", timing_raw):
            reference = self.actor_rollout_ref_wg.compute_trsi_reference_log_probs(batch)
        batch = batch.union(reference)
        mask = batch.batch["response_mask"].to(torch.float32)
        if float(mask.sum().item()) > 0:
            # The penalty's reported value (k3, as in the update) at the sampling policy pi_old.
            k3 = reference_kl_k3(batch.batch["old_log_probs"], batch.batch["ref_log_probs"]).detach()
            metrics["health/ref_kl"] = float(((k3 * mask).sum() / mask.sum()).item())
        return batch

    def _dump_rollouts(
        self, batch: DataProto, reward_metrics: dict[str, Any], *, unhealthy: bool, reasons: list[str]
    ) -> Optional[str]:
        """Decoded rollouts of this step with lengths, rewards and advantages (``rsi.dump_rollouts_*``)."""

        interval, rows_wanted = int(self.rsi.dump_rollouts_interval), int(self.rsi.dump_rollouts_rows)
        regular = interval > 0 and rows_wanted > 0 and self.global_step % interval == 0
        if not (regular or unhealthy):
            return None
        uids = batch.non_tensor_batch["uid"].tolist()
        if unhealthy:
            rows = list(range(len(batch)))
        else:  # whole GRPO groups, in batch order, until rows_wanted rows are covered
            chosen = []
            for uid in dict.fromkeys(uids):
                if len(chosen) * int(self.config.worker.rollout.n) >= rows_wanted:
                    break
                chosen.append(uid)
            keep = set(chosen)
            rows = sorted((row for row in range(len(batch)) if uids[row] in keep), key=lambda row: (chosen.index(uids[row]), row))
        responses, mask = batch.batch["responses"], batch.batch["response_mask"]
        advantages = batch.batch["advantages"]
        sd_advantages = batch.batch["trsi_sd_advantages"] if "trsi_sd_advantages" in batch.batch.keys() else None
        factors = batch.batch["trsi_reweight_factor"] if "trsi_reweight_factor" in batch.batch.keys() else None
        ids = batch.non_tensor_batch.get("id")
        per_row = {
            key: value
            for key, value in reward_metrics.items()
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == len(batch)
        }
        directory = os.path.join(self.config.trainer.save_checkpoint_path, "logs", "rollouts")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"step_{self.global_step:07d}.jsonl")
        temporary = f"{path}.tmp-{uuid.uuid4().hex}"
        max_length = int(self.config.data.max_response_length)
        true_truncated = (
            termination_flags(batch.non_tensor_batch, expected_rows=len(batch))
            if self.rsi.zero_truncated_rewards else None
        )
        skip_special = bool(getattr(self.config.worker.reward, "skip_special_tokens", True))
        with open(temporary, "w", encoding="utf-8") as handle:
            for row in rows:
                length = int(mask[row].sum().item())
                token_ids = responses[row, :length].tolist()
                item = {
                    "step": int(self.global_step),
                    "row": int(row),
                    "uid": str(uids[row]),
                    "id": None if ids is None else str(ids[row]),
                    "length": length,
                    "truncated": bool(true_truncated[row]) if true_truncated is not None else length >= max_length,
                    "length_at_limit": length >= max_length,
                    "truncation_definition": "generation_termination" if true_truncated is not None else "legacy_length_proxy",
                    "raw_reward": float(batch.batch["token_level_scores"][row].sum().item()),
                    "training_reward": float(batch.batch["token_level_rewards"][row].sum().item()),
                    "truncation_reward_penalty": float(
                        (batch.batch["token_level_rewards"][row] - batch.batch["token_level_scores"][row]).sum().item()
                    ),
                    # The GRPO advantage of the row (before a reward-anchored reweighting; see reweight_factor_mean).
                    "advantage": float(
                        (advantages[row, :length] / (factors[row, :length] if factors is not None else 1.0)).sum().item()
                        / max(length, 1)
                    ),
                    "rewards": {key: float(value[row]) for key, value in per_row.items() if _is_number(value[row])},
                    "unhealthy_step": bool(unhealthy),
                    "reasons": list(reasons),
                    "text": self.tokenizer.decode(token_ids, skip_special_tokens=skip_special),
                }
                for key, values in batch.non_tensor_batch.items():
                    if key.startswith("rollout_"):
                        value = values[row]
                        item[key] = value.tolist() if isinstance(value, np.ndarray) else value.item() if isinstance(value, np.generic) else value
                if sd_advantages is not None:
                    item["sd_advantage_mean"] = float(sd_advantages[row, :length].mean().item()) if length else 0.0
                    item["sd_row_valid"] = bool(batch.batch["trsi_sd_row_mask"][row].item() > 0.5)
                if factors is not None:
                    item["reweight_factor_mean"] = float(factors[row, :length].mean().item()) if length else 1.0
                if row in self._route_c_rows:  # the Route C branch of this row (position, candidates, Q, target)
                    item["route_c"] = self._route_c_rows[row]
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
        return path

    def _last_checkpoints_before(self, run_dir: str, step: int) -> list[int]:
        """Checkpoint steps recorded in this run directory at or before ``step``."""

        steps = []
        path = os.path.join(run_dir, "logs", "trsi_steps.jsonl")
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        continue
                    if record.get("kind") in ("checkpoint", "final_checkpoint") and int(record.get("step", -1)) <= step:
                        steps.append(int(record["step"]))
        for name in os.listdir(run_dir) if os.path.isdir(run_dir) else ():
            if name.startswith("global_step_") and name[len("global_step_") :].isdigit():
                value = int(name[len("global_step_") :])
                if value <= step and os.path.isfile(os.path.join(run_dir, name, TRSI_STATE_FILE)):
                    steps.append(value)
        return sorted(set(steps))

    def _close_tracker(self) -> None:
        """Close the experiment trackers when the driver ends: at a guard stop or after the completed run
        (TensorBoard writes its pending events on close; the Ray actor may exit without running destructors)."""

        finish = getattr(self.logger, "finish", None)
        if callable(finish):
            try:
                finish()
            except Exception as error:  # a tracker problem must not hide the collapse report
                print(f"[Video-t-RSI] WARNING: closing the experiment trackers failed: {error!r}")

    def _stop_for_collapse(self, recorder: StepRecorder, reasons: list[str], loaded_checkpoint: Optional[str]) -> None:
        run_dir = self.config.trainer.save_checkpoint_path
        unhealthy = self.guard.unhealthy_steps(self.global_step)
        start = int(min(unhealthy)) if unhealthy else int(self.global_step)
        # Rollouts of step s come from the student saved at step s - 1: the last student known to sample healthy
        # rollouts is the one saved at start - 2 or earlier.  The same holds for the checkpoint this run resumed
        # from (possibly in another run directory).
        candidates = self._last_checkpoints_before(run_dir, start - 2)
        resumed_step = self._resumed_from_step
        if candidates:
            suggestion = os.path.join(run_dir, f"global_step_{candidates[-1]}")
        elif loaded_checkpoint is not None and resumed_step is not None and resumed_step <= start - 2:
            suggestion = loaded_checkpoint
        else:
            suggestion = None
        message = (
            f"Video-t-RSI collapse guard: {len(unhealthy)} of the last {self.guard.width} steps unhealthy "
            f"(steps {format_steps(unhealthy)}; {'; '.join(reasons)}). Training stopped at step {self.global_step} "
            "without saving this step. "
            + (
                f"Latest checkpoint before the unhealthy steps: {suggestion}. "
                if suggestion
                else "No checkpoint of this run precedes the unhealthy steps"
                + (
                    f" (the student of the checkpoint this run resumed from, {loaded_checkpoint}, sampled unhealthy "
                    "rollouts itself). "
                    if loaded_checkpoint is not None
                    else ". "
                )
            )
            + "Inspect logs/rollouts/ and the health metrics of logs/trsi_steps.jsonl before resuming; resuming "
            "with unchanged settings is likely to collapse again (docs/SERVER_RUNBOOK.md, 'Training collapse')."
        )
        payload = {
            "kind": "collapse_guard",
            "step": int(self.global_step),
            "unhealthy_since": start,
            "unhealthy_steps": unhealthy,
            "patience": int(self.guard.patience),
            "window": int(self.guard.width),
            "reasons": list(reasons),
            "recent_health": list(self.guard.history),
            "checkpoints_before_collapse": candidates,
            "suggested_resume_checkpoint": suggestion,
            "message": message,
        }
        recorder.write(payload)
        write_json_atomic(os.path.join(run_dir, COLLAPSE_MARKER), payload)
        print(f"[Video-t-RSI] {message}")
        self._close_tracker()  # after the marker: a slow tracker close must not delay it (`status` reads it)
        raise TrainingCollapseError(message)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _compute_task_advantage(self, batch: DataProto, group_valid: np.ndarray) -> bool:
        advantages, returns = compute_grpo_advantage(
            token_level_rewards=batch.batch["token_level_rewards"],
            response_mask=batch.batch["response_mask"],
            index=batch.non_tensor_batch["uid"],
            eps=self.config.algorithm.grpo_eps,
        )
        floor = float(self.rsi.task_adv_std_floor)
        if floor > 0.0:
            scale = group_std_floor_scale(
                batch.batch["token_level_rewards"].sum(dim=-1),
                batch.non_tensor_batch["uid"],
                floor=floor,
                eps=float(self.config.algorithm.grpo_eps),
            ).to(advantages.device, advantages.dtype)
            advantages, returns = advantages * scale.unsqueeze(-1), returns * scale.unsqueeze(-1)
        valid = torch.as_tensor(group_valid, dtype=advantages.dtype, device=advantages.device).unsqueeze(-1)
        batch.batch["advantages"] = advantages * valid
        batch.batch["returns"] = returns * valid
        batch.batch["trsi_task_row_mask"] = torch.as_tensor(group_valid, dtype=torch.float32)
        signal_rows = batch.batch["advantages"].abs().sum(dim=-1).gt(0)
        return bool(signal_rows.any().item())

    def fit(self):
        self.logger = self._create_logger()
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None
        loaded_checkpoint = self._load_checkpoint()
        self._resumed_from_step = int(self.global_step) if loaded_checkpoint is not None else None
        main_tqdm.update(self.global_step)
        run_dir = self.config.trainer.save_checkpoint_path
        recorder = StepRecorder(run_dir)
        recorder.write(
            {
                "kind": "run_start",
                "step": int(self.global_step),
                "total_steps": int(self.training_steps),
                "resumed_from_step": int(self.global_step) if loaded_checkpoint is not None else None,
                "resumed_from_checkpoint": loaded_checkpoint,
                "experiment_name": self.config.trainer.experiment_name,
                "world_size": int(self.actor_rollout_ref_wg.world_size),
                "batch_layout": self.batch_layout.to_dict(),
                "schedule": self.schedule.config_dict(),
                "route": self.rsi.route,
                "teacher_current": None if self.schedule.current is None else self.schedule.current.to_dict(),
                "teacher_scored_steps": self.schedule.distill_step_count(self.training_steps),
                "stability": self.stability_settings(),
                # rsi.* settings that differ from the resumed checkpoint (allowed by rsi.resume_allow_changes).
                "protocol_changes": self._protocol_changes,
            }
        )
        print(f"[Video-t-RSI] step records: {recorder.path}")
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return
        self.data_iterator = iter(self.train_dataloader)
        last_checkpoint = loaded_checkpoint
        while self.global_step < self.training_steps:
            self.global_step += 1
            step_started = time.monotonic()
            plan = self.schedule.plan(self.global_step)
            metrics: dict[str, Any] = {
                "trsi/round": float(plan.round_index),
                "trsi/phase_task": float(plan.phase == "task"),
                "trsi/phase_distill": float(plan.phase == "distill"),
                "trsi/phase_joint": float(plan.phase == "joint"),
            }
            timing_raw: dict[str, float] = {}
            # Route C inner loop (package v6) on the teacher-scored steps.
            route_c = bool(plan.needs_teacher and self.rsi.inner_loop == "route_c")
            self._route_c_rows = {}
            with timer("step", timing_raw):
                with timer("trsi_events", timing_raw):
                    self._apply_teacher_events(plan, metrics)
                # After the events: the student entering this step and the teachers this step uses.
                fingerprint_before = self._state_fingerprint()
                self._rollout_calls_in_step = 0
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    self._route_c_this_step = route_c
                    batch = self._make_batch_data(metrics=metrics)
                    self._route_c_this_step = False
                    self.actor_rollout_ref_wg.release_rollout_engine()
                data_ids = self._prompt_ids(batch)
                self._balance_batch(batch, metrics=metrics)
                # The token prompts vLLM generated each row from, kept for the Route C continuations.
                route_c_prompts = batch.non_tensor_batch.pop("trsi_raw_prompt_ids") if route_c else None
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
                    if route_c:  # also the student's top candidates at every response position, same forward
                        batch.meta_info["trsi_candidates"] = int(self.rsi.route_c_candidates)
                        batch.meta_info["trsi_candidate_chunk"] = int(self.rsi.teacher_position_chunk)
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch.meta_info.pop("trsi_candidates", None)
                    batch.meta_info.pop("trsi_candidate_chunk", None)
                    batch = batch.union(old_log_probs)
                self._rollout_correction(batch, metrics)
                batch = self._reference_log_probs(batch, metrics, timing_raw)
                with timer("adv", timing_raw):
                    reward_tensor, reward_metrics = ray.get(reward_ref)
                    if reward_metrics is None:
                        raise RuntimeError("Video-t-RSI requires synchronous reward metrics.")
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()})
                    batch.batch["token_level_scores"] = reward_tensor
                    if self.rsi.zero_truncated_rewards:
                        train_rewards, reward_stats = effective_training_rewards(reward_tensor, batch.non_tensor_batch)
                        batch.batch["token_level_rewards"] = train_rewards
                        metrics.update(reward_stats)
                        metrics["training_reward/enabled"] = 1.0
                        truncated = termination_flags(batch.non_tensor_batch, expected_rows=len(batch))
                    else:
                        batch.batch["token_level_rewards"] = reward_tensor.clone()
                        truncated = None  # legacy health semantics are retained with the v7 policy disabled
                    group_valid, validity_metrics = compute_group_validity(
                        reward_metrics, batch.non_tensor_batch["uid"], self.config.worker.rollout.n
                    )
                    metrics.update(validity_metrics)
                    task_signal = self._compute_task_advantage(batch, group_valid)
                    metrics["trsi/task_signal"] = float(task_signal)
                health = rollout_health(
                    response_mask=batch.batch["response_mask"],
                    reward_metrics=reward_metrics,
                    old_log_probs=batch.batch["old_log_probs"],
                    max_response_length=int(self.config.data.max_response_length),
                    advantages=batch.batch["advantages"],
                    uids=batch.non_tensor_batch["uid"],
                    row_valid=group_valid,  # media or ground-truth failures are not a collapse
                    truncated=truncated,
                )
                if "health/ref_kl" in metrics:  # from the reference log-probs (_reference_log_probs), for the guard
                    health["health/ref_kl"] = metrics["health/ref_kl"]
                reasons = unhealthy_reasons(
                    health,
                    self.health_thresholds,
                    ref_kl_history=self.guard.recent_values("ref_kl", int(self.rsi.health_ref_kl_window)),
                )
                stop_run = self.guard.update(self.global_step, health, reasons)
                in_window = len(self.guard.unhealthy_steps(self.global_step))
                health["health/unhealthy"] = float(bool(reasons))
                health["health/unhealthy_streak"] = float(self.guard.streak)
                health["health/unhealthy_in_window"] = float(in_window)
                metrics.update(health)
                if plan.phase != PHASE_DISTILL:  # task-phase (or joint) steps feed the teacher health gate
                    self.phase_health.add(self.global_step, health)
                if reasons:
                    patience = f"; the guard stops at {self.guard.patience}" if self.guard.patience else ""
                    print(
                        f"[Video-t-RSI] WARNING step {self.global_step}: unhealthy rollouts ({'; '.join(reasons)}), "
                        f"{in_window} of the last {self.guard.width} steps unhealthy{patience}."
                    )
                if plan.needs_teacher:
                    with timer("trsi_teacher", timing_raw):
                        self._attach_sd_targets(batch, plan, group_valid, metrics, timing_raw, route_c_prompts=route_c_prompts)
                route_c_rows = (
                    float(batch.batch["trsi_rc_row_mask"].sum().item()) if "trsi_rc_row_mask" in batch.batch.keys() else 0.0
                )
                route_c_coef = float(self.rsi.route_c_coef) if route_c_rows > 0 else 0.0
                batch.meta_info["trsi_loss"] = {
                    "task_coef": float(plan.task_coef),
                    "sd_coef": float(plan.sd_coef),
                    "task_clip_low": float(self.rsi.task_clip_low),
                    "task_clip_high": float(self.rsi.task_clip_high),
                    "sd_clip_low": float(self.rsi.sd_clip_low),
                    "sd_clip_high": float(self.rsi.sd_clip_high),
                    "task_loss_agg": str(self.rsi.task_loss_agg),
                    "rollout_is": "trsi_rollout_is_weights" in batch.batch.keys(),
                    "ref_kl_coef": float(self.rsi.ref_kl_coef),
                    # Package v6: the Route C term (0 on steps without branch targets).
                    "route_c_coef": route_c_coef,
                }
                task_rows = float(batch.batch["trsi_task_row_mask"].sum().item())
                sd_rows = float(batch.batch["trsi_sd_row_mask"].sum().item()) if plan.needs_teacher else 0.0
                has_work = (
                    (plan.task_coef > 0 and task_rows > 0)
                    or (plan.sd_coef > 0 and sd_rows > 0)
                    or (route_c_coef > 0 and route_c_rows > 0)
                )
                update_applied = False
                if has_work:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)
                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    for key in PEAK_MEMORY_METRICS:  # the fullest GPU sets the headroom, not the mean
                        if key in actor_output.non_tensor_batch:
                            actor_metrics[key] = float(np.max(actor_output.non_tensor_batch[key]))
                    metrics.update(actor_metrics)
                    if "trsi/optimizer_steps" in actor_metrics:  # k mini-batches: applied if any step was applied
                        update_applied = float(actor_metrics["trsi/optimizer_steps"]) > 0.0
                    else:
                        update_applied = float(actor_metrics.get("trsi/update_applied", 0.0)) > 0.5
                metrics["trsi/update_skipped"] = float(not has_work)
                self.schedule.record_step(phase=plan.phase, task_signal=task_signal, update_applied=update_applied)
                metrics["trsi/signal_steps_in_stage"] = float(self.schedule.signal_steps)
                fingerprint_after = self._state_fingerprint()
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and (self.global_step % self.config.trainer.val_freq == 0)
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)
                checkpoint_due = (
                    self.config.trainer.save_freq > 0
                    and self.global_step % self.config.trainer.save_freq == 0
                    and not stop_run  # a stopped run does not save the collapsed student
                )
                dump = self._dump_rollouts(batch, reward_metrics, unhealthy=bool(reasons), reasons=reasons)
                # The step record is written before this step's checkpoint: a run killed during or right after
                # the save resumes from that checkpoint at the next step and would never rewrite this record.
                record = self._step_record(
                    plan=plan,
                    metrics=metrics,
                    fingerprint_before=fingerprint_before,
                    fingerprint_after=fingerprint_after,
                    data_ids=data_ids,
                    has_work=has_work,
                    update_applied=update_applied,
                    checkpoint_due=checkpoint_due,
                    timing={"step_s": time.monotonic() - step_started, **timing_raw},
                )
                record["health"] = {
                    "unhealthy": bool(reasons),
                    "reasons": reasons,
                    "streak": int(self.guard.streak),
                    "in_window": int(in_window),
                }
                if dump is not None:
                    record["rollout_dump"] = os.path.relpath(dump, run_dir)
                recorder.write(record)
                if checkpoint_due:
                    with timer("save_checkpoint", timing_raw):
                        last_checkpoint = self._save_checkpoint()
                    recorder.write(
                        {
                            "kind": "checkpoint",
                            "step": int(self.global_step),
                            "checkpoint": last_checkpoint,
                            "seconds": round(float(timing_raw["save_checkpoint"]), 3),
                        }
                    )
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))
            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()
            if stop_run:  # after the stop step's full metrics reached the trackers (no checkpoint was saved)
                self._stop_for_collapse(recorder, reasons, loaded_checkpoint)
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)
        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            last_checkpoint = self._save_checkpoint()
            recorder.write({"kind": "final_checkpoint", "step": int(self.global_step), "checkpoint": last_checkpoint})
        marker = os.path.join(run_dir, COMPLETION_MARKER)
        write_json_atomic(
            marker,
            completion_payload(
                experiment_name=self.config.trainer.experiment_name,
                global_step=int(self.global_step),
                training_steps=int(self.training_steps),
                final_checkpoint=last_checkpoint,
                resumed_from_checkpoint=loaded_checkpoint,
                world_size=int(self.actor_rollout_ref_wg.world_size),
                teacher_versions=[version.to_dict() for version in self.schedule.history],
                step_records=recorder.path,
            ),
        )
        recorder.write({"kind": "run_complete", "step": int(self.global_step), "marker": marker})
        print(f"[Video-t-RSI] training complete: {marker}")
        self._close_tracker()

    def _create_logger(self):
        from ..utils.logger import Tracker

        return Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
