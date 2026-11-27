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
"""Per-step run records and the completion marker of a Video-t-RSI run.

Files (``<run_dir>`` is ``trainer.save_checkpoint_path``):

* ``<run_dir>/logs/trsi_steps.jsonl`` -- one JSON object per line: a
  ``run_start`` record per driver start (``resumed_from_step`` set when the
  run continued from a checkpoint) and one ``step`` record per global step
  with the phase, the teacher version and its correction gate, how the step
  consumed the teacher (``sd_mode``: ``surrogate`` = L_SD, ``reweight`` =
  GRPO with reward-anchored chunk factors; ``null`` on task steps), the
  correction magnitude, losses, gradient norm, whether the optimizer actually
  updated (Adam step counter and parameter fingerprint before/after the step)
  and the checkpoint written at that step.
* ``<run_dir>/TRAINING_COMPLETE.json`` -- written once, after the final
  checkpoint of a run that reached ``training_steps``.
* ``<run_dir>/COLLAPSE_DETECTED.json`` -- written when the collapse guard
  stopped the run (also recorded as a ``collapse_guard`` record).
* ``<run_dir>/logs/rollouts/step_NNNNNNN.jsonl`` -- decoded rollouts with
  rewards and advantages, every ``rsi.dump_rollouts_interval`` steps and at
  every unhealthy step.

These records track training progress and checkpoint resume history.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import time
import uuid
from typing import Any, Iterable, Optional


STEP_RECORD_SCHEMA = "video_trsi.step_record.v1"
COMPLETION_SCHEMA = "video_trsi.training_complete.v1"
STEP_RECORD_FILE = os.path.join("logs", "trsi_steps.jsonl")
COMPLETION_MARKER = "TRAINING_COMPLETE.json"
COLLAPSE_MARKER = "COLLAPSE_DETECTED.json"

_SUMMED = ("sum", "abs_sum", "numel")
_FINGERPRINT_MODULES = ("student", "student_as_teacher", "updated", "anchor")


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _json_safe(value: Any) -> Any:
    """Non-finite floats become strings, so a NaN loss is recorded instead of breaking the record."""

    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def combine_fingerprints(per_rank: Iterable[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Combine ``FSDPTRSIWorker.trsi_state_fingerprint`` results of all ranks (rank order)."""

    ranks = sorted(per_rank, key=lambda item: item.get("rank", {}).get("value", 0.0))
    if not ranks:
        return {}
    combined: dict[str, dict[str, float]] = {}
    for name in _FINGERPRINT_MODULES:
        parts = [rank[name] for rank in ranks]
        combined[name] = {key: math.fsum(part[key] for part in parts) for key in _SUMMED}
        # Buffers are replicated (FSDP does not shard them); every rank holds the same values.
        combined[name]["buffer_sum"] = float(parts[0]["buffer_sum"])
        combined[name]["buffer_sum_rank_spread"] = float(
            max(part["buffer_sum"] for part in parts) - min(part["buffer_sum"] for part in parts)
        )
    optim = [rank["optimizer"] for rank in ranks]
    combined["optimizer"] = {
        "step": max(part["step"] for part in optim),
        "step_min": min(part["step"] for part in optim),
        "exp_avg_sum": math.fsum(part["exp_avg_sum"] for part in optim),
        "exp_avg_sq_sum": math.fsum(part["exp_avg_sq_sum"] for part in optim),
        "states": math.fsum(part["states"] for part in optim),
    }
    combined["lr"] = {"value": float(ranks[0]["lr"]["value"])}
    combined["world"] = {"ranks": float(len(ranks))}
    return combined


def ids_digest(ids: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def correction_reason(*, needs_teacher: bool, route: str, version: Optional[dict[str, Any]], rule: Optional[dict[str, Any]]) -> str:
    """Why the Route-A correction is (not) applied at this step; never bypasses the gate."""

    if not needs_teacher:
        return "no_teacher_in_this_phase"
    if route != "A":
        return "route_base"
    if version is None:
        return "no_teacher_version"
    if not bool(version.get("correction_enabled")):
        return str(version.get("gate_reason"))
    if rule is not None and float(rule.get("base_mix", 1.0)) <= 0.0:
        return "base_mix_beta_zero"
    if rule is not None and float(rule.get("alpha", 0.0)) <= 0.0:
        return "alpha_zero"
    return "enabled"


_METRIC_GROUPS = {
    "task": (
        "trsi/task_signal",
        "trsi/group_valid_fraction",
        "trsi/task_loss",
        "trsi/task_clipfrac_higher",
        "trsi/task_clipfrac_lower",
    ),
    "teacher": (
        "trsi/teacher_correction_active",
        "trsi/teacher_alpha",
        "trsi/teacher_kl_to_base",
        "trsi/teacher_clip_fraction",
        "trsi/teacher_entropy",
        "trsi/teacher_max_log_ratio",
        "trsi/teacher_ratio_bound_violation",
        "trsi/teacher_correction_mean",
        "trsi/teacher_correction_abs_mean",
        "trsi/teacher_phase_degraded",
        "trsi/delta_base_mean",
        "trsi/view_answer_coverage",
        "trsi/view_temporal_coverage",
        "trsi/view_spatial_coverage",
    ),
    "distill": (
        "trsi/sd_row_fraction",
        "trsi/delta_mean",
        "trsi/delta_abs_mean",
        "trsi/chunks_per_response",
        "trsi/chunk_tokens_mean",
        "trsi/chunk_D_mean",
        "trsi/chunk_D_abs_mean",
        "trsi/chunk_G_mean",
        "trsi/chunk_G_abs_mean",
        "trsi/credit_clip_fraction",
        "trsi/sd_loss",
        "trsi/sd_ratio_mean",
        "trsi/sd_clipfrac_higher",
        "trsi/sd_clipfrac_lower",
        "trsi/verified_error_rate",
        # rsi.sd_mode=reweight (package v5): the teacher reweights the GRPO advantages.
        "trsi/reweight_token_fraction",
        "trsi/reweight_factor_mean",
        "trsi/reweight_factor_min",
        "trsi/reweight_factor_max",
        "trsi/reweight_up_fraction",
        "trsi/reweight_clipped_fraction",
        "trsi/reweight_factor_mean_pos",
        "trsi/reweight_factor_mean_neg",
        "trsi/reweight_adv_scale",
        "trsi/reweight_row_credit_mean",
        "trsi/reweight_row_credit_mean_pos",
        "trsi/reweight_row_credit_mean_neg",
    ),
    "update": (
        "actor/pg_loss",
        "actor/grad_norm",
        "actor/ppo_kl",
        "actor/entropy_loss",
        "trsi/ref_kl",
        "actor/lr",
        "trsi/update_applied",
        "trsi/optimizer_steps",
        "trsi/update_skipped",
        "trsi/loss_nonfinite",
        "trsi/empty_minibatch_skip",
    ),
    "perf": (
        "perf/mfu_actor",
        "perf/experience_peak_allocated_gb",
        "perf/update_peak_allocated_gb",
        "perf/train_peak_reserved_gb",
        "perf/max_memory_allocated_gb",
        "perf/max_memory_reserved_gb",
        "perf/cpu_memory_used_gb",
        "global_seqlen/update_sync_efficiency",
        "global_seqlen/experience_sync_efficiency",
    ),
}


def metric_groups(metrics: dict[str, Any]) -> dict[str, dict[str, Optional[float]]]:
    groups: dict[str, dict[str, Optional[float]]] = {}
    for group, names in _METRIC_GROUPS.items():
        groups[group] = {name: _finite_or_none(metrics[name]) for name in names if name in metrics}
    for group, prefix in (
        ("chunk_status", "trsi/chunk_status/"),
        ("reward", "reward/"),
        ("training_reward", "training_reward/"),  # v7: raw vs effective reward, true truncation gating
        ("health", "health/"),
        ("mismatch", "mismatch/"),
        ("route_c", "trsi/route_c_"),  # package v6: the inner loop (branches, continuations, targets, its loss)
    ):
        groups[group] = {name: _finite_or_none(value) for name, value in metrics.items() if name.startswith(prefix)}
    return groups


class StepRecorder:
    """Append-only JSONL writer; every line is flushed and fsynced (the run may be killed at any time).

    A run killed during a write leaves a last line without its newline.  The
    next write ends that torn line before appending the new record.
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.path = os.path.join(run_dir, STEP_RECORD_FILE)
        self.attempt = uuid.uuid4().hex[:12]
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        record = {
            "schema": STEP_RECORD_SCHEMA,
            "attempt": self.attempt,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **record,
        }
        line = json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True, allow_nan=False, default=str)
        with open(self.path, "a+b") as handle:  # writes always append; reads are allowed for the check below
            prefix = b""
            if handle.seek(0, os.SEEK_END) > 0:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    prefix = b"\n"
            handle.write(prefix + line.encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())


def write_json_atomic(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = f"{path}.tmp-{uuid.uuid4().hex}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def completion_payload(**fields: Any) -> dict[str, Any]:
    return {
        "schema": COMPLETION_SCHEMA,
        "status": "complete",
        "host": socket.gethostname(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **fields,
    }


__all__ = [
    "COLLAPSE_MARKER",
    "COMPLETION_MARKER",
    "STEP_RECORD_FILE",
    "STEP_RECORD_SCHEMA",
    "StepRecorder",
    "combine_fingerprints",
    "completion_payload",
    "correction_reason",
    "ids_digest",
    "metric_groups",
    "write_json_atomic",
]
