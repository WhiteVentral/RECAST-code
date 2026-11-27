"""Run health of Video-t-RSI: per-step rollout statistics, the collapse guard and the teacher health gate.

Per step (``health/*`` metrics, recorded in ``logs/trsi_steps.jsonl``)::

    response_length_mean / _min / _p10 / _max   tokens of the sampled responses (EOS included)
    truncated_fraction                   responses that reached data.max_response_length
    format_rate                          mean format reward (when the reward reports format_reward)
    reward_mean                          mean overall reward
    sampled_nll                          -mean log pi_old(y_t) over response tokens (an entropy estimate)
    zero_signal_group_fraction           GRPO groups whose advantages are all zero
    adv_length_corr                      Pearson correlation of the GRPO advantage with the response length
                                         over rows with signal: negative = shorter responses score better
                                         (a brevity drift), the early sign of the collapse of run
                                         video_trsi_4n8g_mb2_full_20261019_170451
    invalid_row_fraction                 rows left out of all of the above (media or ground truth unusable:
                                         every reward component, format included, is 0 for them)

A step is *unhealthy* when

* its format rate or its mean response length is below
  ``rsi.health_min_format_rate`` / ``rsi.health_min_response_length`` (a
  collapse to short, unformatted responses), or
* more than ``rsi.health_max_truncated_fraction`` of its responses reached
  data.max_response_length (a runaway to long responses that are cut off
  before their answer: run v4 stopped at step 725 on the format rate, after
  its responses had grown to the 768-token limit in the GRPO phase 701-750), or
* its KL to the initialization model jumped: ``health/ref_kl`` above both
  ``rsi.health_ref_kl_jump_floor`` and ``rsi.health_ref_kl_jump_ratio`` times
  the median of the previous ``rsi.health_ref_kl_window`` steps (checked once 5
  earlier values exist).

The guard stops the run once ``rsi.guard_patience`` of the last
``rsi.guard_window`` steps were unhealthy and the current step is unhealthy
(window = patience: consecutive steps only, the guard of package v4).  A
collapsed policy gives GRPO no signal and its teacher no information, so
continuing only consumes compute; an intermittent runaway (healthy steps in
between) is caught by the window.

The teacher health gate compares the first and the last steps of a task phase
(``phase_degradation``): format rate or mean length fell, or (package v5) mean
length or truncated fraction rose.  It is a decision about the teacher, not
about the student's loss: a degraded phase still trains, but the next teacher
version is not extrapolated along the degradation (see ``RSISchedule``).

Rollout statistics describe the policy *before* the step's update: the rollouts
of step ``s`` were sampled by the student saved in the checkpoint of step ``s - 1``.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import torch


__all__ = [
    "CollapseGuard",
    "HealthThresholds",
    "PhaseHealth",
    "REF_KL_MIN_HISTORY",
    "TrainingCollapseError",
    "format_steps",
    "phase_degradation",
    "rollout_health",
    "unhealthy_reasons",
]


class TrainingCollapseError(RuntimeError):
    """Raised by the driver when the collapse guard stops a run."""


REF_KL_MIN_HISTORY = 5
"""Earlier health/ref_kl values needed before a jump is judged (the start of a run or of a resumed attempt)."""


@dataclass(frozen=True)
class HealthThresholds:
    """The unhealthy-step criteria (``rsi.health_*``); ``None`` switches a criterion off."""

    min_format_rate: Optional[float]
    min_response_length: Optional[float]
    max_truncated_fraction: Optional[float]
    ref_kl_jump_ratio: Optional[float]
    ref_kl_jump_floor: float


def _mean(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else None


def rollout_health(
    *,
    response_mask: torch.Tensor,
    reward_metrics: dict[str, Sequence[float]],
    old_log_probs: Optional[torch.Tensor],
    max_response_length: int,
    advantages: Optional[torch.Tensor] = None,
    uids: Optional[Sequence[Any]] = None,
    row_valid: Optional[Sequence[bool]] = None,
    truncated: Optional[Sequence[bool]] = None,
) -> dict[str, float]:
    """``health/*`` metrics of one rollout batch (``reward_metrics``: per-row lists from the reward function).

    ``row_valid``: the rows that count (the trainer passes its GRPO group validity).  Rows whose media could not
    be read or whose ground truth is unusable get reward 0 for every component, format included; counting them
    would make a media outage look like a collapse.  Their share is ``health/invalid_row_fraction``.
    """

    # v7 uses the actual generation termination.  Keep the old length proxy
    # only for legacy callers/checkpoints; report it separately for comparison.
    if truncated is not None:
        truncated = np.asarray(truncated, dtype=bool)
        if truncated.shape != (int(response_mask.shape[0]),):
            raise ValueError("truncated must contain one termination flag per response.")
    metrics: dict[str, float] = {}
    if row_valid is not None:
        keep = np.asarray(row_valid, dtype=bool)
        if keep.shape != (int(response_mask.shape[0]),):
            raise ValueError(f"row_valid must have one entry per row, got shape {keep.shape}.")
        metrics["health/invalid_row_fraction"] = float(1.0 - keep.mean()) if keep.size else 0.0
        index = torch.as_tensor(np.flatnonzero(keep), dtype=torch.long)
        response_mask = response_mask[index.to(response_mask.device)]
        if old_log_probs is not None:
            old_log_probs = old_log_probs[index.to(old_log_probs.device)]
        if advantages is not None:
            advantages = advantages[index.to(advantages.device)]
        if uids is not None:
            uids = np.asarray(uids, dtype=object)[keep]
        if truncated is not None:
            truncated = truncated[keep]
        kept = np.flatnonzero(keep).tolist()
        reward_metrics = {
            key: [value[row] for row in kept]
            if isinstance(value, (list, tuple, np.ndarray)) and len(value) == keep.size
            else value
            for key, value in reward_metrics.items()
        }
    mask = response_mask.to(torch.float32)
    lengths = mask.sum(dim=-1)
    if lengths.numel():
        metrics["health/response_length_mean"] = float(lengths.mean().item())
        metrics["health/response_length_min"] = float(lengths.min().item())
        metrics["health/response_length_p10"] = float(torch.quantile(lengths, 0.1).item())
        metrics["health/response_length_max"] = float(lengths.max().item())
        at_limit = float((lengths >= float(max_response_length)).float().mean().item())
        metrics["health/truncated_fraction"] = at_limit if truncated is None else float(truncated.mean())
        if truncated is not None:
            metrics["health/length_limit_fraction"] = at_limit
    for key, name in (("format_reward", "health/format_rate"), ("overall", "health/reward_mean")):
        if key in reward_metrics:
            value = _mean(list(reward_metrics[key]))
            if value is not None:
                metrics[name] = value
    if old_log_probs is not None and float(mask.sum().item()) > 0:
        metrics["health/sampled_nll"] = float((-(old_log_probs.float() * mask).sum() / mask.sum()).item())
    if advantages is not None and uids is not None and len(uids):
        signal = advantages.abs().sum(dim=-1).gt(0).cpu().numpy()
        uids = np.asarray(uids, dtype=object)
        groups = list(dict.fromkeys(uids.tolist()))
        silent = sum(1 for uid in groups if not bool(signal[uids == uid].any()))
        metrics["health/zero_signal_group_fraction"] = float(silent / len(groups))
        row_advantage = (advantages.float() * mask).sum(dim=-1) / lengths.clamp_min(1.0)
        active = torch.as_tensor(signal, dtype=torch.bool, device=row_advantage.device)
        if int(active.sum().item()) >= 3:
            x, y = row_advantage[active], lengths[active]
            x, y = x - x.mean(), y - y.mean()
            denominator = float((x.norm() * y.norm()).item())
            if denominator > 0:
                metrics["health/adv_length_corr"] = float((x * y).sum().item() / denominator)
    return metrics


def unhealthy_reasons(
    metrics: dict[str, Any], thresholds: HealthThresholds, *, ref_kl_history: Sequence[Optional[float]]
) -> list[str]:
    """Why the step is unhealthy (empty: healthy or nothing to check).

    ``ref_kl_history``: health/ref_kl of the previous steps (the guard's recent history), for the jump check.
    """

    reasons = []
    rate = metrics.get("health/format_rate")
    if thresholds.min_format_rate is not None and rate is not None and float(rate) < float(thresholds.min_format_rate):
        reasons.append(f"format rate {float(rate):.3f} < {float(thresholds.min_format_rate):g}")
    length = metrics.get("health/response_length_mean")
    if (
        thresholds.min_response_length is not None
        and length is not None
        and float(length) < float(thresholds.min_response_length)
    ):
        reasons.append(f"mean response length {float(length):.1f} < {float(thresholds.min_response_length):g} tokens")
    truncated = metrics.get("health/truncated_fraction")
    if (
        thresholds.max_truncated_fraction is not None
        and truncated is not None
        and float(truncated) > float(thresholds.max_truncated_fraction)
    ):
        reasons.append(f"truncated fraction {float(truncated):.3f} > {float(thresholds.max_truncated_fraction):g}")
    kl = metrics.get("health/ref_kl")
    history = [float(value) for value in ref_kl_history if value is not None and math.isfinite(float(value))]
    if thresholds.ref_kl_jump_ratio is not None and kl is not None and len(history) >= REF_KL_MIN_HISTORY:
        median = float(statistics.median(history))
        limit = max(float(thresholds.ref_kl_jump_floor), float(thresholds.ref_kl_jump_ratio) * median)
        if float(kl) > limit:
            reasons.append(
                f"KL to the initialization model {float(kl):.3f} > {limit:.3f} (the larger of "
                f"{float(thresholds.ref_kl_jump_floor):g} and {float(thresholds.ref_kl_jump_ratio):g} x the median "
                f"{median:.3f} of the previous {len(history)} steps)"
            )
    return reasons


def format_steps(steps: Sequence[int]) -> str:
    """``[5, 6, 7, 9]`` -> ``"5-7, 9"``."""

    ranges: list[tuple[int, int]] = []
    for step in sorted(int(value) for value in steps):
        if ranges and step == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], step)
        else:
            ranges.append((step, step))
    return ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in ranges)


@dataclass
class CollapseGuard:
    """Stops the run when ``patience`` of the last ``window`` steps were unhealthy, the current one included.

    ``window`` below ``patience`` means ``patience``: consecutive steps only.  ``streak`` counts
    the consecutive unhealthy steps up to the latest one (reported, not used for the stop).  The window is read
    from ``history``, which holds one entry per step, so a checkpoint saved during unhealthy steps (or written
    by package v4, whose entries lack the newer fields) continues it.
    """

    patience: int
    window: int
    """Steps over which ``patience`` unhealthy ones stop the run (below ``patience``: ``patience``)."""
    history_size: int
    """Entries kept (at least the window); the trainer keeps enough for the KL-jump median."""
    streak: int = 0
    streak_start: Optional[int] = None
    history: list[dict[str, Any]] = field(default_factory=list)
    """The latest steps' health (for the window, the KL-jump median and the stop report)."""

    @property
    def width(self) -> int:
        return max(int(self.window), int(self.patience), 1)

    def update(self, step: int, metrics: dict[str, Any], reasons: list[str]) -> bool:
        entry = {
            "step": int(step),
            "format_rate": metrics.get("health/format_rate"),
            "response_length_mean": metrics.get("health/response_length_mean"),
            "truncated_fraction": metrics.get("health/truncated_fraction"),
            "ref_kl": metrics.get("health/ref_kl"),
            "reward_mean": metrics.get("health/reward_mean"),
            "reasons": list(reasons),
        }
        self.history = (self.history + [entry])[-max(int(self.history_size), self.width) :]
        if reasons:
            if self.streak == 0:
                self.streak_start = int(step)
            self.streak += 1
        else:
            self.streak, self.streak_start = 0, None
        return bool(reasons) and int(self.patience) > 0 and len(self.unhealthy_steps(step)) >= int(self.patience)

    def unhealthy_steps(self, step: Optional[int] = None) -> list[int]:
        """Unhealthy steps among the ``width`` steps ending at ``step`` (default: the latest recorded step)."""

        if not self.history:
            return []
        last = int(self.history[-1]["step"]) if step is None else int(step)
        return [
            int(item["step"])
            for item in self.history
            if last - self.width < int(item["step"]) <= last and item.get("reasons")
        ]

    def recent_values(self, name: str, count: int) -> list[Optional[float]]:
        """``name`` of the latest ``count`` entries (for the KL-jump median; missing values are None)."""

        return [item.get(name) for item in self.history[-int(count) :]] if int(count) > 0 else []

    def state_dict(self) -> dict[str, Any]:
        return {"streak": int(self.streak), "streak_start": self.streak_start, "history": [dict(item) for item in self.history]}

    def load_state_dict(self, state: Optional[dict[str, Any]]) -> None:
        """Continue the streak and window of a checkpoint saved during unhealthy steps (before package v4: none)."""

        state = state or {}
        self.streak = int(state.get("streak", 0))
        start = state.get("streak_start")
        self.streak_start = None if start is None else int(start)
        self.history = [dict(item) for item in state.get("history", [])][-max(int(self.history_size), self.width) :]


@dataclass
class PhaseHealth:
    """Format rate, mean response length and truncated fraction of the steps of the current teacher interval."""

    steps: list[dict[str, Any]] = field(default_factory=list)

    def add(self, step: int, metrics: dict[str, Any]) -> None:
        self.steps.append(
            {
                "step": int(step),
                "format_rate": metrics.get("health/format_rate"),
                "response_length_mean": metrics.get("health/response_length_mean"),
                "truncated_fraction": metrics.get("health/truncated_fraction"),
            }
        )

    def reset(self) -> None:
        self.steps = []

    def state_dict(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self.steps]

    def load_state_dict(self, state: Optional[list[dict[str, Any]]]) -> None:
        self.steps = [dict(item) for item in (state or [])]


def phase_degradation(
    steps: Sequence[dict[str, Any]],
    *,
    window: int,
    max_format_drop: float,
    max_length_drop: float,
    max_length_rise: Optional[float],
    max_truncation_rise: Optional[float],
) -> Optional[str]:
    """Why the task phase degraded between its first and last ``window`` steps, or ``None``.

    Degraded: the format rate fell by more than ``max_format_drop``, the mean length fell by more than the
    fraction ``max_length_drop``, rose by more than the fraction ``max_length_rise``, or the truncated fraction
    rose by more than ``max_truncation_rise`` (``None``: not checked; steps recorded before package v5 have no
    truncated fraction).  The two windows never overlap: a phase of ``k`` recorded steps compares
    ``min(window, k // 2)`` steps at each end, so a phase (or the part of it seen after a resume) with fewer
    than 2 steps is never judged degraded.
    """

    width = min(int(window), len(steps) // 2)
    if width < 1:
        return None
    first, last = steps[:width], steps[-width:]
    rate_first = _mean([item.get("format_rate") for item in first])
    rate_last = _mean([item.get("format_rate") for item in last])
    if rate_first is not None and rate_last is not None and rate_last < rate_first - float(max_format_drop):
        return (
            f"format rate fell from {rate_first:.3f} (steps {first[0]['step']}-{first[-1]['step']}) to "
            f"{rate_last:.3f} (steps {last[0]['step']}-{last[-1]['step']}), more than {float(max_format_drop):g}"
        )
    length_first = _mean([item.get("response_length_mean") for item in first])
    length_last = _mean([item.get("response_length_mean") for item in last])
    if (
        length_first is not None
        and length_last is not None
        and length_first > 0
        and length_last < (1.0 - float(max_length_drop)) * length_first
    ):
        return (
            f"mean response length fell from {length_first:.1f} (steps {first[0]['step']}-{first[-1]['step']}) to "
            f"{length_last:.1f} (steps {last[0]['step']}-{last[-1]['step']}), by more than {float(max_length_drop):.0%}"
        )
    if (
        max_length_rise is not None
        and length_first is not None
        and length_last is not None
        and length_first > 0
        and length_last > (1.0 + float(max_length_rise)) * length_first
    ):
        return (
            f"mean response length rose from {length_first:.1f} (steps {first[0]['step']}-{first[-1]['step']}) to "
            f"{length_last:.1f} (steps {last[0]['step']}-{last[-1]['step']}), by more than {float(max_length_rise):.0%}"
        )
    truncated_first = _mean([item.get("truncated_fraction") for item in first])
    truncated_last = _mean([item.get("truncated_fraction") for item in last])
    if (
        max_truncation_rise is not None
        and truncated_first is not None
        and truncated_last is not None
        and truncated_last > truncated_first + float(max_truncation_rise)
    ):
        return (
            f"truncated fraction rose from {truncated_first:.3f} (steps {first[0]['step']}-{first[-1]['step']}) to "
            f"{truncated_last:.3f} (steps {last[0]['step']}-{last[-1]['step']}), more than {float(max_truncation_rise):g}"
        )
    return None
