"""Stage schedules and persisted teacher-version state (method sections 13 and 15).

Two schedules are implemented for Route A.

``alternating`` (schedule 2, default).  Round ``r`` covers
``task_steps + distill_steps`` rollout steps::

    offset 0            snapshot_anchor   theta_r^-  <- current student
    offsets [0, T)      task phase        L_task only (GRPO on STGR-RL)
    offset T            snapshot_updated  theta_r^+  <- current student;
                                          freeze (theta_r^-, theta_r^+) as teacher version r
    offsets [T, T + D)  distill phase     fresh rollouts of the student, scored by teacher version r:
                                          sd_mode=surrogate: L_SD only (method section 12);
                                          sd_mode=reweight:  L_task (GRPO) with the advantages reweighted
                                          by the teacher's chunk credit (package v5)

The anchor/updated difference therefore reflects *only* task learning of
round ``r``.

``sd_mode`` sets the loss coefficients of the steps that use a teacher
(``StepPlan.task_coef`` / ``sd_coef``), not the phases or the events:

    phase     surrogate                  reweight
    task      task 1, SD 0, no teacher   task 1, SD 0, no teacher
    distill   task 0, SD 1, teacher      task 1, SD 0, teacher (reweighting)
    joint     task 1, SD lambda_SD       task 1, SD 0, teacher (reweighting)

``joint`` (schedule 1).  Stage ``r`` covers ``stage_steps`` steps trained with
``L_task + lambda_SD * L_SD`` (``sd_mode=reweight``: ``L_task`` with reweighted
advantages) against the frozen teacher version ``r``::

    stage 0 start   init_teacher      anchor <- updated <- theta_0, Delta = 0 (base teacher S_0)
    stage r start   refresh_teacher   anchor <- previous updated snapshot, updated <- student

The extrapolation direction of a joint refresh spans task learning *and* the
previous stage's distillation; it is recorded as
``direction_source="joint_task_and_distill"`` and must not be described as
driven by task feedback alone.

A correction is only enabled when its basis exists: the interval between the
snapshots must contain at least ``min_task_signal_steps`` rollout steps that
carried a non-zero task advantage and applied at least one optimizer update
(one update per rollout step in the base configs, several in the throughput
configuration).  Otherwise the version falls back to
the evidence-conditioned base teacher (``Delta = 0``).  Teacher parameters and
rules stay frozen within a stage; versions only change at stage boundaries.

Teacher health gate (``rsi.teacher_health_gate``).  When the interval's rollouts
degraded (``verl.trainer.trsi.health.phase_degradation``), the driver copies the
anchor into the updated snapshot instead of the student, and the version is
built with ``Delta = 0`` (gate reason ``task_phase_degraded``,
``base_snapshot="anchor"``): the next distillation phase is taught by the
student as it was before the degradation, and no correction extrapolates along
the degradation.
"""

from __future__ import annotations

import math
from dataclasses import MISSING, asdict, dataclass, field
from typing import Any, Optional


__all__ = [
    "EVENT_INIT_TEACHER",
    "EVENT_REFRESH_TEACHER",
    "EVENT_SNAPSHOT_ANCHOR",
    "EVENT_SNAPSHOT_UPDATED",
    "PHASE_DISTILL",
    "PHASE_JOINT",
    "PHASE_TASK",
    "RSISchedule",
    "StepPlan",
    "TeacherVersion",
]

PHASE_TASK = "task"
PHASE_DISTILL = "distill"
PHASE_JOINT = "joint"

EVENT_SNAPSHOT_ANCHOR = "snapshot_anchor"
EVENT_SNAPSHOT_UPDATED = "snapshot_updated"
EVENT_INIT_TEACHER = "init_teacher"
EVENT_REFRESH_TEACHER = "refresh_teacher"

SCHEDULE_STATE_VERSION = 1
SD_MODES = ("surrogate", "reweight")


@dataclass
class TeacherVersion:
    """Provenance of one frozen teacher version (the state the loop must carry)."""

    version: int
    round_index: int
    first_step: int
    """First global step whose rollouts are scored by this version."""
    anchor_step: int
    """Number of completed training (rollout) steps when the anchor snapshot was taken.  With k update
    mini-batches per rollout step the optimizer has made up to k times as many steps; the Adam step
    counter is in the step records' fingerprints."""
    updated_step: int
    """Number of completed training (rollout) steps when the updated snapshot was taken (see anchor_step)."""
    alpha: float
    correction_enabled: bool
    gate_reason: str
    direction_source: str
    task_signal_steps: int
    base_snapshot: str = "updated"
    """Which student the base teacher holds: ``updated`` (the student at updated_step) or ``anchor`` (the teacher
    health gate kept the student of the interval's start; updated_step == anchor_step then)."""
    degradation: Optional[str] = None
    """Why the teacher health gate judged the interval degraded (``None``: not degraded)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TeacherVersion":
        # Fields added later (base_snapshot, degradation) keep their defaults for states written before them.
        fields = cls.__dataclass_fields__
        required = [name for name, item in fields.items() if item.default is MISSING]
        missing = [name for name in required if name not in value]
        if missing:
            raise KeyError(f"Teacher version state lacks {missing}.")
        return cls(**{name: value[name] for name in fields if name in value})


@dataclass(frozen=True)
class StepPlan:
    step: int
    round_index: int
    phase: str
    offset: int
    events: tuple[str, ...]
    needs_teacher: bool
    task_coef: float
    sd_coef: float


@dataclass
class RSISchedule:
    mode: str = "alternating"
    task_steps: int = 50
    distill_steps: int = 50
    stage_steps: int = 100
    sd_coef: float = 1.0
    alpha: float = 0.5
    alpha_decay: float = 1.0
    alpha_min: float = 0.0
    min_task_signal_steps: int = 1
    total_steps: Optional[int] = None
    sd_mode: str = field(kw_only=True)
    """How teacher-using steps train (see the module docstring): ``surrogate`` (the method as specified) or
    ``reweight``; required, the trainer passes ``rsi.sd_mode``.  Not part of ``config_dict``: it changes the loss
    coefficients of the plans, not the phases, events or teacher versions this state describes, and the trainer
    checks it as a protocol setting on resume (``rsi.resume_allow_changes``), so checkpoints of either mode load."""

    signal_steps: int = field(default=0, init=False)
    current: Optional[TeacherVersion] = field(default=None, init=False)
    history: list[TeacherVersion] = field(default_factory=list, init=False)
    _pending_anchor_step: Optional[int] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.mode not in ("alternating", "joint"):
            raise ValueError(f"Unknown Video-t-RSI schedule {self.mode!r}.")
        if self.mode == "alternating" and (int(self.task_steps) < 1 or int(self.distill_steps) < 1):
            raise ValueError("The alternating schedule needs task_steps >= 1 and distill_steps >= 1.")
        if self.mode == "joint" and int(self.stage_steps) < 1:
            raise ValueError("The joint schedule needs stage_steps >= 1.")
        if not (math.isfinite(float(self.sd_coef)) and float(self.sd_coef) >= 0.0):
            raise ValueError("sd_coef must be finite and non-negative.")
        if not (math.isfinite(float(self.alpha)) and float(self.alpha) >= 0.0):
            raise ValueError("alpha must be finite and non-negative.")
        if not (math.isfinite(float(self.alpha_decay)) and 0.0 <= float(self.alpha_decay) <= 1.0):
            raise ValueError("alpha_decay must be in [0, 1].")
        if not (math.isfinite(float(self.alpha_min)) and 0.0 <= float(self.alpha_min) <= float(self.alpha)):
            raise ValueError("alpha_min must be in [0, alpha].")
        if int(self.min_task_signal_steps) < 0:
            raise ValueError("min_task_signal_steps must be non-negative.")
        if self.sd_mode not in SD_MODES:
            raise ValueError(f"sd_mode must be one of {SD_MODES}, got {self.sd_mode!r}.")

    # ------------------------------------------------------------------ plan
    @property
    def cycle(self) -> int:
        return int(self.task_steps) + int(self.distill_steps) if self.mode == "alternating" else int(self.stage_steps)

    def plan(self, step: int) -> StepPlan:
        """Deterministic plan of the 1-indexed global ``step``."""

        if int(step) < 1:
            raise ValueError("Global steps are 1-indexed.")
        round_index, offset = divmod(int(step) - 1, self.cycle)
        reweight = self.sd_mode == "reweight"
        if self.mode == "alternating":
            if offset < int(self.task_steps):
                events = (EVENT_SNAPSHOT_ANCHOR,) if offset == 0 else ()
                return StepPlan(step, round_index, PHASE_TASK, offset, events, False, 1.0, 0.0)
            events = (EVENT_SNAPSHOT_UPDATED,) if offset == int(self.task_steps) else ()
            task_coef, sd_coef = (1.0, 0.0) if reweight else (0.0, 1.0)
            return StepPlan(step, round_index, PHASE_DISTILL, offset, events, True, task_coef, sd_coef)
        events: tuple[str, ...] = ()
        if offset == 0:
            events = (EVENT_INIT_TEACHER,) if round_index == 0 else (EVENT_REFRESH_TEACHER,)
        return StepPlan(step, round_index, PHASE_JOINT, offset, events, True, 1.0, 0.0 if reweight else float(self.sd_coef))

    def alpha_for_round(self, round_index: int) -> float:
        """Alpha of the teacher version built in ``round_index`` (alternating round or joint stage).

        Alpha decays once per teacher version that can carry a correction, so
        the first such version uses the undecayed alpha in both schedules:
        alternating round 0 and joint stage 1 (joint stage 0 is the base teacher).
        """

        exponent = int(round_index) if self.mode == "alternating" else max(int(round_index) - 1, 0)
        return max(float(self.alpha_min), float(self.alpha) * float(self.alpha_decay) ** exponent)

    def distill_step_count(self, total_steps: int) -> int:
        """How many of ``total_steps`` steps score rollouts with a teacher."""

        return sum(1 for step in range(1, int(total_steps) + 1) if self.plan(step).needs_teacher)

    # ---------------------------------------------------------------- events
    def _new_version(
        self,
        *,
        step: int,
        round_index: int,
        anchor_step: int,
        direction: str,
        initial: bool,
        degradation: Optional[str] = None,
    ) -> TeacherVersion:
        alpha = self.alpha_for_round(round_index)
        if initial:
            enabled, reason = False, "initial_stage"
        elif degradation is not None:
            enabled, reason = False, "task_phase_degraded"
        elif alpha <= 0.0:
            enabled, reason = False, "alpha_zero"
        elif self.signal_steps < int(self.min_task_signal_steps):
            enabled, reason = False, "insufficient_task_signal"
        else:
            enabled, reason = True, "enabled"
        version = TeacherVersion(
            version=len(self.history),
            round_index=int(round_index),
            first_step=int(step),
            anchor_step=int(anchor_step),
            updated_step=int(anchor_step) if degradation is not None else int(step) - 1,
            alpha=float(alpha) if enabled else 0.0,
            correction_enabled=enabled,
            gate_reason=reason,
            direction_source=direction if not initial else "none",
            task_signal_steps=int(self.signal_steps),
            base_snapshot="anchor" if degradation is not None else "updated",
            degradation=degradation,
        )
        self.current = version
        self.history.append(version)
        return version

    def apply_event(self, event: str, step: int, degradation: Optional[str] = None) -> Optional[TeacherVersion]:
        """Update the bookkeeping for ``event`` executed before ``step``'s rollout.

        ``degradation`` (updated snapshot / joint refresh only): the teacher health gate's reason; the driver
        then copied the anchor into the updated snapshot and the version carries no correction.
        """

        plan = self.plan(step)
        if event not in plan.events:
            raise ValueError(f"Event {event!r} is not scheduled before step {step}.")
        if degradation is not None and event not in (EVENT_SNAPSHOT_UPDATED, EVENT_REFRESH_TEACHER):
            raise ValueError(f"A degraded interval only affects the updated snapshot, not {event!r}.")
        if event == EVENT_SNAPSHOT_ANCHOR:
            self._pending_anchor_step = int(step) - 1
            self.signal_steps = 0
            return None
        if event == EVENT_SNAPSHOT_UPDATED:
            if self._pending_anchor_step is None:
                raise RuntimeError("The updated snapshot requires an anchor snapshot from the same round.")
            version = self._new_version(
                step=step,
                round_index=plan.round_index,
                anchor_step=self._pending_anchor_step,
                direction="task_phase",
                initial=False,
                degradation=degradation,
            )
            return version
        if event == EVENT_INIT_TEACHER:
            self.signal_steps = 0
            return self._new_version(step=step, round_index=0, anchor_step=int(step) - 1, direction="none", initial=True)
        if event == EVENT_REFRESH_TEACHER:
            if self.current is None:
                raise RuntimeError("A joint refresh requires the previous stage's teacher version.")
            version = self._new_version(
                step=step,
                round_index=plan.round_index,
                anchor_step=self.current.updated_step,
                direction="joint_task_and_distill",
                initial=False,
                degradation=degradation,
            )
            self.signal_steps = 0
            return version
        raise ValueError(f"Unknown schedule event {event!r}.")

    def record_step(self, *, phase: str, task_signal: bool, update_applied: bool) -> None:
        """Count rollout steps whose applied optimizer update(s) carried task feedback (the correction basis).

        Distillation-phase steps never count, also with ``sd_mode=reweight`` (they train on the task reward
        then): they lie between the updated snapshot and the next anchor, outside every snapshot interval.
        """

        if phase in (PHASE_TASK, PHASE_JOINT) and task_signal and update_applied:
            self.signal_steps += 1

    def teacher_for_step(self, step: int) -> Optional[TeacherVersion]:
        plan = self.plan(step)
        if not plan.needs_teacher:
            return None
        if self.current is None:
            raise RuntimeError(f"Step {step} needs a teacher, but no teacher version has been built.")
        return self.current

    # ------------------------------------------------------------- persistence
    def config_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "task_steps": int(self.task_steps),
            "distill_steps": int(self.distill_steps),
            "stage_steps": int(self.stage_steps),
            "sd_coef": float(self.sd_coef),
            "alpha": float(self.alpha),
            "alpha_decay": float(self.alpha_decay),
            "alpha_min": float(self.alpha_min),
            "min_task_signal_steps": int(self.min_task_signal_steps),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEDULE_STATE_VERSION,
            "config": self.config_dict(),
            "signal_steps": int(self.signal_steps),
            "pending_anchor_step": self._pending_anchor_step,
            "current": None if self.current is None else self.current.to_dict(),
            "history": [version.to_dict() for version in self.history],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != SCHEDULE_STATE_VERSION:
            raise ValueError("Unsupported Video-t-RSI schedule state version.")
        saved = state.get("config")
        if saved != self.config_dict():
            raise ValueError(
                "The Video-t-RSI schedule configuration changed since this checkpoint was written: "
                f"saved={saved}, current={self.config_dict()}."
            )
        self.signal_steps = int(state["signal_steps"])
        pending = state.get("pending_anchor_step")
        self._pending_anchor_step = None if pending is None else int(pending)
        self.current = None if state.get("current") is None else TeacherVersion.from_dict(state["current"])
        self.history = [TeacherVersion.from_dict(item) for item in state.get("history", [])]
