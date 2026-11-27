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
"""Configuration of Video-t-RSI training (token-level recursive teacher updates)."""

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..workers.config import WorkerConfig
from .config import AlgorithmConfig, DataConfig, TrainerConfig, recursive_post_init
from .trsi.credit import REWEIGHT_CENTER_MODES
from .trsi.health import REF_KL_MIN_HISTORY
from .trsi.route_c import INNER_LOOP_MODES, RouteCSettings


SUPPORTED_ROUTES = ("A", "base")
SUPPORTED_SCHEDULES = ("alternating", "joint")
SUPPORTED_VIEWS = ("answer", "temporal", "spatial")
SUPPORTED_TASK_LOSS_AGG = ("token_mean", "seq_mean_token_mean")
SUPPORTED_SEED_MODES = ("per_step", "fixed")
SUPPORTED_ROLLOUT_CORRECTIONS = ("none", "tis")
SUPPORTED_SD_MODES = ("reweight", "surrogate")


@dataclass
class TRSIConfig:
    # ---------------------------------------------------------------- route
    route: str = "A"
    """``A``: training-trajectory extrapolation (section 6).  ``base``: the
    evidence-conditioned base teacher only (``Delta = 0``), an ablation that keeps
    every other component identical."""
    schedule: str = "alternating"
    """``alternating`` (schedule 2: task phase, then distillation phase) or
    ``joint`` (schedule 1: L_task + lambda_SD L_SD with periodic teacher refresh)."""
    task_steps_per_round: int = 50
    """Alternating schedule: GRPO-only rollout steps per round (theta^- -> theta^+)."""
    distill_steps_per_round: int = 50
    """Alternating schedule: distillation rollout steps per round."""
    joint_stage_steps: int = 100
    """Joint schedule: rollout steps per teacher stage."""
    sd_coef: float = 1.0
    """lambda_SD of the joint schedule with ``sd_mode=surrogate`` (the alternating distillation phase uses L_SD
    alone).  Not used with ``sd_mode=reweight``."""
    sd_mode: str = "reweight"
    """How the steps that use a teacher consume it.

    ``reweight`` (package v5, reward-anchored as in RLSD and MeSD): the step trains GRPO on the task reward, and
    the advantage of every token is multiplied by the factor of its chunk,
    ``clip(exp(sign(A_i) * (G_{i,k} - Gbar_i)), 1 - reweight_clip, 1 + reweight_clip)``, with G the chunk credit
    of the Route-A teacher (method sections 11-12) and Gbar_i its token-weighted mean over response i
    (``reweight_center``).  The reward sets the direction of every update and how much each response is
    pushed; the teacher only shifts the magnitude between the chunks of a response.  (MeSD mixed its factor
    with 1 by a weight annealed from 0.5 to 0 over 1,000 steps, so its factors stayed within [0.9, 1.1];
    ``reweight_clip=0.1`` gives that range.)

    ``surrogate`` (method section 12, packages v1-v4): the step optimizes L_SD, a PPO-style surrogate with the
    clipped chunk credit as its advantage and no task loss (``sd_coef`` x L_SD added to L_task in the joint
    schedule), so the teacher alone sets the direction during a distillation phase."""
    reweight_clip: float = 0.2
    """c of the reweighting factor, in [0, 1): factors lie in [1 - c, 1 + c]; 0 trains plain GRPO on those steps."""
    reweight_center: str = "row"
    """``row``: the factor uses G minus the response's token-weighted mean G, so the factors of a response average
    about 1.  G carries an offset shared by the whole response: delta = log q - log pi_old is measured on the
    student's own samples (drawn from pi_old itself, its expectation would be -KL(pi_old || q) <= 0; top-p/top-k
    sampling breaks that identity).  Without centring, a negative offset lowers the factors of every response
    with A > 0 and raises those with A < 0, which pushes the tokens a GRPO group shares down at every reweighted
    step (a positive offset pushes them up).  ``none``: G as it is (the formula of the v5 plan;
    ``trsi/reweight_row_credit_mean`` shows the offset it carries)."""

    # ------------------------------------------------- Route C inner loop (v6)
    inner_loop: str = "route_c"
    """``route_c`` (package v6): on every teacher-scored step, before its update, find the least absorbed chunk of
    every row with distillation targets (the largest gap between the teacher's and the student's distributions over
    the student's top candidates), branch once at that position, sample continuations of each candidate with the
    student and score them with the task reward, and train the student toward the reward-tilted candidate
    distribution there (``verl.trainer.trsi.route_c``).  The tilt's base is the student, so the continuation
    rewards set its direction.  Needs ``worker.rollout.temperature == rsi.teacher_temperature``.
    ``none``: package v5 (no inner loop)."""
    route_c_candidates: int = 4
    """k: the student's top-k tokens at a branch position (from the trainer's old-log-prob forward)."""
    route_c_continuations: int = 2
    """N: continuations sampled per candidate (a candidate that ends the response is scored once)."""
    route_c_min_gap: float = 0.05
    """A row is branched only where its largest eligible gap u_t (nats, over the k + 1 buckets) is at least this."""
    route_c_min_coverage: float = 0.5
    """Positions whose k candidates cover less of the student's probability are not branched."""
    route_c_value_clip: float = 1.0
    """c_Q: bound of Q(c_j) - Q0 in reward units (the STGR reward of a response lies in [0, 3])."""
    route_c_kappa: float = 1.0
    """kappa: tilt b_j = kappa * w * clip(Q(c_j) - Q0, -c_Q, c_Q); since |b_j| <= kappa * c_Q, the target changes the
    probability ratio of two candidates by at most exp(2 * kappa * c_Q)."""
    route_c_coef: float = 1.0
    """lambda_C: weight of the mean bucket KL(q^C || pi_theta) over the branched rows in the update."""

    # ------------------------------------------------------- Route A operator
    extrapolation_alpha: float = 0.5
    """alpha >= 0: extrapolation strength along log p_{theta+} - log p_{theta-}."""
    extrapolation_alpha_decay: float = 1.0
    """Per-round multiplicative decay of alpha in [0, 1] (1 keeps alpha constant)."""
    extrapolation_alpha_min: float = 0.0
    """Lower bound of the decayed alpha."""
    correction_clip: Optional[float] = 1.0
    """c_Delta: bound on the additive correction; q/p_b stays in [e^{-2c}, e^{2c}]."""
    base_mix_beta: float = 1.0
    """beta in [0, 1]: q_used = (1 - beta) p_b + beta q_prop (0 disables the correction)."""
    teacher_temperature: float = 1.0
    """tau: softmax temperature of every teacher view (the draft fixes tau = 1)."""
    min_task_signal_steps: int = 1
    """A correction is enabled only if the snapshot interval contains at least this
    many rollout steps with a non-zero task advantage and at least one applied
    optimizer update (one update per rollout step in the base configs)."""

    # ---------------------------------------------------------------- views
    views: list[str] = field(default_factory=lambda: list(SUPPORTED_VIEWS))
    """Evidence views constructed when valid: A (answer), AT (temporal), AS (spatial)."""
    teacher_max_prompt_length: int = 18432
    teacher_fp32_logits: bool = True
    """Evaluate teacher logits in float32 at response positions only."""
    teacher_vocab_chunk: int = 16384
    teacher_position_chunk: int = 128

    # ------------------------------------------------------------- chunking
    chunk_min_tokens: int = 4
    chunk_max_tokens: int = 128
    chunk_fallback_tokens: int = 32

    # --------------------------------------------------------------- credit
    credit_horizon: int = 2
    """H: maximum number of subsequent chunks in the chunk return-to-go."""
    credit_discount: float = 0.8
    """gamma in (0, 1]."""
    credit_clip: float = 1.0
    """c: symmetric clip of G before it becomes the distillation advantage."""

    # ----------------------------------------------------------------- loss
    task_clip_low: float = 0.2
    task_clip_high: float = 0.2
    sd_clip_low: float = 0.2
    sd_clip_high: float = 0.2
    task_loss_agg: str = "seq_mean_token_mean"
    """How the GRPO task surrogate is averaged over an optimizer mini-batch.  ``seq_mean_token_mean``: mean over
    the tokens of each response, then over responses (MeSD's aggregation; MeSD trained the full epoch of 2,327
    steps with it, 1,327 of them pure GRPO, without a collapse).  Every response has the same total weight, so
    a long response cannot dominate the update; each token is weighted by 1 / |o_i|, so a prefix shared by a
    group is pushed down when its short responses score lower.  ``token_mean`` (package v4): sum over all valid
    response tokens of the mini-batch divided by their number (all ranks), as in DAPO.  A shared token gets no
    net update, but a response's weight grows with its length, so long responses dominate the update while
    lengths increase."""

    zero_truncated_rewards: bool = False
    """Package v7 training-reward protocol: replace a response's effective reward by zero when generation
    metadata says it exhausted its response-token budget without terminating. Apply the same rule before
    GRPO group normalization and when estimating Route C candidate values. Raw STGR scores and evaluation
    stay unchanged. Requires explicit, valid generation termination metadata; response length alone is not
    sufficient. False preserves the legacy reward protocol of packages v1-v6; shipped v7 YAMLs enable it."""

    task_adv_std_floor: float = 0.0
    """GRPO advantage (r - mean) / (max(std, floor) + eps) per group.  0 keeps GRPO's (r - mean) / (std + eps):
    with continuous rewards and n = 4, a group whose rewards differ only by noise then gets advantages of
    about +-1 like a group with a real difference; a floor (e.g. 0.1 on the summed STGR reward) scales such
    near-ties down (Dr. GRPO and later work drop or bound the division by the group std)."""

    # ------------------------------------------------- rollout noise and IS
    rollout_seed_mode: str = "per_step"
    """``per_step``: every rollout request gets its own seed derived from (worker.rollout.seed, global step,
    prompt position), reproducible on resume.  ``fixed``: every request of every step uses worker.rollout.seed
    (sample j: seed + j), so vLLM replays the same n noise streams at every step."""
    rollout_correction: str = "none"
    """Correction for the difference between the vLLM sampler and the FSDP policy that scores the samples.
    ``none``; ``tis``: every surrogate term is multiplied by min(pi_old / pi_vllm, rollout_correction_cap) per
    token (truncated importance sampling, detached).  Needs worker.rollout.calculate_log_probs=true."""
    rollout_correction_cap: float = 2.0

    # ------------------------------------------------ KL to the initialization model
    ref_kl_coef: float = 0.04
    """beta of the penalty beta * KL(pi_theta || pi_ref) to the initialization model (the SFT checkpoint of
    worker.actor.model.model_path, kept frozen), per response token with the token mean of the optimizer
    mini-batch, added to every update (task and distillation phases).  The reported value is the k3 estimate,
    the gradient that of k2 (verl's "k3+"; see trsi/losses.py reference_kl_k3), which near the reference equals
    the gradient of k3 as the loss.  Open-o3 Video's own RL recipe uses beta = 0.04 with k3 as the loss.
    0 (the method as specified, and the behaviour of checkpoints written before package v4) adds nothing and
    builds no reference model."""
    ref_kl_monitor: bool = False
    """Compute the reference log-probabilities every step for the health metric health/ref_kl (the k3 KL of the
    sampling policy to the initialization model) even when ref_kl_coef is 0 (one extra forward pass per step)."""

    # ------------------------------------------------- run health and teacher gate
    health_min_format_rate: Optional[float] = 0.5
    """A step is unhealthy when the mean format reward of its rollouts is below this value (null: not checked)."""
    health_min_response_length: Optional[float] = 32.0
    """A step is unhealthy when the mean response length (tokens) is below this value (null: not checked)."""
    health_max_truncated_fraction: Optional[float] = 0.25
    """A step is unhealthy when more than this fraction of its responses are truly length-truncated according
    to generation termination metadata (zero_truncated_rewards=True). When zero_truncated_rewards=False,
    retain the historical proxy: response length >= data.max_response_length. Null: not checked."""
    health_ref_kl_jump_ratio: Optional[float] = 2.5
    """A step is unhealthy when health/ref_kl exceeds both ``health_ref_kl_jump_floor`` and this multiple of the
    median of the previous ``health_ref_kl_window`` steps (null: not checked; needs the reference model, i.e.
    ref_kl_coef > 0 or ref_kl_monitor).  It catches a fast drift from the SFT model while a slow one passes."""
    health_ref_kl_jump_floor: float = 0.3
    health_ref_kl_window: int = 20
    guard_patience: int = 3
    """Stop the run (no checkpoint at that step) once ``guard_patience`` of the last ``guard_window`` steps were
    unhealthy; 0 disables the stop (unhealthy steps are still recorded).  The stop writes
    <run_dir>/COLLAPSE_DETECTED.json."""
    guard_window: int = 4
    """Steps over which ``guard_patience`` unhealthy steps stop the run (``guard_window == guard_patience``:
    consecutive steps only, the behaviour of package v4)."""
    teacher_health_gate: bool = True
    """At the updated snapshot, compare the first and last ``teacher_health_window`` task steps of the phase.  If
    the format rate fell by more than ``teacher_health_max_format_drop`` or the mean response length by more than
    the fraction ``teacher_health_max_length_drop``, the phase is degraded: the new teacher version uses the
    anchor (the student at the start of the phase) as its base and no correction (gate reason
    ``task_phase_degraded``), instead of extrapolating along the degradation."""
    teacher_health_window: int = 5
    teacher_health_max_format_drop: float = 0.1
    teacher_health_max_length_drop: float = 0.5
    teacher_health_max_length_rise: Optional[float] = 0.5
    """The phase is also degraded when the mean response length rose by more than this fraction (null: not
    checked; package v4 checked drops only)."""
    teacher_health_max_truncation_rise: Optional[float] = 0.1
    """... or when the truncated fraction rose by more than this (absolute; null: not checked)."""

    # --------------------------------------------------------- rollout dumps
    dump_rollouts_interval: int = 10
    """Write decoded rollouts with their rewards and advantages to <run_dir>/logs/rollouts/ every this many steps
    (0: never).  Unhealthy steps are always written."""
    dump_rollouts_rows: int = 8
    """Rows written per regular dump (all rows on an unhealthy step)."""

    # ---------------------------------------------------------------- resume
    resume_allow_changes: list[str] = field(default_factory=list)
    """Protocol settings (rsi.* keys) allowed to differ from the checkpoint being resumed, e.g.
    [task_loss_agg, teacher_health_gate] to continue a run trained before these options existed.  Each change is
    printed and recorded in the run_start record; any other difference still stops the resume."""

    # ---------------------------------------------------------- verification
    verify_fields: bool = True
    """Record supported / contradicted / unknown chunk status (diagnostics only)."""
    verify_time_tolerance: float = 1.0
    verify_supported_iou: float = 0.5

    # ---------------------------------------------------------------- debug
    log_chunk_examples: int = 0
    """Print this many chunked responses every ``log_chunk_interval`` distillation steps."""
    log_chunk_interval: int = 50
    record_state_fingerprints: bool = True
    """Record parameter/optimizer fingerprints before and after every step in
    ``<run_dir>/logs/trsi_steps.jsonl`` (verifies snapshots, updates and resume)."""

    def route_c_settings(self) -> RouteCSettings:
        """The frozen estimation rule of the Route C inner loop (validated)."""

        return RouteCSettings(
            candidates=int(self.route_c_candidates),
            continuations=int(self.route_c_continuations),
            min_gap=float(self.route_c_min_gap),
            min_coverage=float(self.route_c_min_coverage),
            value_clip=float(self.route_c_value_clip),
            kappa=float(self.route_c_kappa),
        )

    def post_init(self):
        if self.route not in SUPPORTED_ROUTES:
            raise ValueError(f"rsi.route must be one of {SUPPORTED_ROUTES}, got {self.route!r}.")
        if self.schedule not in SUPPORTED_SCHEDULES:
            raise ValueError(f"rsi.schedule must be one of {SUPPORTED_SCHEDULES}, got {self.schedule!r}.")
        for name in ("task_steps_per_round", "distill_steps_per_round", "joint_stage_steps"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"rsi.{name} must be >= 1.")
        for name in ("sd_coef", "extrapolation_alpha", "extrapolation_alpha_min"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"rsi.{name} must be finite and non-negative.")
        if not (0.0 <= float(self.extrapolation_alpha_decay) <= 1.0):
            raise ValueError("rsi.extrapolation_alpha_decay must be in [0, 1].")
        if float(self.extrapolation_alpha_min) > float(self.extrapolation_alpha):
            raise ValueError("rsi.extrapolation_alpha_min must not exceed rsi.extrapolation_alpha.")
        if self.correction_clip is not None and not (
            math.isfinite(float(self.correction_clip)) and float(self.correction_clip) > 0.0
        ):
            raise ValueError("rsi.correction_clip must be positive (or null to disable the bound).")
        if not (0.0 <= float(self.base_mix_beta) <= 1.0):
            raise ValueError("rsi.base_mix_beta must be in [0, 1].")
        if not (math.isfinite(float(self.teacher_temperature)) and float(self.teacher_temperature) > 0.0):
            raise ValueError("rsi.teacher_temperature must be positive.")
        if float(self.teacher_temperature) != 1.0:
            print(
                "[Video-t-RSI] WARNING: rsi.teacher_temperature != 1. Temperature sharpens or flattens existing "
                "preferences and is not evidence of teacher improvement (method section 9)."
            )
        if int(self.min_task_signal_steps) < 0:
            raise ValueError("rsi.min_task_signal_steps must be non-negative.")
        views = [str(view) for view in self.views]
        if not views or len(set(views)) != len(views) or any(view not in SUPPORTED_VIEWS for view in views):
            raise ValueError(f"rsi.views must be a non-empty unique subset of {SUPPORTED_VIEWS}, got {views}.")
        # Keep the fixed A/AT/AS order expected by the prompt builder.
        self.views = [view for view in SUPPORTED_VIEWS if view in views]
        for name in (
            "teacher_max_prompt_length",
            "teacher_vocab_chunk",
            "teacher_position_chunk",
            "chunk_min_tokens",
            "chunk_max_tokens",
            "chunk_fallback_tokens",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"rsi.{name} must be positive.")
        if int(self.chunk_max_tokens) < int(self.chunk_min_tokens):
            raise ValueError("rsi.chunk_max_tokens must be >= rsi.chunk_min_tokens.")
        if int(self.credit_horizon) < 0:
            raise ValueError("rsi.credit_horizon must be non-negative.")
        if not (0.0 < float(self.credit_discount) <= 1.0):
            raise ValueError("rsi.credit_discount must be in (0, 1].")
        if not (math.isfinite(float(self.credit_clip)) and float(self.credit_clip) > 0.0):
            raise ValueError("rsi.credit_clip must be positive.")
        for name in ("task_clip_low", "sd_clip_low"):
            if not (0.0 <= float(getattr(self, name)) < 1.0):
                raise ValueError(f"rsi.{name} must be in [0, 1).")
        for name in ("task_clip_high", "sd_clip_high"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"rsi.{name} must be non-negative.")
        if float(self.verify_time_tolerance) < 0.0 or not (0.0 < float(self.verify_supported_iou) <= 1.0):
            raise ValueError("rsi.verify_time_tolerance must be >= 0 and rsi.verify_supported_iou in (0, 1].")
        if int(self.log_chunk_examples) < 0 or int(self.log_chunk_interval) < 1:
            raise ValueError("rsi.log_chunk_examples must be >= 0 and rsi.log_chunk_interval >= 1.")
        if self.task_loss_agg not in SUPPORTED_TASK_LOSS_AGG:
            raise ValueError(f"rsi.task_loss_agg must be one of {SUPPORTED_TASK_LOSS_AGG}, got {self.task_loss_agg!r}.")
        if self.sd_mode not in SUPPORTED_SD_MODES:
            raise ValueError(f"rsi.sd_mode must be one of {SUPPORTED_SD_MODES}, got {self.sd_mode!r}.")
        if not (math.isfinite(float(self.reweight_clip)) and 0.0 <= float(self.reweight_clip) < 1.0):
            raise ValueError("rsi.reweight_clip must be in [0, 1).")
        if self.reweight_center not in REWEIGHT_CENTER_MODES:
            raise ValueError(f"rsi.reweight_center must be one of {REWEIGHT_CENTER_MODES}, got {self.reweight_center!r}.")
        if self.inner_loop not in INNER_LOOP_MODES:
            raise ValueError(f"rsi.inner_loop must be one of {INNER_LOOP_MODES}, got {self.inner_loop!r}.")
        try:
            self.route_c_settings()
        except ValueError as error:
            raise ValueError(f"Invalid rsi.route_c_* settings: {error}") from error
        if not (math.isfinite(float(self.route_c_coef)) and float(self.route_c_coef) >= 0.0):
            raise ValueError("rsi.route_c_coef must be finite and non-negative.")
        if self.rollout_seed_mode not in SUPPORTED_SEED_MODES:
            raise ValueError(f"rsi.rollout_seed_mode must be one of {SUPPORTED_SEED_MODES}, got {self.rollout_seed_mode!r}.")
        if self.rollout_correction not in SUPPORTED_ROLLOUT_CORRECTIONS:
            raise ValueError(
                f"rsi.rollout_correction must be one of {SUPPORTED_ROLLOUT_CORRECTIONS}, got {self.rollout_correction!r}."
            )
        if not (math.isfinite(float(self.rollout_correction_cap)) and float(self.rollout_correction_cap) >= 1.0):
            raise ValueError("rsi.rollout_correction_cap must be finite and >= 1.")
        if not (math.isfinite(float(self.ref_kl_coef)) and float(self.ref_kl_coef) >= 0.0):
            raise ValueError("rsi.ref_kl_coef must be finite and non-negative.")
        if not isinstance(self.zero_truncated_rewards, bool):
            raise ValueError("rsi.zero_truncated_rewards must be a boolean.")
        if not (math.isfinite(float(self.task_adv_std_floor)) and float(self.task_adv_std_floor) >= 0.0):
            raise ValueError("rsi.task_adv_std_floor must be finite and non-negative.")
        if self.health_min_format_rate is not None and not (0.0 <= float(self.health_min_format_rate) <= 1.0):
            raise ValueError("rsi.health_min_format_rate must be in [0, 1] (or null).")
        if self.health_min_response_length is not None and float(self.health_min_response_length) < 0.0:
            raise ValueError("rsi.health_min_response_length must be non-negative (or null).")
        if self.health_max_truncated_fraction is not None and not (0.0 <= float(self.health_max_truncated_fraction) <= 1.0):
            raise ValueError("rsi.health_max_truncated_fraction must be in [0, 1] (or null).")
        if self.health_ref_kl_jump_ratio is not None and not (
            math.isfinite(float(self.health_ref_kl_jump_ratio)) and float(self.health_ref_kl_jump_ratio) > 1.0
        ):
            raise ValueError("rsi.health_ref_kl_jump_ratio must be > 1 (or null).")
        if not (math.isfinite(float(self.health_ref_kl_jump_floor)) and float(self.health_ref_kl_jump_floor) >= 0.0):
            raise ValueError("rsi.health_ref_kl_jump_floor must be finite and non-negative.")
        if int(self.health_ref_kl_window) < REF_KL_MIN_HISTORY:
            raise ValueError(
                f"rsi.health_ref_kl_window must be >= {REF_KL_MIN_HISTORY}: the KL-jump check needs that many earlier "
                "values (set rsi.health_ref_kl_jump_ratio=null to switch the check off)."
            )
        if int(self.guard_patience) < 0:
            raise ValueError("rsi.guard_patience must be non-negative (0 disables the stop).")
        if int(self.guard_window) < max(int(self.guard_patience), 1):
            raise ValueError("rsi.guard_window must be >= rsi.guard_patience (and >= 1).")
        if int(self.teacher_health_window) < 1:
            raise ValueError("rsi.teacher_health_window must be >= 1.")
        if not (0.0 <= float(self.teacher_health_max_format_drop) <= 1.0):
            raise ValueError("rsi.teacher_health_max_format_drop must be in [0, 1].")
        if not (0.0 <= float(self.teacher_health_max_length_drop) <= 1.0):
            raise ValueError("rsi.teacher_health_max_length_drop must be in [0, 1].")
        if self.teacher_health_max_length_rise is not None and not (
            math.isfinite(float(self.teacher_health_max_length_rise)) and float(self.teacher_health_max_length_rise) >= 0.0
        ):
            raise ValueError("rsi.teacher_health_max_length_rise must be finite and non-negative (or null).")
        if self.teacher_health_max_truncation_rise is not None and not (
            0.0 <= float(self.teacher_health_max_truncation_rise) <= 1.0
        ):
            raise ValueError("rsi.teacher_health_max_truncation_rise must be in [0, 1] (or null).")
        if int(self.dump_rollouts_interval) < 0 or int(self.dump_rollouts_rows) < 0:
            raise ValueError("rsi.dump_rollouts_interval and rsi.dump_rollouts_rows must be non-negative.")
        self.resume_allow_changes = [str(key) for key in self.resume_allow_changes]


@dataclass
class VideoTRSITrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    rsi: TRSIConfig = field(default_factory=TRSIConfig)

    def post_init(self):
        if self.data.dataset_format != "stgr":
            raise ValueError("Video-t-RSI trains on the STGR schema; set data.dataset_format=stgr.")
        if self.algorithm.adv_estimator != "grpo":
            raise ValueError("Video-t-RSI keeps the original GRPO task loss; set algorithm.adv_estimator=grpo.")
        if not self.algorithm.disable_kl or self.algorithm.use_kl_loss or self.algorithm.kl_coef != 0:
            raise ValueError("Video-t-RSI does not use a reference-policy KL; keep algorithm.disable_kl=true.")
        if self.algorithm.online_filtering or self.algorithm.pipeline_reward:
            raise ValueError("Video-t-RSI requires synchronous, unfiltered rewards.")
        if self.worker.actor.ppo_epochs != 1:
            raise ValueError("Video-t-RSI uses one policy epoch per rollout batch (worker.actor.ppo_epochs=1).")
        if self.worker.actor.dynamic_batching:
            raise ValueError(
                "Video-t-RSI requires worker.actor.dynamic_batching=false so every FSDP rank runs identical "
                "teacher-forward schedules."
            )
        if self.worker.actor.ulysses_size != 1:
            raise ValueError("Video-t-RSI teacher scoring currently requires worker.actor.ulysses_size=1.")
        if self.worker.ref.offload.offload_params:
            raise ValueError("Video-t-RSI snapshots require worker.ref.offload.offload_params=false.")
        if self.worker.rollout.n <= 1:
            raise ValueError("GRPO requires worker.rollout.n > 1.")
        if self.rsi.rollout_correction != "none" and not self.worker.rollout.calculate_log_probs:
            raise ValueError(
                f"rsi.rollout_correction={self.rsi.rollout_correction} needs the sampler's log-probabilities; set "
                "worker.rollout.calculate_log_probs=true."
            )
        if self.rsi.inner_loop == "route_c":
            if float(self.worker.rollout.temperature) != float(self.rsi.teacher_temperature):
                raise ValueError(
                    "rsi.inner_loop=route_c compares the student's candidate distribution (at the rollout temperature) "
                    "with the teacher's (at rsi.teacher_temperature); a difference would be measured as a gap "
                    f"everywhere. Got worker.rollout.temperature={self.worker.rollout.temperature} and "
                    f"rsi.teacher_temperature={self.rsi.teacher_temperature} (or set rsi.inner_loop=none)."
                )
            if int(self.data.max_response_length) < 2:
                raise ValueError("rsi.inner_loop=route_c needs data.max_response_length >= 2 (room to continue).")
        global_batch, rollout_batch = int(self.worker.actor.global_batch_size), int(self.data.rollout_batch_size)
        if global_batch <= 0 or global_batch > rollout_batch or rollout_batch % global_batch != 0:
            raise ValueError(
                "worker.actor.global_batch_size (prompts per optimizer step) must divide data.rollout_batch_size "
                f"(prompts per rollout step); got {global_batch} and {rollout_batch}. Equal values give one "
                "optimizer step per rollout step (the base configs); a rollout batch of k update batches gives k "
                "optimizer steps per rollout step, with the PPO-style ratio to the generating policy (the throughput config)."
            )
        if self.trainer.save_model_only:
            raise ValueError(
                "Video-t-RSI checkpoints must stay resumable (student, optimizer, scheduler, teachers and RSI "
                "state); keep trainer.save_model_only=false."
            )
        # The MeSD EMA-teacher checkpoint path is not used by the Video-t-RSI worker.
        self.worker.actor.mesd_enabled = False
        self.worker.actor.freeze_teacher_model = True
        self.worker.actor.trsi_reference = float(self.rsi.ref_kl_coef) > 0.0 or bool(self.rsi.ref_kl_monitor)
        if self.rsi.health_ref_kl_jump_ratio is not None and not self.worker.actor.trsi_reference:
            print(
                "[Video-t-RSI] WARNING: rsi.health_ref_kl_jump_ratio is set, but no reference model is built "
                "(rsi.ref_kl_coef=0, rsi.ref_kl_monitor=false): the collapse guard's KL-jump check is inactive. Set "
                "rsi.ref_kl_monitor=true to keep it, or rsi.health_ref_kl_jump_ratio=null to silence this warning."
            )
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)


def load_trsi_config(argv: Optional[list[str]] = None) -> VideoTRSITrainingConfig:
    """Merge dataclass defaults, the YAML given as ``config=...`` and CLI overrides.

    ``argv`` defaults to ``sys.argv[1:]``.  Unknown keys in the YAML or the
    overrides are rejected by OmegaConf's structured-config merge.
    """

    from omegaconf import OmegaConf

    cli_args = OmegaConf.from_cli() if argv is None else OmegaConf.from_dotlist(list(argv))
    default_config = OmegaConf.structured(VideoTRSITrainingConfig())
    if "config" in cli_args:
        config_path = cli_args.pop("config")
        default_config = OmegaConf.merge(default_config, OmegaConf.load(config_path))
    merged = OmegaConf.merge(default_config, cli_args)
    training_config = OmegaConf.to_object(merged)
    training_config.deep_post_init()
    return training_config
