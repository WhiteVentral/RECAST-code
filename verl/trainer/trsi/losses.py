"""Student objectives of Video-t-RSI (method section 12).

PPO-style surrogate shared by both terms::

    rho_{i,t}(theta) = pi_theta(y_{i,t} | h_{i,t}) / pi_old(y_{i,t} | h_{i,t})
    Psi(rho, A)      = min(rho * A, clip(rho, 1 - eps_low, 1 + eps_high) * A)

Task loss (original reward, GRPO group-relative advantages), with one of two
aggregations over the valid rows ``i`` of an optimizer mini-batch
(``rsi.task_loss_agg``)::

    token_mean:           L_task = -(1 / sum_i T_i) sum_i sum_t Psi(rho_{i,t}, A^task_i)
    seq_mean_token_mean:  L_task = -(1 / B_task) sum_i (1 / T_i) sum_t Psi(rho_{i,t}, A^task_i)

``seq_mean_token_mean`` is MeSD's aggregation.  Its 1 / T_i weights let short
responses dominate the update and give a token shared by all responses of a
group the net coefficient sum_i A_i / T_i, which is non-zero although the
mean-centred advantages sum to zero; ``token_mean`` gives that token
sum_i A_i / sum_j T_j = 0 (DAPO, Dr. GRPO).

Chunk-balanced self-distillation loss with equal chunk weights::

    L_SD = -(1 / B_SD) sum_i (1 / K_i) sum_k (1 / |C_{i,k}|) sum_{t in C_{i,k}} Psi(rho_{i,t}, A^SD_{i,k})
         = -(1 / B_SD) sum_i sum_t w_{i,t} Psi(rho_{i,t}, A^SD_{i,t})

Joint objective (schedule 1) or phase objective (schedule 2)::

    L = c_task * L_task + c_SD * L_SD + beta * L_KL

with the optional penalty to the frozen initialization model (``rsi.ref_kl_coef``
= beta; token mean over the valid rows; value: the k3 estimate, gradient: that of k2,
see ``reference_kl_k3``)::

    L_KL = (1 / sum_i T_i) sum_i sum_t [exp(l_{i,t}) - l_{i,t} - 1],  l = log pi_ref - log pi_theta
    grad L_KL = (1 / sum_i T_i) sum_i sum_t (log pi_theta - log pi_ref)_{i,t} grad log pi_theta_{i,t}

The teacher, old-policy probabilities, segmentation and credit are detached;
gradients reach the student only through ``rho``.  With ``rsi.sd_mode=surrogate``
no sign of the task advantage is applied to the distillation credit and no
objective is switched according to response success or failure (method
section 3).  With ``rsi.sd_mode=reweight`` (package v5) distillation steps
optimize ``L_task`` alone (``c_SD = 0``) with advantages the driver multiplied
by the reward-anchored chunk factors ``clip(exp(sign(A) (G - Gbar)), 1 - c_w, 1 + c_w)``,
Gbar the response's mean credit (``verl.trainer.trsi.credit.reward_anchored_factors``): the task advantage's
sign then decides the direction of the teacher's influence, as in RLSD and
MeSD.  This module sees only the reweighted ``A^task``.

With ``rsi.rollout_correction=tis`` every ``Psi`` is multiplied by the detached
truncated importance weight ``min(pi_old / pi_vllm, C)`` of its token, which
corrects for the difference between the vLLM sampler and the FSDP policy.

Under data-parallel training with gradient averaging over ``world_size``
ranks and gradient accumulation over micro-batches, returning
``sum_rows(term) * world_size / global_count`` for every micro-batch yields the
exact global mean above (the convention already used by MeSD); the global
count is the number of valid rows, or of valid tokens for ``token_mean``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


__all__ = [
    "RSILossOutput",
    "TASK_LOSS_AGGREGATIONS",
    "clipped_policy_surrogate",
    "compute_rsi_policy_loss",
    "reference_kl_k3",
]

TASK_LOSS_AGGREGATIONS = ("token_mean", "seq_mean_token_mean")


REF_KL_LOG_RATIO_CLAMP = 20.0


def reference_kl_k3(log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """Per-token penalty for KL(pi_theta || pi_ref) at the sampled tokens: k3 value, k2 gradient (verl's ``k3+``).

    Value: Schulman's k3 estimate ``exp(l) - l - 1``, ``l = log pi_ref - log pi_theta`` clamped to [-20, 20] and
    the estimate to [0, 10], as verl's ``low_var_kl``; it is the reported ``ref_kl``.

    Gradient: that of k2 = (log pi_theta - log pi_ref)^2 / 2, i.e. ``(log pi_theta - log pi_ref) * d log pi_theta``
    with the coefficient clamped to [-20, 20].  On the policy's own samples this is an unbiased estimate of
    grad KL(pi_theta || pi_ref), and near the reference it equals the gradient of k3 (1 - e^l ~ -l), so beta
    keeps its meaning.  k3 itself used as the loss has the expected gradient of KL(pi_ref || pi_theta),
    exponential in l for tokens the policy under-weights, and its clamped value has no gradient at all once a
    token has drifted far (pi_ref / pi_theta > 13.6 or < 1.7e-5) -- the collapse regime, where the pull back
    matters most.  The k2 gradient is linear in l and does not vanish.
    """

    log_probs = log_probs.float()
    log_ratio = (ref_log_probs.detach().float() - log_probs.detach()).clamp(-REF_KL_LOG_RATIO_CLAMP, REF_KL_LOG_RATIO_CLAMP)
    value = (torch.exp(log_ratio) - log_ratio - 1.0).clamp(min=0.0, max=10.0)
    surrogate = -log_ratio * log_probs  # d/d log pi_theta = log pi_theta - log pi_ref (clamped)
    return value + (surrogate - surrogate.detach())


def clipped_policy_surrogate(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_low: float,
    clip_high: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(Psi, rho)``; ``Psi`` keeps the gradient of ``log_probs`` only."""

    if not (0.0 <= float(clip_low) < 1.0) or float(clip_high) < 0.0:
        raise ValueError(f"invalid PPO clip range low={clip_low}, high={clip_high}.")
    ratio = torch.exp(log_probs - old_log_probs.detach())
    advantages = advantages.detach()
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_low), 1.0 + float(clip_high))
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    return surrogate, ratio


@dataclass
class RSILossOutput:
    loss: torch.Tensor
    """Scaled micro-batch loss to call ``backward`` on."""
    task_row_loss: torch.Tensor
    """Per-row task losses (zero on invalid rows), detached: ``-sum_t Psi`` for ``token_mean``,
    ``-(1/T_i) sum_t Psi`` for ``seq_mean_token_mean``."""
    sd_row_loss: torch.Tensor
    """Per-row distillation losses ``-sum_t w_t Psi`` (zero on invalid rows), detached."""
    metrics: dict
    task_loss_count: float = 0.0
    """Local normalizer of ``task_row_loss`` (valid tokens or valid rows of this micro-batch): their ratio is
    this micro-batch's task loss in the units of the optimized objective."""


def compute_rsi_policy_loss(
    *,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    task_advantages: Optional[torch.Tensor],
    task_row_mask: Optional[torch.Tensor],
    sd_advantages: Optional[torch.Tensor],
    sd_token_weights: Optional[torch.Tensor],
    sd_row_mask: Optional[torch.Tensor],
    task_coef: float,
    sd_coef: float,
    task_global_rows: float,
    sd_global_rows: float,
    world_size: int,
    task_clip: tuple[float, float] = (0.2, 0.2),
    sd_clip: tuple[float, float] = (0.2, 0.2),
    task_loss_agg: str = "seq_mean_token_mean",
    task_global_tokens: Optional[float] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
    ref_kl_coef: float = 0.0,
    ref_log_probs: Optional[torch.Tensor] = None,
    kl_row_mask: Optional[torch.Tensor] = None,
    kl_global_tokens: Optional[float] = None,
) -> RSILossOutput:
    """Combine the task and chunk-balanced distillation surrogates for one micro-batch.

    ``task_global_rows`` / ``sd_global_rows`` are the numbers of valid rows in
    the whole optimizer mini-batch across all ranks (after an all-reduce);
    ``task_global_tokens`` is the number of response tokens of the valid task
    rows (all ranks), required by ``task_loss_agg="token_mean"``.
    ``rollout_is_weights`` are per-token truncated importance weights
    ``min(pi_old / pi_vllm, C)`` (``None``: no rollout correction).
    ``ref_kl_coef > 0`` adds ``ref_kl_coef * L_KL`` over the rows of
    ``kl_row_mask``; ``kl_global_tokens`` is their response-token count on all ranks.
    """

    for name, coefficient in (("task_coef", task_coef), ("sd_coef", sd_coef), ("ref_kl_coef", ref_kl_coef)):
        if not (math.isfinite(float(coefficient)) and float(coefficient) >= 0.0):
            raise ValueError(f"{name} must be finite and non-negative, got {coefficient}.")
    if int(world_size) <= 0:
        raise ValueError("world_size must be positive.")
    if task_loss_agg not in TASK_LOSS_AGGREGATIONS:
        raise ValueError(f"task_loss_agg must be one of {TASK_LOSS_AGGREGATIONS}, got {task_loss_agg!r}.")
    mask = response_mask.to(dtype=torch.float32)
    batch = mask.size(0)
    zero = log_probs.sum() * 0.0
    task_term = zero
    sd_term = zero
    kl_term = zero
    task_rows = torch.zeros(batch, dtype=torch.float32, device=log_probs.device)
    sd_rows = torch.zeros(batch, dtype=torch.float32, device=log_probs.device)
    task_count = 0.0
    metrics: dict[str, float] = {}
    is_weights = None
    if rollout_is_weights is not None:
        if tuple(rollout_is_weights.shape) != tuple(mask.shape):
            raise ValueError(f"rollout_is_weights must have shape {tuple(mask.shape)}, got {tuple(rollout_is_weights.shape)}.")
        is_weights = rollout_is_weights.detach().to(dtype=torch.float32, device=mask.device)

    ratio_for_metrics = torch.exp((log_probs - old_log_probs).detach())
    metrics["ppo_kl"] = float(((old_log_probs - log_probs).detach() * mask).sum().div(mask.sum().clamp_min(1.0)).item())

    if float(task_coef) > 0.0:
        if task_advantages is None or task_row_mask is None:
            raise ValueError("task_coef > 0 requires task advantages and a task row mask.")
        valid = task_row_mask.to(dtype=torch.float32, device=mask.device)
        psi, ratio = clipped_policy_surrogate(
            log_probs, old_log_probs, task_advantages, clip_low=task_clip[0], clip_high=task_clip[1]
        )
        if is_weights is not None:
            psi = psi * is_weights
        token_mask = mask * valid.unsqueeze(-1)
        if task_loss_agg == "token_mean":
            if task_global_tokens is None:
                raise ValueError("task_loss_agg='token_mean' requires task_global_tokens (all-reduced token count).")
            row_loss = -(psi * mask).sum(dim=-1) * valid
            task_term = row_loss.sum() * float(world_size) / max(float(task_global_tokens), 1.0)
            task_count = float(token_mask.sum().item())
        else:
            sequence_mean = (psi * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1.0e-8)
            row_loss = -sequence_mean * valid
            task_term = row_loss.sum() * float(world_size) / max(float(task_global_rows), 1.0)
            task_count = float(valid.sum().item())
        task_rows = row_loss.detach()
        token_denom = token_mask.sum().clamp_min(1.0)
        upper = ((ratio > 1.0 + task_clip[1]) & (task_advantages > 0)).float()
        lower = ((ratio < 1.0 - task_clip[0]) & (task_advantages < 0)).float()
        metrics["task_clipfrac_higher"] = float((upper * token_mask).sum().div(token_denom).item())
        metrics["task_clipfrac_lower"] = float((lower * token_mask).sum().div(token_denom).item())

    if float(sd_coef) > 0.0:
        if sd_advantages is None or sd_token_weights is None or sd_row_mask is None:
            raise ValueError("sd_coef > 0 requires distillation advantages, token weights and a row mask.")
        valid = sd_row_mask.to(dtype=torch.float32, device=mask.device)
        weights = sd_token_weights.detach().float() * mask * valid.unsqueeze(-1)
        psi, ratio = clipped_policy_surrogate(
            log_probs, old_log_probs, sd_advantages, clip_low=sd_clip[0], clip_high=sd_clip[1]
        )
        if is_weights is not None:
            psi = psi * is_weights
        row_loss = -(psi * weights).sum(dim=-1)
        sd_rows = row_loss.detach()
        sd_term = row_loss.sum() * float(world_size) / max(float(sd_global_rows), 1.0)
        active = weights.gt(0).float()
        active_denom = active.sum().clamp_min(1.0)
        upper = ((ratio > 1.0 + sd_clip[1]) & (sd_advantages > 0)).float()
        lower = ((ratio < 1.0 - sd_clip[0]) & (sd_advantages < 0)).float()
        metrics["sd_clipfrac_higher"] = float((upper * active).sum().div(active_denom).item())
        metrics["sd_clipfrac_lower"] = float((lower * active).sum().div(active_denom).item())
        metrics["sd_ratio_mean"] = float((ratio_for_metrics * active).sum().div(active_denom).item())

    if float(ref_kl_coef) > 0.0:
        if ref_log_probs is None or kl_row_mask is None or kl_global_tokens is None:
            raise ValueError("ref_kl_coef > 0 requires reference log-probs, a row mask and the global token count.")
        valid = kl_row_mask.to(dtype=torch.float32, device=mask.device)
        token_mask = mask * valid.unsqueeze(-1)
        kl = reference_kl_k3(log_probs, ref_log_probs)
        kl_term = (kl * token_mask).sum() * float(world_size) / max(float(kl_global_tokens), 1.0)
        metrics["ref_kl"] = float((kl.detach() * token_mask).sum().div(token_mask.sum().clamp_min(1.0)).item())

    loss = float(task_coef) * task_term + float(sd_coef) * sd_term + float(ref_kl_coef) * kl_term
    # Not raised: under FSDP one rank raising would leave the others waiting in the backward
    # collectives.  A non-finite loss yields a non-finite global gradient norm, so the
    # optimizer step is skipped on every rank and ``trsi/update_applied`` records it.
    metrics["loss_nonfinite"] = float(not bool(torch.isfinite(loss.detach()).item()))
    return RSILossOutput(
        loss=loss, task_row_loss=task_rows, sd_row_loss=sd_rows, metrics=metrics, task_loss_count=task_count
    )
