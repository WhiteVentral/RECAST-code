"""Chunk aggregation and finite-horizon future credit (method sections 11 and 12).

Local feedback on the current student trajectory::

    delta_{i,t} = sg[ log q_{r,i,t}(y_{i,t}) - log pi_old(y_{i,t} | h_{i,t}) ]

Chunk mean and normalized finite-horizon chunk return-to-go::

    D_{i,k} = (1 / |C_{i,k}|) * sum_{t in C_{i,k}} delta_{i,t}
    G_{i,k} = sum_{j=k}^{min(K_i - 1, k + H)} gamma^{j-k} D_{i,j}
              / sum_{j=k}^{min(K_i - 1, k + H)} gamma^{j-k}

Chunks are 0-indexed here.  The current chunk is counted once, the final
chunk has ``G == D`` and ``H == 0`` keeps only the current chunk.  Token
values that already contain future information are never accumulated again.

Distillation credit and equal-chunk token weights (``rsi.sd_mode=surrogate``,
the advantage and weights of L_SD)::

    A^SD_{i,k} = sg[ clip(G_{i,k}, -c, c) ]
    w_{i,t}    = 1 / (K_i * |C_{i,k(t)}|)            (sum_t w_{i,t} = 1)

Reward-anchored reweighting (``rsi.sd_mode=reweight``, package v5; the RLSD
form, with the chunk credit in place of a per-token gap.  MeSD used it as
``1 + lambda (m - 1)`` with lambda annealed from 0.5 to 0 over 1,000 steps;
here lambda = 1 and the range is set by c_w)::

    Gbar_i     = sum_{t in chunks of i} G_{i,k(t)} / #(chunk tokens of i)      (rsi.reweight_center=row)
    m_{i,t}    = clip( exp(sign(A_{i,t}) * (G_{i,k(t)} - Gbar_i)), 1 - c_w, 1 + c_w )
    A'_{i,t}   = m_{i,t} * A_{i,t}

``A`` is the GRPO task advantage.  The task reward sets the sign of every
update (m > 0); the teacher moves its magnitude between the chunks of a
response: in a response that scored above its group, chunks the teacher
prefers more than the rest of the response are pushed up more; in one that
scored below, chunks it rejects more are pushed down more.  Tokens without
task signal (A = 0) or outside the chunks of a distillable row keep m = 1.

Why the credit is centred per response.  G carries an offset that the whole
response shares and that the teacher, not the reward, sets: delta is measured
on the student's own samples, and drawn from pi_old itself its expectation
would be -KL(pi_old || q) <= 0 (top-p/top-k sampling breaks that identity, so
the offset can have either sign).  Uncentred (``reweight_center=none``), a
negative offset lowers the factors of every response with A > 0 and raises
those with A < 0, so sum_i m_i A_i < 0 within a GRPO group: every reweighted
step would push the tokens a group shares (the format prefix) down, the
mechanism of the first run's collapse; a positive offset pushes them up.
Centred, the log-factors of a response average exactly 0 before clipping and
the factors about 1, the reward alone sets how much each response is pushed,
and the teacher only redistributes it between the chunks.  Differences between
positions remain: a teacher that credits the opening chunk systematically
above or below the rest still pushes the shared tokens, by at most
c_w * sum_i |A_i| per group.

All functions are stop-gradient and operate on padded ``[B, R]`` tensors with
``chunk_ids == -1`` on padding.  Aggregation never crosses rows or padding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch


REWEIGHT_CENTER_MODES = ("row", "none")


__all__ = [
    "REWEIGHT_CENTER_MODES",
    "RewardAnchoredFactors",
    "broadcast_chunk_values",
    "chunk_ids_from_spans",
    "chunk_means",
    "clip_chunk_credit",
    "equal_chunk_token_weights",
    "finite_horizon_chunk_credit",
    "local_token_feedback",
    "reward_anchored_factors",
    "reweight_metrics",
]


def chunk_ids_from_spans(
    spans_per_row: Sequence[Sequence[tuple[int, int]]],
    response_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn per-row contiguous token spans into a ``[B, R]`` chunk-id map.

    Spans must be sorted, non-overlapping, contiguous and start at token 0 of
    the response.  Positions outside every span are padding (``-1``).
    """

    batch = len(spans_per_row)
    chunk_ids = torch.full((batch, int(response_length)), -1, dtype=torch.long)
    num_chunks = torch.zeros(batch, dtype=torch.long)
    for row, spans in enumerate(spans_per_row):
        cursor = 0
        for index, (start, end) in enumerate(spans):
            start, end = int(start), int(end)
            if start != cursor or end <= start or end > response_length:
                raise ValueError(
                    f"Row {row} chunk {index} span {(start, end)} is not a contiguous partition "
                    f"starting at {cursor} within response length {response_length}."
                )
            chunk_ids[row, start:end] = index
            cursor = end
        num_chunks[row] = len(spans)
    return chunk_ids, num_chunks


def local_token_feedback(
    teacher_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    token_mask: torch.Tensor,
) -> torch.Tensor:
    """``delta = sg[log q(y) - log pi_old(y)]`` on valid tokens, zero elsewhere."""

    if teacher_log_probs.shape != old_log_probs.shape or teacher_log_probs.shape != token_mask.shape:
        raise ValueError("teacher_log_probs, old_log_probs and token_mask must share one [B, R] shape.")
    mask = token_mask.to(torch.bool)
    delta = teacher_log_probs.detach().float() - old_log_probs.detach().float()
    delta = torch.where(mask, delta, torch.zeros_like(delta))
    if not bool(torch.isfinite(delta).all().item()):
        raise FloatingPointError("Local distillation feedback contains a non-finite value.")
    return delta


def chunk_means(
    values: torch.Tensor,
    chunk_ids: torch.Tensor,
    num_chunks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-chunk means ``D`` and sizes ``|C|`` with shape ``[B, K_max]``."""

    if values.shape != chunk_ids.shape or values.ndim != 2:
        raise ValueError("values and chunk_ids must share one [B, R] shape.")
    batch = values.size(0)
    k_max = max(int(num_chunks.max().item()) if num_chunks.numel() else 0, 1)
    valid = chunk_ids.ge(0)
    index = chunk_ids.clamp_min(0)
    sums = torch.zeros(batch, k_max, dtype=torch.float32, device=values.device)
    sizes = torch.zeros(batch, k_max, dtype=torch.float32, device=values.device)
    sums.scatter_add_(1, index, torch.where(valid, values.detach().float(), torch.zeros_like(values, dtype=torch.float32)))
    sizes.scatter_add_(1, index, valid.float())
    chunk_valid = torch.arange(k_max, device=values.device).unsqueeze(0) < num_chunks.to(values.device).unsqueeze(-1)
    if bool((chunk_valid & sizes.eq(0)).any().item()):
        raise ValueError("Every declared chunk must contain at least one token.")
    means = torch.where(chunk_valid, sums / sizes.clamp_min(1.0), torch.zeros_like(sums))
    sizes = torch.where(chunk_valid, sizes, torch.zeros_like(sizes))
    return means, sizes


def finite_horizon_chunk_credit(
    chunk_values: torch.Tensor,
    num_chunks: torch.Tensor,
    *,
    horizon: int,
    discount: float,
) -> torch.Tensor:
    """Discount-mass-normalized finite-horizon chunk return-to-go ``G``."""

    if chunk_values.ndim != 2:
        raise ValueError("chunk_values must have shape [B, K].")
    if int(horizon) < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}.")
    if not (math.isfinite(float(discount)) and 0.0 < float(discount) <= 1.0):
        raise ValueError(f"discount must be in (0, 1], got {discount}.")
    batch, k_max = chunk_values.shape
    device = chunk_values.device
    values = chunk_values.detach().float()
    chunk_valid = torch.arange(k_max, device=device).unsqueeze(0) < num_chunks.to(device).unsqueeze(-1)
    numerator = torch.zeros(batch, k_max, dtype=torch.float32, device=device)
    denominator = torch.zeros(batch, k_max, dtype=torch.float32, device=device)
    for offset in range(min(int(horizon), max(k_max - 1, 0)) + 1):
        weight = float(discount) ** offset
        future_valid = chunk_valid[:, offset:].float()
        numerator[:, : k_max - offset] += weight * values[:, offset:] * future_valid
        denominator[:, : k_max - offset] += weight * future_valid
    credit = numerator / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    return torch.where(chunk_valid, credit, torch.zeros_like(credit))


def clip_chunk_credit(credit: torch.Tensor, clip: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``A^SD = sg[clip(G, -c, c)]`` and a boolean mask of clipped chunks."""

    if not (math.isfinite(float(clip)) and float(clip) > 0.0):
        raise ValueError(f"credit clip must be positive and finite, got {clip}.")
    value = credit.detach().float()
    return value.clamp(-float(clip), float(clip)), value.abs().gt(float(clip))


def broadcast_chunk_values(chunk_values: torch.Tensor, chunk_ids: torch.Tensor) -> torch.Tensor:
    """Broadcast ``[B, K]`` chunk values to their ``[B, R]`` tokens (zero on padding)."""

    valid = chunk_ids.ge(0)
    gathered = torch.gather(chunk_values.float(), 1, chunk_ids.clamp_min(0))
    return torch.where(valid, gathered, torch.zeros_like(gathered))


def equal_chunk_token_weights(
    chunk_ids: torch.Tensor,
    chunk_sizes: torch.Tensor,
    num_chunks: torch.Tensor,
    row_valid: torch.Tensor,
) -> torch.Tensor:
    """Token weights ``1 / (K_i |C_{i,k}|)`` so every chunk of a row has equal total weight."""

    valid = chunk_ids.ge(0) & row_valid.to(torch.bool).unsqueeze(-1) & num_chunks.gt(0).unsqueeze(-1)
    sizes = torch.gather(chunk_sizes.float(), 1, chunk_ids.clamp_min(0))
    weights = 1.0 / (num_chunks.clamp_min(1).float().unsqueeze(-1) * sizes.clamp_min(1.0))
    return torch.where(valid, weights, torch.zeros_like(weights))


@dataclass(frozen=True)
class RewardAnchoredFactors:
    """Result of ``reward_anchored_factors`` (``[B, R]`` unless noted)."""

    factor: torch.Tensor
    """m; 1 where not applied."""
    applied: torch.Tensor
    """Tokens inside a chunk of a distillable row with A != 0."""
    clipped: torch.Tensor
    """Applied tokens whose ``exp(sign(A) * credit)`` lay outside the bounds."""
    row_credit: torch.Tensor
    """``[B]`` token-weighted mean of G over the row's chunk tokens, before any centring (0 without chunks)."""
    has_chunks: torch.Tensor
    """``[B]`` rows with chunk tokens (rows with distillation targets)."""


def reward_anchored_factors(
    task_advantages: torch.Tensor,
    chunk_credit: torch.Tensor,
    chunk_ids: torch.Tensor,
    *,
    clip: float,
    center: str,
) -> RewardAnchoredFactors:
    """Factors ``m = clip(exp(sign(A) * (G - Gbar)), 1 - clip, 1 + clip)`` of the reward-anchored reweighting.

    ``task_advantages``: ``[B, R]`` GRPO advantages (constant per row on its response tokens);
    ``chunk_credit``: ``[B, K]`` chunk credit ``G`` (unclipped); ``chunk_ids``: ``[B, R]`` (``-1`` outside the
    chunks of distillable rows).  ``center``: ``row`` subtracts each row's token-weighted mean credit ``Gbar``
    (see the module docstring), ``none`` uses ``G`` as it is (``Gbar = 0``).  ``clip = 0`` gives factors of
    exactly 1 (plain GRPO).
    """

    if not (math.isfinite(float(clip)) and 0.0 <= float(clip) < 1.0):
        raise ValueError(f"the reweighting clip must be in [0, 1), got {clip}.")
    if center not in REWEIGHT_CENTER_MODES:
        raise ValueError(f"center must be one of {REWEIGHT_CENTER_MODES}, got {center!r}.")
    if task_advantages.shape != chunk_ids.shape or task_advantages.ndim != 2:
        raise ValueError("task_advantages and chunk_ids must share one [B, R] shape.")
    if chunk_credit.ndim != 2 or chunk_credit.size(0) != chunk_ids.size(0):
        raise ValueError("chunk_credit must have shape [B, K].")
    sign = torch.sign(task_advantages.detach().float())
    tokens = chunk_ids.ge(0)
    applied = tokens & sign.ne(0)
    credit = broadcast_chunk_values(chunk_credit.detach().float(), chunk_ids)
    if not bool(torch.isfinite(credit).all().item()):
        raise FloatingPointError("The chunk credit contains a non-finite value.")
    counts = tokens.float().sum(dim=-1)
    row_credit = (credit * tokens.float()).sum(dim=-1) / counts.clamp_min(1.0)
    if center == "row":
        credit = torch.where(tokens, credit - row_credit.unsqueeze(-1), torch.zeros_like(credit))
    raw = torch.exp(sign * credit)  # may overflow to inf for a huge credit; the clip bounds it
    low, high = 1.0 - float(clip), 1.0 + float(clip)
    factor = torch.where(applied, raw.clamp(low, high), torch.ones_like(raw))
    clipped = applied & (raw.lt(low) | raw.gt(high))
    return RewardAnchoredFactors(
        factor=factor, applied=applied, clipped=clipped, row_credit=row_credit, has_chunks=counts.gt(0)
    )


def reweight_metrics(
    factors: RewardAnchoredFactors, task_advantages: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, float]:
    """``trsi/reweight_*`` metrics of one batch.

    ``token_fraction``: reweighted tokens among the response tokens with task signal (the rest: rows without
    evidence views or invalid rows); ``factor_mean/min/max``, ``up_fraction`` (m > 1) and ``clipped_fraction``
    over the reweighted tokens; ``factor_mean_pos`` / ``_neg``: mean m on rows with A > 0 / A < 0 (about 1
    with ``center=row``; with ``none``, a difference between the two pushes the tokens a group shares);
    ``adv_scale``: sum |m A| / sum |A| over the tokens with task signal (the change of the update's total
    advantage mass); ``row_credit_mean`` (``_pos`` / ``_neg``): the mean of the rows' mean credit Gbar over the
    rows with targets (with A > 0 / A < 0).  ``row_credit_mean`` is the bias that centring removes;
    ``_pos`` above ``_neg`` means the teacher prefers the responses the reward prefers.
    """

    factor, clipped = factors.factor, factors.clipped
    signal = response_mask.to(torch.bool) & task_advantages.ne(0)
    applied = factors.applied.to(torch.bool) & signal
    metrics: dict[str, float] = {}
    signal_count = int(signal.sum().item())
    applied_count = int(applied.sum().item())
    metrics["trsi/reweight_token_fraction"] = float(applied_count / signal_count) if signal_count else 0.0
    if applied_count:
        values = factor[applied].float()
        metrics["trsi/reweight_factor_mean"] = float(values.mean().item())
        metrics["trsi/reweight_factor_min"] = float(values.min().item())
        metrics["trsi/reweight_factor_max"] = float(values.max().item())
        metrics["trsi/reweight_up_fraction"] = float(values.gt(1.0).float().mean().item())
        metrics["trsi/reweight_clipped_fraction"] = float(clipped[applied].float().mean().item())
        for name, side in (("pos", task_advantages.gt(0)), ("neg", task_advantages.lt(0))):
            chosen = applied & side
            if bool(chosen.any().item()):
                metrics[f"trsi/reweight_factor_mean_{name}"] = float(factor[chosen].float().mean().item())
    if signal_count:
        magnitude = task_advantages.detach().float().abs()[signal]
        metrics["trsi/reweight_adv_scale"] = float((magnitude * factor[signal].float()).sum().item() / magnitude.sum().item())
    rows = factors.has_chunks.to(torch.bool)
    if bool(rows.any().item()):
        row_sign = torch.sign((task_advantages.detach().float() * response_mask.float()).sum(dim=-1))
        credit = factors.row_credit.float()
        metrics["trsi/reweight_row_credit_mean"] = float(credit[rows].mean().item())
        for name, side in (("pos", row_sign.gt(0)), ("neg", row_sign.lt(0))):
            chosen = rows & side
            if bool(chosen.any().item()):
                metrics[f"trsi/reweight_row_credit_mean_{name}"] = float(credit[chosen].mean().item())
    return metrics
