"""Full-vocabulary token teacher distributions (method sections 5, 6, 9 and 10).

Shared interface (section 5)::

    q^m_{r,i,t}(v) = softmax_v[ log p^m_{b,r}(v | h_{i,t}, E^m_i) + Delta^m_r(v | h_{i,t}, E^m_i; S_r) ]

Route A (section 6) uses the post-task-update snapshot as base teacher and the
probability change relative to the pre-task-update anchor as correction::

    d(v)      = log p_{theta+}(v | h, E) - log p_{theta-}(v | h, E)
    Delta(v)  = clip(alpha * d(v), -c_Delta, c_Delta)
    q_prop    = softmax_v[ log p_{theta+}(v | h, E) + Delta(v) ]
    q_used    = (1 - beta) * p_{theta+} + beta * q_prop             (section 9)

Contracts enforced here:

* every position gets one complete distribution normalized over the *full*
  vocabulary; the sampled token is only used to *read* the finished
  distribution (no token-specific temperature or normalization);
* the correction follows one rule for every candidate ``v``;
* all values are detached and computed in float32 (the teacher is frozen and
  receives no gradient);
* with ``e^{-c}`` <= normalizer ratio <= ``e^{c}``, the bounded correction
  satisfies ``e^{-2 c_Delta} <= q_prop / p_b <= e^{2 c_Delta}``; the realized
  maximum is returned as a diagnostic so a violation is observable.

Evidence views (section 10) are combined *after* each view has been
normalized, by an arithmetic probability mixture over the available views.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


__all__ = [
    "RouteATeacherRule",
    "TeacherTokenScores",
    "mix_view_log_probs",
    "route_a_token_log_probs",
]


@dataclass(frozen=True)
class RouteATeacherRule:
    """Frozen construction rule of one teacher version (the non-tensor part of S_r).

    ``use_anchor=False`` (or ``alpha == 0`` or ``base_mix == 0``) reduces the
    rule to the evidence-conditioned base teacher, i.e. ``Delta = 0``.
    """

    alpha: float = 0.0
    """Extrapolation strength alpha >= 0 (not a temperature or a loss weight)."""
    correction_clip: Optional[float] = 1.0
    """c_Delta: bound on the additive correction; ``None`` disables the bound."""
    base_mix: float = 1.0
    """beta in [0, 1]: probability mixture weight of the proposed teacher."""
    temperature: float = 1.0
    """tau: one softmax temperature applied to every candidate (draft default 1)."""
    use_anchor: bool = False
    """Whether an anchor snapshot is part of this teacher version."""

    def __post_init__(self) -> None:
        if not (math.isfinite(float(self.alpha)) and float(self.alpha) >= 0.0):
            raise ValueError(f"alpha must be finite and >= 0, got {self.alpha}.")
        if self.correction_clip is not None and not (
            math.isfinite(float(self.correction_clip)) and float(self.correction_clip) > 0.0
        ):
            raise ValueError(f"correction_clip must be positive and finite when set, got {self.correction_clip}.")
        if not (math.isfinite(float(self.base_mix)) and 0.0 <= float(self.base_mix) <= 1.0):
            raise ValueError(f"base_mix must be in [0, 1], got {self.base_mix}.")
        if not (math.isfinite(float(self.temperature)) and float(self.temperature) > 0.0):
            raise ValueError(f"temperature must be positive and finite, got {self.temperature}.")

    @property
    def correction_active(self) -> bool:
        """True when the anchor changes the target (otherwise Delta == 0 exactly)."""

        return bool(self.use_anchor and float(self.alpha) > 0.0 and float(self.base_mix) > 0.0)

    @property
    def ratio_bound(self) -> float:
        """Upper bound on |log q_prop - log p_b| implied by ``correction_clip``."""

        if not self.correction_active:
            return 0.0
        if self.correction_clip is None:
            return math.inf
        return 2.0 * float(self.correction_clip)

    def to_dict(self) -> dict:
        return {
            "alpha": float(self.alpha),
            "correction_clip": None if self.correction_clip is None else float(self.correction_clip),
            "base_mix": float(self.base_mix),
            "temperature": float(self.temperature),
            "use_anchor": bool(self.use_anchor),
        }


@dataclass
class TeacherTokenScores:
    """Per-position teacher values read at the realized tokens of one response."""

    log_q: torch.Tensor
    """log q_used(y_t | h_t), shape [N]."""
    log_base: torch.Tensor
    """log p_b(y_t | h_t) of the base (post-update) teacher, shape [N]."""
    kl_to_base: torch.Tensor
    """KL(q_used || p_b) over the full vocabulary, shape [N]."""
    clip_fraction: torch.Tensor
    """Fraction of vocabulary entries whose |alpha * d| exceeded c_Delta, shape [N]."""
    max_log_ratio: torch.Tensor
    """max_v |log q_prop(v) - log p_b(v)|, shape [N] (bounded by 2 c_Delta)."""
    entropy: torch.Tensor
    """Entropy of q_used, shape [N]."""
    candidate_log_q: Optional[torch.Tensor]
    """log q_used at the candidate tokens, shape [N, k] (package v6, Route C), or ``None`` when none were given."""


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} contains NaN or infinity; teacher targets must be finite.")


@torch.no_grad()
def route_a_token_log_probs(
    updated_logits: torch.Tensor,
    tokens: torch.Tensor,
    rule: RouteATeacherRule,
    anchor_logits: Optional[torch.Tensor] = None,
    *,
    position_chunk: int = 256,
    candidate_ids: Optional[torch.Tensor],
) -> TeacherTokenScores:
    """Construct Route-A teacher distributions and read them at ``tokens`` (and at ``candidate_ids``).

    Args:
        updated_logits: ``[N, V]`` logits of the evidence-conditioned post-update
            snapshot ``theta+`` at the N response positions (the base teacher).
        tokens: ``[N]`` realized student tokens ``y_t``.
        rule: frozen teacher-construction rule of this stage.
        anchor_logits: ``[N, V]`` logits of the anchor ``theta-`` at the same
            prefixes, evidence and vocabulary.  Required when
            ``rule.correction_active``.
        position_chunk: number of positions processed at once (memory bound
            only; results are independent of it).
        candidate_ids: ``[N, k]`` token ids at which the finished distributions
            are also read (the student's top candidates of the Route C inner
            loop), or ``None``.  Like ``tokens`` they only index the finished
            distributions.

    Returns:
        :class:`TeacherTokenScores` with float32 tensors on the logits device.
    """

    if updated_logits.ndim != 2:
        raise ValueError(f"updated_logits must have shape [N, V], got {tuple(updated_logits.shape)}.")
    num_positions, vocab_size = updated_logits.shape
    if tokens.ndim != 1 or tokens.numel() != num_positions:
        raise ValueError(f"tokens must have shape [{num_positions}], got {tuple(tokens.shape)}.")
    if rule.correction_active:
        if anchor_logits is None:
            raise ValueError("An active Route-A correction requires anchor_logits.")
        if tuple(anchor_logits.shape) != (num_positions, vocab_size):
            raise ValueError(
                f"anchor_logits must match updated_logits {(num_positions, vocab_size)}, "
                f"got {tuple(anchor_logits.shape)}."
            )
    if int(position_chunk) <= 0:
        raise ValueError("position_chunk must be positive.")
    tokens = tokens.to(device=updated_logits.device, dtype=torch.long)
    if num_positions and (int(tokens.min()) < 0 or int(tokens.max()) >= vocab_size):
        raise ValueError("Realized tokens must index the teacher vocabulary.")
    if candidate_ids is not None:
        if candidate_ids.ndim != 2 or candidate_ids.size(0) != num_positions:
            raise ValueError(f"candidate_ids must have shape [{num_positions}, k], got {tuple(candidate_ids.shape)}.")
        candidate_ids = candidate_ids.to(device=updated_logits.device, dtype=torch.long)
        if candidate_ids.numel() and (int(candidate_ids.min()) < 0 or int(candidate_ids.max()) >= vocab_size):
            raise ValueError("Candidate tokens must index the teacher vocabulary.")

    device = updated_logits.device
    outputs = {
        name: torch.empty(num_positions, dtype=torch.float32, device=device)
        for name in ("log_q", "log_base", "kl_to_base", "clip_fraction", "max_log_ratio", "entropy")
    }
    outputs["candidate_log_q"] = (
        None
        if candidate_ids is None
        else torch.empty(num_positions, candidate_ids.size(1), dtype=torch.float32, device=device)
    )
    temperature = float(rule.temperature)
    beta = float(rule.base_mix)
    clip = None if rule.correction_clip is None else float(rule.correction_clip)
    log_beta = math.log(beta) if beta > 0.0 else -math.inf
    log_one_minus_beta = math.log1p(-beta) if beta < 1.0 else -math.inf

    for start in range(0, num_positions, int(position_chunk)):
        end = min(start + int(position_chunk), num_positions)
        index = tokens[start:end].unsqueeze(-1)
        plus = updated_logits[start:end].detach().float()
        _require_finite("updated teacher logits", plus)
        log_base = torch.log_softmax(plus / temperature, dim=-1)
        del plus

        if rule.correction_active:
            minus = anchor_logits[start:end].detach().float()
            _require_finite("anchor teacher logits", minus)
            log_anchor = torch.log_softmax(minus / temperature, dim=-1)
            del minus
            correction = (log_base - log_anchor).mul_(float(rule.alpha))
            del log_anchor
            if clip is not None:
                clip_fraction = correction.abs().gt(clip).float().mean(dim=-1)
                correction.clamp_(-clip, clip)
            else:
                clip_fraction = torch.zeros(end - start, dtype=torch.float32, device=device)
            log_prop = torch.log_softmax(log_base + correction, dim=-1)
            del correction
            max_log_ratio = (log_prop - log_base).abs().amax(dim=-1)
        else:
            clip_fraction = torch.zeros(end - start, dtype=torch.float32, device=device)
            log_prop = log_base
            max_log_ratio = torch.zeros(end - start, dtype=torch.float32, device=device)

        if beta >= 1.0 or not rule.correction_active:
            log_used = log_prop
        elif beta <= 0.0:
            log_used = log_base
        else:
            # Probability mixture of two normalized full-vocabulary distributions.
            log_used = torch.logaddexp(log_base + log_one_minus_beta, log_prop + log_beta)

        used_probs = log_used.exp()
        outputs["log_q"][start:end] = log_used.gather(-1, index).squeeze(-1)
        if candidate_ids is not None:
            outputs["candidate_log_q"][start:end] = log_used.gather(-1, candidate_ids[start:end])
        outputs["log_base"][start:end] = log_base.gather(-1, index).squeeze(-1)
        outputs["kl_to_base"][start:end] = (used_probs * (log_used - log_base)).sum(dim=-1).clamp_min(0.0)
        outputs["entropy"][start:end] = -(used_probs * log_used).sum(dim=-1)
        outputs["clip_fraction"][start:end] = clip_fraction
        outputs["max_log_ratio"][start:end] = max_log_ratio
        del log_base, log_prop, log_used, used_probs

    for name in ("log_q", "log_base"):
        _require_finite(f"teacher {name}", outputs[name])
    if outputs["candidate_log_q"] is not None:
        _require_finite("teacher candidate log_q", outputs["candidate_log_q"])
    return TeacherTokenScores(**outputs)


def mix_view_log_probs(
    view_log_probs: torch.Tensor,
    view_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Arithmetic probability mixture over the available evidence views.

    ``q(v) = (1 / |M_i|) * sum_{m in M_i} q^m(v)`` evaluated at the realized
    token (or at each candidate token), i.e. ``logsumexp_m(log q^m) - log |M_i|``.
    Averaging log probabilities would be a different (geometric) fusion rule.

    Args:
        view_log_probs: ``[B, M, R]`` per-view log probabilities at the
            realized tokens, or ``[B, M, R, k]`` at k candidate tokens (values
            of unavailable views are ignored).
        view_mask: ``[B, M]`` boolean availability of each view.

    Returns:
        ``(log_q, view_count)`` with shapes ``[B, R]`` (or ``[B, R, k]``) and
        ``[B]``.  Rows without any available view return zeros and
        ``view_count == 0``; they must be excluded from distillation by the caller.
    """

    if view_log_probs.ndim not in (3, 4):
        raise ValueError(f"view_log_probs must have shape [B, M, R] or [B, M, R, k], got {tuple(view_log_probs.shape)}.")
    batch, views = view_log_probs.shape[:2]
    if tuple(view_mask.shape) != (batch, views):
        raise ValueError(f"view_mask must have shape {(batch, views)}, got {tuple(view_mask.shape)}.")
    mask = view_mask.to(device=view_log_probs.device, dtype=torch.bool)
    trailing = (1,) * (view_log_probs.ndim - 2)
    values = view_log_probs.detach().float().masked_fill(~mask.view(batch, views, *trailing), float("-inf"))
    count = mask.sum(dim=-1)
    shape = (batch,) + (1,) * (view_log_probs.ndim - 2)
    mixed = torch.logsumexp(values, dim=1) - torch.log(count.clamp_min(1).to(torch.float32)).view(shape)
    mixed = torch.where(count.gt(0).view(shape), mixed, torch.zeros_like(mixed))
    return mixed, count
