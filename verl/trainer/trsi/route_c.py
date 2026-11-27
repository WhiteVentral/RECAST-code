"""Route C at the chunks the student has not absorbed: the inner loop of package v6 (method sections 8, 9, 12).

Every teacher-scored step with ``rsi.inner_loop=route_c`` runs, before its update:

1. **Gap.**  The trainer's old-log-prob forward also returns the student's top-k candidate tokens ``c_1..c_k`` at
   every response position with their log-probabilities ``pi(c_j)`` under the policy that generated the rollouts
   (``pi_old``, at the rollout temperature).  Teacher scoring reads the round's teacher ``q`` (Route A, mixed over
   the evidence views) at the same candidates.  Over the buckets ``(c_1, ..., c_k, rest of the vocabulary)``::

       u_t = KL( q_hat_t || pi_hat_t ),   q_hat = (q(c_1), ..., q(c_k), 1 - sum_j q(c_j)),  pi_hat likewise.

   ``u_t`` says how far the student's next-token choice at the prefix ``h_t`` still is from the teacher's.  It
   depends on the prefix and the two frozen distributions, not on the token that happened to be sampled at ``t``.

2. **Selection** (where to investigate; a gap is not the basis of a correction, method section 9).  A row with
   distillation targets is branched once: at the chunk position with the largest ``u_t`` among the positions whose
   candidates cover at least ``min_coverage`` of the student's probability (``w_t = sum_j pi(c_j)``) and that leave
   room for a continuation, provided that ``u_t >= min_gap``.  The chunk holding it is the row's least absorbed
   chunk.  The student generated this step's rollouts after its previous update, so the gap measures what that
   update left unabsorbed.

3. **Estimates** (method section 8).  From the exact prefix ``h_t`` followed by ``c_j``, ``continuations`` responses
   are sampled with the same policy (``mu = pi_old``, frozen for the step) and scored with the task reward::

       Q(c_j) = mean reward of its continuations,
       Q0     = sum_j pi(c_j) Q(c_j) / w      (the value of h_t under mu, over the investigated candidates).

   A candidate that ends the response (an end-of-sequence id) is scored once, without generation.  The rollout
   itself is not one of the samples of its own token ``y_t``: the branch position was chosen by comparing gaps that
   depend on the rollout's later tokens, so its reward is not an independent draw of ``Q(y_t)``.  Every investigated
   candidate gets the same ``continuations`` fresh samples.

4. **Teacher at the branch position** (its base is the student itself: the continuation rewards alone decide the
   direction, the teacher only where to look)::

       b_j   = kappa * w * clip(Q(c_j) - Q0, -c_Q, c_Q)
       q^C_j = pi(c_j) exp(b_j) / Z,   q^C_rest = (1 - sum_j pi(c_j)) / Z,
       Z     = sum_j pi(c_j) exp(b_j) + 1 - sum_j pi(c_j).

   Tokens outside the candidates keep the student's relative probabilities (no tilt), as do candidates without an
   estimate.  The estimates are frozen for the step's update and never reused at another step.

5. **Loss** (token level, inside the chunk): ``coef * mean over the branched rows of KL(q^C_hat || pi_theta_hat)`` at
   the branch position over the same buckets.  It moves the relative probabilities of the candidate tokens at that
   prefix, including candidates the student did not sample; the chunk credit cannot, since it gives every token of
   a chunk one value.  ``Q`` never enters the chunk credit ``G`` (method section 8).

Everything here is stop-gradient except :func:`bucket_log_probs` and :func:`bucket_kl` on the student's
log-probabilities in the update.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Hashable, Optional, Sequence

import numpy as np
import torch


__all__ = [
    "INNER_LOOP_MODES",
    "Branch",
    "ContinuationRequest",
    "RouteCSettings",
    "RouteCTargets",
    "bucket_kl",
    "bucket_log_probs",
    "candidate_gap",
    "candidate_values",
    "continuation_layout",
    "continuation_requests",
    "continuation_responses",
    "prompt_digest",
    "route_c_policy_loss",
    "route_c_seed_offset",
    "route_c_targets",
    "select_branches",
]


INNER_LOOP_MODES = ("route_c", "none")
"""``rsi.inner_loop``: ``route_c`` (package v6) or ``none`` (package v5: no inner loop).  Not ``off``: YAML and the
OmegaConf command line read an unquoted ``off`` as the boolean false."""

ROUTE_C_SEED_TAG = 1_000_003
"""Third SeedSequence entry of the continuation seeds; the rollout's own generation calls use 0, 1, ..."""

_SEED_LIMIT = 2**31
_MASS_CEILING = 1.0 - 1.0e-6
"""Candidate mass above this counts as full: the rest bucket keeps probability 1e-6 (finite logarithms)."""


@dataclass(frozen=True)
class RouteCSettings:
    """``rsi.route_c_*``: the frozen estimation rule of the inner loop."""

    candidates: int
    """k: candidates per branch position (the student's top-k at the prefix)."""
    continuations: int
    """N: continuations sampled per candidate."""
    min_gap: float
    """Smallest u_t (nats) that counts as not absorbed."""
    min_coverage: float
    """Smallest share of the student's probability the candidates must cover at a branch position."""
    value_clip: float
    """c_Q: bound of Q(c_j) - Q0 (reward units)."""
    kappa: float
    """kappa: tilt per unit of clipped value difference."""

    def __post_init__(self) -> None:
        for name in ("candidates", "continuations"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f"route_c {name} must be an integer, got {value!r}.")
        if int(self.candidates) < 2:
            raise ValueError(f"Route C needs at least 2 candidates per position, got {self.candidates}.")
        if int(self.continuations) < 1:
            raise ValueError(f"Route C needs at least 1 continuation per candidate, got {self.continuations}.")
        if not (math.isfinite(float(self.min_gap)) and float(self.min_gap) >= 0.0):
            raise ValueError(f"route_c min_gap must be finite and >= 0, got {self.min_gap}.")
        if not (math.isfinite(float(self.min_coverage)) and 0.0 <= float(self.min_coverage) <= 1.0):
            raise ValueError(f"route_c min_coverage must be in [0, 1], got {self.min_coverage}.")
        if not (math.isfinite(float(self.value_clip)) and float(self.value_clip) > 0.0):
            raise ValueError(f"route_c value_clip must be positive and finite, got {self.value_clip}.")
        if not (math.isfinite(float(self.kappa)) and float(self.kappa) >= 0.0):
            raise ValueError(f"route_c kappa must be finite and >= 0, got {self.kappa}.")


# ----------------------------------------------------------------------
# Buckets
# ----------------------------------------------------------------------
def bucket_log_probs(candidate_log_probs: torch.Tensor) -> torch.Tensor:
    """``[..., k]`` candidate log-probabilities -> ``[..., k + 1]`` log-probabilities of ``(c_1, ..., c_k, rest)``.

    The rest bucket holds ``1 - sum_j p(c_j)``.  Both distributions are normalized over the full vocabulary, so the
    candidate mass exceeds 1 only by rounding; it is capped at ``1 - 1e-6``.  Differentiable in the input.
    """

    values = candidate_log_probs.float()
    mass = torch.logsumexp(values, dim=-1, keepdim=True).exp().clamp(max=_MASS_CEILING)
    return torch.cat([values, torch.log1p(-mass)], dim=-1)


def bucket_kl(target: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
    """``KL(target || p)`` over the last dimension; ``target`` are probabilities (no gradient), ``log_probs`` of p."""

    if tuple(target.shape) != tuple(log_probs.shape):
        raise ValueError(f"target {tuple(target.shape)} and log_probs {tuple(log_probs.shape)} must match.")
    target = target.detach().float()
    log_target = torch.log(target.clamp_min(1.0e-12))
    return (target * (log_target - log_probs.float())).sum(dim=-1)


def candidate_gap(
    teacher_log_probs: torch.Tensor, student_log_probs: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-position gap ``u_t`` and candidate coverage ``w_t`` (both ``[B, R]``, 0 outside ``mask``).

    ``teacher_log_probs`` / ``student_log_probs``: ``[B, R, k]`` log-probabilities of the student's candidates under
    the teacher and under the student.
    """

    if tuple(teacher_log_probs.shape) != tuple(student_log_probs.shape) or teacher_log_probs.ndim != 3:
        raise ValueError("teacher and student candidate log-probabilities must share one [B, R, k] shape.")
    if tuple(mask.shape) != tuple(student_log_probs.shape[:2]):
        raise ValueError(f"mask must have shape {tuple(student_log_probs.shape[:2])}, got {tuple(mask.shape)}.")
    for name, value in (("teacher", teacher_log_probs), ("student", student_log_probs)):
        if not bool(torch.isfinite(value.float()).all().item()):
            raise FloatingPointError(f"The {name} candidate log-probabilities contain a non-finite value.")
    valid = mask.to(torch.bool)
    teacher = bucket_log_probs(teacher_log_probs.detach()).exp()
    gap = bucket_kl(teacher, bucket_log_probs(student_log_probs.detach())).clamp_min(0.0)
    coverage = student_log_probs.detach().float().exp().sum(dim=-1).clamp(max=1.0)
    zero = torch.zeros_like(gap)
    return torch.where(valid, gap, zero), torch.where(valid, coverage, zero)


# ----------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Branch:
    """One branch position: the candidates replace the token at ``position`` (prefix ``responses[row, :position]``)."""

    row: int
    position: int
    chunk: int
    gap: float
    coverage: float


def select_branches(
    gap: torch.Tensor,
    coverage: torch.Tensor,
    chunk_ids: torch.Tensor,
    row_mask: torch.Tensor,
    *,
    min_gap: float,
    min_coverage: float,
) -> tuple[list[Branch], torch.Tensor]:
    """At most one branch per row of ``row_mask`` (see the module docstring, step 2).

    Eligible positions lie in a chunk (``chunk_ids >= 0``), have candidate coverage ``>= min_coverage`` and leave
    room for a continuation (``position < R - 1``).  Returns the branches in row order and every row's largest
    eligible gap (``[B]``, ``nan`` for rows without targets or without an eligible position).
    """

    if gap.ndim != 2 or tuple(coverage.shape) != tuple(gap.shape) or tuple(chunk_ids.shape) != tuple(gap.shape):
        raise ValueError("gap, coverage and chunk_ids must share one [B, R] shape.")
    if tuple(row_mask.shape) != (gap.size(0),):
        raise ValueError(f"row_mask must have shape {(gap.size(0),)}, got {tuple(row_mask.shape)}.")
    batch, length = gap.shape
    positions = torch.arange(length).unsqueeze(0)
    eligible = chunk_ids.ge(0) & coverage.ge(float(min_coverage)) & positions.lt(length - 1)
    eligible &= row_mask.to(torch.bool).unsqueeze(-1)
    scores = gap.float().masked_fill(~eligible, float("-inf"))
    best, where = scores.max(dim=-1)
    row_best = torch.where(eligible.any(dim=-1), best, torch.full_like(best, float("nan")))
    branches = []
    for row in range(batch):
        value = float(row_best[row])
        if math.isnan(value) or value < float(min_gap):
            continue
        position = int(where[row])
        branches.append(
            Branch(
                row=row,
                position=position,
                chunk=int(chunk_ids[row, position]),
                gap=value,
                coverage=float(coverage[row, position]),
            )
        )
    return branches, row_best


# ----------------------------------------------------------------------
# Continuations
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ContinuationRequest:
    """One sampled continuation of ``prompt + tail``, the tail being the rollout's prefix plus a candidate."""

    branch: int
    candidate: int
    sample: int
    tail: tuple[int, ...]
    max_tokens: int
    seed: int


def route_c_seed_offset(base_seed: int, step: int, count: int) -> int:
    """First seed of a step's continuation requests (request ``i`` uses ``offset + i``, ``i < count``):
    deterministic per step, so a resumed run draws the same continuations, and different between steps."""

    count = int(count)
    if count < 0 or count >= _SEED_LIMIT:
        raise ValueError(f"Invalid number of continuation requests: {count}.")
    entropy = [int(base_seed) % (2**32), int(step), ROUTE_C_SEED_TAG]
    if min(entropy) < 0:
        raise ValueError(f"Seed inputs must be non-negative, got step={step}.")
    draw = int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])
    return draw % (_SEED_LIMIT - count)


def continuation_requests(
    branches: Sequence[Branch],
    *,
    responses: torch.Tensor,
    candidate_ids: torch.Tensor,
    continuations: int,
    terminal_ids: Sequence[int],
    skip_ids: Sequence[int],
    seed_offset: int,
) -> list[ContinuationRequest]:
    """The sampled continuations of every investigated candidate of every branch, in (branch, candidate, sample)
    order; request ``i`` is seeded with ``seed_offset + i``.

    ``responses``: ``[B, R]`` rollout tokens; ``candidate_ids``: ``[B, R, k]``.  A continuation may fill the response
    up to ``R`` tokens.  Candidates in ``terminal_ids`` end the response (scored once, see
    :func:`continuation_responses`); candidates in ``skip_ids`` (tokens the sampler can never produce, such as the
    vision placeholders) are not investigated.
    """

    length = int(responses.size(1))
    terminal = {int(value) for value in terminal_ids}
    skipped = {int(value) for value in skip_ids}
    requests: list[ContinuationRequest] = []
    for index, branch in enumerate(branches):
        prefix = [int(value) for value in responses[branch.row, : branch.position].tolist()]
        max_tokens = length - branch.position - 1
        if max_tokens < 1:
            raise ValueError(f"Branch at position {branch.position} leaves no room for a continuation (R={length}).")
        for candidate, token in enumerate(candidate_ids[branch.row, branch.position].tolist()):
            if int(token) in terminal or int(token) in skipped:
                continue
            tail = tuple(prefix + [int(token)])
            for sample in range(int(continuations)):
                requests.append(
                    ContinuationRequest(
                        branch=index,
                        candidate=candidate,
                        sample=sample,
                        tail=tail,
                        max_tokens=max_tokens,
                        seed=int(seed_offset) + len(requests),
                    )
                )
    return requests


def continuation_responses(
    branches: Sequence[Branch],
    requests: Sequence[ContinuationRequest],
    generated: Sequence[Optional[Sequence[int]]],
    *,
    responses: torch.Tensor,
    candidate_ids: torch.Tensor,
    terminal_ids: Sequence[int],
    skip_ids: Sequence[int],
    pad_id: int,
) -> tuple[list[tuple[int, int]], torch.Tensor, torch.Tensor]:
    """The responses to score: ``prefix + candidate (+ continuation)``, ``[M, R]`` padded with ``pad_id``.

    ``generated[i]`` holds the tokens vLLM generated for ``requests[i]`` (``None``: dropped, e.g. its prompt was not
    the rollout's).  Terminal candidates contribute one row each (not skipped ones).  Returns, per scored row, its
    ``(branch, candidate)``, the response tensor and the row lengths ``[M]`` (the caller applies the trainer's
    end-of-sequence rule on top).
    """

    if len(generated) != len(requests):
        raise ValueError(f"Got {len(generated)} generations for {len(requests)} requests.")
    length = int(responses.size(1))
    terminal = {int(value) for value in terminal_ids}
    skipped = {int(value) for value in skip_ids}
    keys: list[tuple[int, int]] = []
    rows: list[list[int]] = []
    for index, branch in enumerate(branches):
        prefix = [int(value) for value in responses[branch.row, : branch.position].tolist()]
        for candidate, token in enumerate(candidate_ids[branch.row, branch.position].tolist()):
            if int(token) in terminal and int(token) not in skipped:
                keys.append((index, candidate))
                rows.append(prefix + [int(token)])
    for request, tokens in zip(requests, generated):
        if tokens is None:
            continue
        keys.append((request.branch, request.candidate))
        rows.append(list(request.tail) + [int(value) for value in tokens])
    out = torch.full((len(rows), length), int(pad_id), dtype=torch.long)
    lengths = torch.zeros(len(rows), dtype=torch.long)
    for index, row in enumerate(rows):
        row = row[:length]
        out[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        lengths[index] = len(row)
    return keys, out, lengths


def continuation_layout(
    groups: Sequence[Hashable], costs: Sequence[float], world_size: int
) -> tuple[list[Optional[int]], int]:
    """Rows of the dispatched continuation batch: rank ``r`` gets rows ``[r * per_rank, (r + 1) * per_rank)``
    (``Dispatch.DP_COMPUTE_PROTO`` splits a batch into equal slices).

    Requests of one group (the rollout prompt they continue) stay on one rank, so its media are sent to that rank
    once; groups go largest total cost first to the least loaded rank.  Returns the request index of every row
    (``None``: padding, not generated) and ``per_rank``.
    """

    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world_size must be positive.")
    if len(groups) != len(costs):
        raise ValueError(f"Got {len(costs)} costs for {len(groups)} requests.")
    if not groups:
        return [], 0
    units: dict[Hashable, list[int]] = {}
    for index, key in enumerate(groups):
        units.setdefault(key, []).append(index)
    ordered = sorted(units.values(), key=lambda members: (-sum(float(costs[i]) for i in members), members[0]))
    loads = [0.0] * world_size
    assigned: list[list[int]] = [[] for _ in range(world_size)]
    for members in ordered:
        rank = min(range(world_size), key=lambda candidate: (loads[candidate], candidate))
        assigned[rank].extend(members)
        loads[rank] += sum(float(costs[i]) for i in members)
    per_rank = max(len(rows) for rows in assigned)
    layout: list[Optional[int]] = []
    for rows in assigned:
        layout.extend(rows + [None] * (per_rank - len(rows)))
    return layout, per_rank


def prompt_digest(token_ids: Sequence[int]) -> int:
    """64-bit digest (signed, fits an int64 tensor) of a token sequence: compares the prompt vLLM processed for a
    continuation with the prompt the trainer scores the rollout row with."""

    data = np.asarray([int(value) for value in token_ids], dtype=np.int64).tobytes()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little", signed=True)


# ----------------------------------------------------------------------
# Values and targets
# ----------------------------------------------------------------------
def candidate_values(
    keys: Sequence[tuple[int, int]], rewards: Sequence[float], *, branches: int, candidates: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum and count of the scored rewards of every ``(branch, candidate)``: two ``[branches, k]`` tensors."""

    if len(keys) != len(rewards):
        raise ValueError(f"Got {len(rewards)} rewards for {len(keys)} scored responses.")
    total = torch.zeros(int(branches), int(candidates), dtype=torch.float64)
    count = torch.zeros(int(branches), int(candidates), dtype=torch.float64)
    for (branch, candidate), reward in zip(keys, rewards):
        value = float(reward)
        if not math.isfinite(value):
            raise FloatingPointError(f"Non-finite continuation reward {value} (branch {branch}, candidate {candidate}).")
        total[branch, candidate] += value
        count[branch, candidate] += 1.0
    return total, count


@dataclass
class RouteCTargets:
    """Teacher distributions at the branch positions (one row per branch)."""

    target: torch.Tensor
    """``[n, k + 1]`` probabilities of ``(c_1, ..., c_k, rest)`` under q^C."""
    values: torch.Tensor
    """``[n, k]`` Q(c_j) (``nan`` without an estimate)."""
    baseline: torch.Tensor
    """``[n]`` Q0."""
    tilt: torch.Tensor
    """``[n, k]`` b_j (0 without an estimate)."""
    weight: torch.Tensor
    """``[n]`` w: the student's probability covered by the investigated candidates."""
    valid: torch.Tensor
    """``[n]`` bool: at least two candidates have an estimate (otherwise nothing is compared and no target is set)."""
    metrics: dict[str, float] = field(default_factory=dict)


def route_c_targets(
    student_log_probs: torch.Tensor,
    value_sum: torch.Tensor,
    value_count: torch.Tensor,
    *,
    kappa: float,
    value_clip: float,
) -> RouteCTargets:
    """q^C at every branch from the student's candidate log-probabilities ``[n, k]`` and the scored rewards."""

    if student_log_probs.ndim != 2 or tuple(value_sum.shape) != tuple(student_log_probs.shape):
        raise ValueError("student_log_probs and value_sum must share one [n, k] shape.")
    if tuple(value_count.shape) != tuple(value_sum.shape):
        raise ValueError("value_count must match value_sum.")
    if not (math.isfinite(float(value_clip)) and float(value_clip) > 0.0) or not (
        math.isfinite(float(kappa)) and float(kappa) >= 0.0
    ):
        raise ValueError(f"Invalid tilt settings: kappa={kappa}, value_clip={value_clip}.")
    probs = student_log_probs.detach().double().exp()
    investigated = value_count.gt(0)
    values = torch.where(investigated, value_sum.double() / value_count.double().clamp_min(1.0), torch.full_like(probs, float("nan")))
    covered = (probs * investigated.double()).sum(dim=-1)
    valid = investigated.sum(dim=-1).ge(2) & covered.gt(0)
    weights = probs * investigated.double() / covered.clamp_min(1.0e-12).unsqueeze(-1)
    baseline = (weights * torch.nan_to_num(values)).sum(dim=-1)
    difference = (torch.nan_to_num(values) - baseline.unsqueeze(-1)).clamp(-float(value_clip), float(value_clip))
    tilt = float(kappa) * covered.unsqueeze(-1) * difference * investigated.double() * valid.double().unsqueeze(-1)
    rest = (1.0 - probs.sum(dim=-1)).clamp_min(0.0)
    scaled = probs * tilt.exp()
    normalizer = scaled.sum(dim=-1) + rest
    target = torch.cat([scaled, rest.unsqueeze(-1)], dim=-1) / normalizer.unsqueeze(-1)
    result = RouteCTargets(
        target=target.float(),
        values=values.float(),
        baseline=torch.where(valid, baseline, torch.full_like(baseline, float("nan"))).float(),
        tilt=tilt.float(),
        weight=covered.float(),
        valid=valid,
    )
    if bool(valid.any()):
        rows = valid
        student = bucket_log_probs(student_log_probs.detach())[rows].double()
        result.metrics.update(
            {
                "trsi/route_c_value_spread": float(
                    (torch.nan_to_num(values, nan=-math.inf).amax(dim=-1) - torch.nan_to_num(values, nan=math.inf).amin(dim=-1))[rows]
                    .mean()
                    .item()
                ),
                "trsi/route_c_tilt_abs": float(tilt[rows].abs().sum().item() / max(int(investigated[rows].sum()), 1)),
                "trsi/route_c_target_kl": float(bucket_kl(target[rows], student).mean().item()),
                "trsi/route_c_top_changed": float(
                    target[rows][:, :-1].argmax(dim=-1).ne(probs[rows].argmax(dim=-1)).double().mean().item()
                ),
                "trsi/route_c_baseline": float(baseline[rows].mean().item()),
            }
        )
    return result


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------
def route_c_policy_loss(
    candidate_log_probs: torch.Tensor,
    target: torch.Tensor,
    *,
    global_rows: float,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The Route C term of one micro-batch: ``sum_rows KL(q^C_hat || pi_theta_hat) * world_size / global_rows``.

    ``candidate_log_probs``: ``[n, k]`` log-probabilities of the candidates under the policy being updated (with
    gradient), at the branch positions of the micro-batch's branched rows; ``target``: ``[n, k + 1]``.  Summed over
    the micro-batches of an optimizer step and averaged over the data-parallel ranks (FSDP divides gradients by the
    world size), this is the mean over the step's ``global_rows`` branched rows, the convention of the other terms
    (``losses.compute_rsi_policy_loss``).  Returns the term and the detached per-row KLs.
    """

    if candidate_log_probs.ndim != 2:
        raise ValueError(f"candidate_log_probs must have shape [n, k], got {tuple(candidate_log_probs.shape)}.")
    rows, candidates = candidate_log_probs.shape
    if tuple(target.shape) != (rows, candidates + 1):
        raise ValueError(f"target must have shape {(rows, candidates + 1)}, got {tuple(target.shape)}.")
    if not (math.isfinite(float(global_rows)) and float(global_rows) > 0.0):
        raise ValueError(f"global_rows must be positive, got {global_rows}.")
    if int(world_size) <= 0:
        raise ValueError("world_size must be positive.")
    row_kl = bucket_kl(target, bucket_log_probs(candidate_log_probs))
    return row_kl.sum() * float(world_size) / float(global_rows), row_kl.detach()
