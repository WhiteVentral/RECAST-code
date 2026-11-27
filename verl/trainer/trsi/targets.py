"""Driver-side assembly of Video-t-RSI distillation targets for one rollout batch.

Given the realized student tokens, the old-policy log probabilities and the
frozen teacher's log probabilities at the same prefixes, this module

1. partitions each valid response into semantic chunks (no gradients, text only),
2. optionally verifies structured fields against the original annotations,
3. computes ``delta``, chunk means ``D``, the finite-horizon credit ``G``,
   the clipped credit ``A^SD`` and the equal-chunk token weights.

Rows contribute to ``L_SD`` only when they are system-valid, have at least one
available evidence view and at least one generated token.  No row is routed
by task success or failure.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import torch

from .chunks import ChunkedResponse, ChunkingConfig, TokenTextAligner, chunk_response
from .credit import (
    broadcast_chunk_values,
    chunk_ids_from_spans,
    chunk_means,
    clip_chunk_credit,
    equal_chunk_token_weights,
    finite_horizon_chunk_credit,
    local_token_feedback,
)
from .verify import STATUS_CONTRADICTED, STATUS_SUPPORTED, VerificationConfig, build_row_evidence, verify_response


__all__ = ["CreditConfig", "SDTargets", "build_sd_targets"]


@dataclass(frozen=True)
class CreditConfig:
    horizon: int = 2
    """H: number of subsequent chunks considered (0 keeps only the current chunk)."""
    discount: float = 0.8
    """gamma in (0, 1]: discount by chunk distance."""
    clip: float = 1.0
    """c: symmetric clip of the chunk credit."""

    def __post_init__(self) -> None:
        if int(self.horizon) < 0:
            raise ValueError("credit horizon must be non-negative.")
        if not (math.isfinite(float(self.discount)) and 0.0 < float(self.discount) <= 1.0):
            raise ValueError("credit discount must be in (0, 1].")
        if not (math.isfinite(float(self.clip)) and float(self.clip) > 0.0):
            raise ValueError("credit clip must be positive and finite.")


@dataclass
class SDTargets:
    advantages: torch.Tensor
    """``[B, R]`` per-token broadcast of ``A^SD`` (zero outside valid chunks)."""
    token_weights: torch.Tensor
    """``[B, R]`` equal-chunk weights ``1 / (K_i |C_{i,k}|)``."""
    row_mask: torch.Tensor
    """``[B]`` float mask of rows that enter ``L_SD``."""
    delta: torch.Tensor
    chunk_ids: torch.Tensor
    num_chunks: torch.Tensor
    chunk_feedback: torch.Tensor
    chunk_credit: torch.Tensor
    chunked: list[Optional[ChunkedResponse]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def _check_contiguous_mask(response_mask: torch.Tensor) -> torch.Tensor:
    lengths = response_mask.long().sum(dim=-1)
    positions = torch.arange(response_mask.size(1), device=response_mask.device).unsqueeze(0)
    expected = positions < lengths.unsqueeze(-1)
    if not bool(torch.equal(expected, response_mask.bool())):
        raise ValueError("response_mask must be a contiguous prefix of ones for every row.")
    return lengths


def build_sd_targets(
    *,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    view_count: torch.Tensor,
    row_valid: torch.Tensor,
    aligner: TokenTextAligner,
    chunk_cfg: ChunkingConfig,
    credit_cfg: CreditConfig,
    rows: Optional[Sequence[Mapping[str, Any]]] = None,
    verify_cfg: Optional[VerificationConfig] = None,
) -> SDTargets:
    responses = responses.detach().cpu()
    response_mask = response_mask.detach().cpu()
    batch, response_length = responses.shape
    for name, value in (("old_log_probs", old_log_probs), ("teacher_log_probs", teacher_log_probs)):
        if tuple(value.shape) != (batch, response_length):
            raise ValueError(f"{name} must have shape {(batch, response_length)}, got {tuple(value.shape)}.")
    if tuple(view_count.shape) != (batch,) or tuple(row_valid.shape) != (batch,):
        raise ValueError("view_count and row_valid must have shape [B].")
    if rows is not None and len(rows) != batch:
        raise ValueError("rows must align with the batch.")
    lengths = _check_contiguous_mask(response_mask)

    spans_per_row: list[list[tuple[int, int]]] = []
    chunked_rows: list[Optional[ChunkedResponse]] = []
    row_ok: list[bool] = []
    for index in range(batch):
        length = int(lengths[index])
        valid = bool(row_valid[index]) and int(view_count[index]) > 0 and length > 0
        if not valid:
            spans_per_row.append([])
            chunked_rows.append(None)
            row_ok.append(False)
            continue
        chunked = chunk_response(responses[index, :length].tolist(), aligner, chunk_cfg)
        if verify_cfg is not None and rows is not None:
            verify_response(chunked, build_row_evidence(rows[index]), verify_cfg)
        spans = chunked.token_spans()
        if not spans or spans[-1][1] != length:
            raise RuntimeError(f"Chunking of row {index} does not cover its {length} generated tokens.")
        spans_per_row.append(spans)
        chunked_rows.append(chunked)
        row_ok.append(True)

    chunk_ids, num_chunks = chunk_ids_from_spans(spans_per_row, response_length)
    row_mask = torch.tensor(row_ok, dtype=torch.bool)
    token_mask = chunk_ids.ge(0)
    delta = local_token_feedback(teacher_log_probs.detach().cpu(), old_log_probs.detach().cpu(), token_mask)
    feedback, sizes = chunk_means(delta, chunk_ids, num_chunks)
    credit = finite_horizon_chunk_credit(
        feedback, num_chunks, horizon=int(credit_cfg.horizon), discount=float(credit_cfg.discount)
    )
    clipped_credit, clipped = clip_chunk_credit(credit, float(credit_cfg.clip))
    advantages = broadcast_chunk_values(clipped_credit, chunk_ids)
    weights = equal_chunk_token_weights(chunk_ids, sizes, num_chunks, row_mask)

    metrics = _metrics(
        delta=delta,
        token_mask=token_mask,
        feedback=feedback,
        credit=credit,
        clipped=clipped,
        sizes=sizes,
        num_chunks=num_chunks,
        row_mask=row_mask,
        chunked_rows=chunked_rows,
    )
    return SDTargets(
        advantages=advantages,
        token_weights=weights,
        row_mask=row_mask.float(),
        delta=delta,
        chunk_ids=chunk_ids,
        num_chunks=num_chunks,
        chunk_feedback=feedback,
        chunk_credit=credit,
        chunked=chunked_rows,
        metrics=metrics,
    )


def _metrics(
    *,
    delta: torch.Tensor,
    token_mask: torch.Tensor,
    feedback: torch.Tensor,
    credit: torch.Tensor,
    clipped: torch.Tensor,
    sizes: torch.Tensor,
    num_chunks: torch.Tensor,
    row_mask: torch.Tensor,
    chunked_rows: Sequence[Optional[ChunkedResponse]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    batch = row_mask.numel()
    metrics["trsi/sd_row_fraction"] = float(row_mask.float().mean().item()) if batch else 0.0
    tokens = token_mask.float()
    token_count = float(tokens.sum().item())
    if token_count > 0:
        metrics["trsi/delta_mean"] = float((delta * tokens).sum().item() / token_count)
        metrics["trsi/delta_abs_mean"] = float((delta.abs() * tokens).sum().item() / token_count)
    k_max = feedback.size(1)
    chunk_valid = torch.arange(k_max).unsqueeze(0) < num_chunks.unsqueeze(-1)
    chunk_count = float(chunk_valid.sum().item())
    if chunk_count > 0:
        valid = chunk_valid.float()
        metrics["trsi/chunk_D_mean"] = float((feedback * valid).sum().item() / chunk_count)
        metrics["trsi/chunk_D_abs_mean"] = float((feedback.abs() * valid).sum().item() / chunk_count)
        metrics["trsi/chunk_G_mean"] = float((credit * valid).sum().item() / chunk_count)
        metrics["trsi/chunk_G_abs_mean"] = float((credit.abs() * valid).sum().item() / chunk_count)
        metrics["trsi/credit_clip_fraction"] = float((clipped.float() * valid).sum().item() / chunk_count)
        metrics["trsi/chunk_tokens_mean"] = float((sizes * valid).sum().item() / chunk_count)
        metrics["trsi/chunk_tokens_max"] = float(sizes.max().item())
    ok_rows = [chunked for chunked in chunked_rows if chunked is not None]
    if ok_rows:
        metrics["trsi/chunks_per_response"] = float(sum(len(chunked.chunks) for chunked in ok_rows) / len(ok_rows))
        modes = Counter(chunked.parse_mode for chunked in ok_rows)
        for mode, count in modes.items():
            metrics[f"trsi/parse_mode/{mode}"] = count / len(ok_rows)
        metrics["trsi/answer_chunk_fraction"] = sum(
            any(chunk.role == "answer" for chunk in chunked.chunks) for chunked in ok_rows
        ) / len(ok_rows)
        all_chunks = [(row, chunk) for row, chunked in enumerate(chunked_rows) if chunked is not None for chunk in chunked.chunks]
        roles = Counter(chunk.role for _, chunk in all_chunks)
        for role, count in roles.items():
            metrics[f"trsi/chunk_role/{role}"] = count / len(all_chunks)
        statuses = Counter(chunk.status for _, chunk in all_chunks)
        for status, count in statuses.items():
            metrics[f"trsi/chunk_status/{status}"] = count / len(all_chunks)
        decided = statuses.get(STATUS_SUPPORTED, 0) + statuses.get(STATUS_CONTRADICTED, 0)
        if decided:
            # Unknown chunks are excluded from the denominator on purpose.
            metrics["trsi/verified_error_rate"] = statuses.get(STATUS_CONTRADICTED, 0) / decided
        for status in statuses:
            values = [float(feedback[row, chunk.index]) for row, chunk in all_chunks if chunk.status == status]
            if values:
                metrics[f"trsi/chunk_D_by_status/{status}"] = sum(values) / len(values)
        total_claims = sum(len(chunk.claims) for _, chunk in all_chunks)
        if total_claims:
            metrics["trsi/claim_split_fraction"] = sum(chunked.claim_split_count for chunked in ok_rows) / total_claims
    return metrics
