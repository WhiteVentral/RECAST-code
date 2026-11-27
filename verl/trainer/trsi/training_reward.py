"""Generation termination metadata and training-only rewards (package v7).

Raw task rewards remain the benchmark's scores.  A separate cloned tensor zeros
only responses that actually exhausted their generation budget.  Length alone is
not a termination reason: vLLM V1 checks its length limit *before* EOS, so an EOS
on the last permitted token can have ``finish_reason='length'``.

Metadata collection is permissive for ordinary rollout/evaluation consumers.
The training helpers are strict: missing/aborted/unknown termination information
must not silently turn into either an unpenalized answer or a length heuristic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


TERMINATION_METADATA_KEYS = (
    "rollout_finish_reason",
    "rollout_stop_reason",
    "rollout_max_tokens",
    "rollout_requested_max_tokens",
    "rollout_generated_tokens",
    "rollout_last_token_id",
    "rollout_eos_token_ids",
    "rollout_stop_token_ids",
    "rollout_ended_with_eos",
    "rollout_ignore_eos",
)


def _token_ids(values: Any) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, (int, np.integer)):
        return (int(values),)
    return tuple(sorted({int(value) for value in values}))


def generation_budget(
    requested_max_tokens: int | None, *, prompt_tokens: int | None, max_model_len: int | None
) -> int | None:
    """Effective request budget, including the model's context limit.

    ``prompt_tokens`` must count vLLM's processed prompt (expanded multimodal
    placeholders), not the unprocessed text prompt. Unknown inputs stay unknown.
    """

    if requested_max_tokens is None or prompt_tokens is None or max_model_len is None:
        return None
    return min(int(requested_max_tokens), max(int(max_model_len) - int(prompt_tokens), 0))


def completion_metadata(
    output: Any,
    *,
    max_tokens: int | None,
    eos_token_ids: Sequence[int] | int | None,
    ignore_eos: bool = False,
    stop_token_ids: Sequence[int] | None = None,
    requested_max_tokens: int | None = None,
) -> dict[str, Any]:
    """Collect one vLLM completion without guessing absent finish information.

    ``max_tokens`` is the actual effective budget; ``requested_max_tokens`` is
    the SamplingParams limit before the model context limit is applied.
    """

    tokens = [int(value) for value in output.token_ids]
    eos_ids = _token_ids(eos_token_ids)
    reason = getattr(output, "finish_reason", None)
    return {
        "rollout_finish_reason": None if reason is None else str(reason),
        "rollout_stop_reason": getattr(output, "stop_reason", None),
        "rollout_max_tokens": None if max_tokens is None else int(max_tokens),
        "rollout_requested_max_tokens": (
            None if requested_max_tokens is None and max_tokens is None
            else int(max_tokens if requested_max_tokens is None else requested_max_tokens)
        ),
        "rollout_generated_tokens": len(tokens),
        "rollout_last_token_id": tokens[-1] if tokens else None,
        "rollout_eos_token_ids": eos_ids,
        "rollout_stop_token_ids": _token_ids(stop_token_ids),
        "rollout_ended_with_eos": bool(tokens and tokens[-1] in eos_ids),
        "rollout_ignore_eos": bool(ignore_eos),
    }


def terminal_metadata(token_id: int, eos_token_ids: Sequence[int] | int) -> dict[str, Any]:
    """A Route C candidate that itself is EOS, with no continuation request.

    The one token is the selected candidate, not a fabricated sampled suffix.
    It is a normal completed branch, even though its one-token budget is full.
    """

    if int(token_id) not in _token_ids(eos_token_ids):
        raise ValueError("A terminal Route C candidate must be an EOS token.")
    return completion_metadata(
        SimpleNamespace(token_ids=[int(token_id)], finish_reason="stop", stop_reason=None),
        max_tokens=1,
        eos_token_ids=eos_token_ids,
    )


def metadata_to_arrays(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """One-dimensional object arrays, including tuple-valued EOS/stop ID cells.

    These survive DataProto repeat/chunk/concat/reorder without accidentally
    turning equal-length token-ID tuples into a second array dimension.
    Padding records may be empty; only accepted real requests are validated.
    """

    arrays = {}
    for key in TERMINATION_METADATA_KEYS:
        values = np.empty(len(records), dtype=object)
        values[:] = [record.get(key) for record in records]
        arrays[key] = values
    return arrays


def termination_flags(non_tensor_batch: Mapping[str, Any], *, expected_rows: int) -> np.ndarray:
    """Return true length truncations, rejecting missing or unsupported metadata.

    EOS and configured stop tokens on the budget's last position are normal
    stops. EOS does not count as a stop when ``ignore_eos=True``. The engine's
    original reason remains in the metadata for auditing.
    """

    size = int(expected_rows)
    missing = [key for key in TERMINATION_METADATA_KEYS if key not in non_tensor_batch]
    if missing:
        raise ValueError(f"Missing termination metadata {missing}; regenerate rollouts with the v7 sampler.")
    for key in TERMINATION_METADATA_KEYS:
        values = non_tensor_batch[key]
        if not isinstance(values, (np.ndarray, list, tuple)) or len(values) != size:
            raise ValueError(f"Termination metadata {key} must have exactly {size} rows.")
        if isinstance(values, np.ndarray) and values.ndim != 1:
            raise ValueError(f"Termination metadata {key} must be one-dimensional.")
    truncated = np.zeros(size, dtype=bool)
    for row in range(size):
        item = {key: non_tensor_batch[key][row] for key in TERMINATION_METADATA_KEYS}
        reason = item["rollout_finish_reason"]
        if reason not in ("stop", "length"):
            raise ValueError(f"Rollout row {row} has missing or unsupported finish_reason={reason!r}.")
        budget, requested, count = (
            item["rollout_max_tokens"], item["rollout_requested_max_tokens"], item["rollout_generated_tokens"]
        )
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in (budget, requested, count)):
            raise ValueError(f"Rollout row {row} needs integer actual/requested budgets and generated token count.")
        if budget <= 0 or requested < budget or count < 0 or count > budget:
            raise ValueError(f"Rollout row {row} has inconsistent generated count/budget: {count}/{budget} (requested {requested}).")
        ignore = item["rollout_ignore_eos"]
        ended = item["rollout_ended_with_eos"]
        if not isinstance(ignore, (bool, np.bool_)) or not isinstance(ended, (bool, np.bool_)):
            raise ValueError(f"Rollout row {row} is missing explicit EOS/ignore_eos flags.")
        eos_ids, stop_ids = item["rollout_eos_token_ids"], item["rollout_stop_token_ids"]
        if not isinstance(eos_ids, (tuple, list, np.ndarray)) or not isinstance(stop_ids, (tuple, list, np.ndarray)):
            raise ValueError(f"Rollout row {row} is missing EOS or stop-token configuration.")
        eos_ids, stop_ids = _token_ids(eos_ids), _token_ids(stop_ids)
        if not eos_ids and not ignore:
            raise ValueError(f"Rollout row {row} has no EOS token IDs; cannot distinguish EOS at the length boundary.")
        last = item["rollout_last_token_id"]
        if (count == 0 and last is not None) or (count > 0 and not isinstance(last, (int, np.integer))):
            raise ValueError(f"Rollout row {row} has inconsistent final token metadata.")
        if bool(ended) != bool(count and last in eos_ids):
            raise ValueError(f"Rollout row {row} has inconsistent ended_with_eos metadata.")
        terminal_token = bool(count and ((not ignore and last in eos_ids) or last in stop_ids))
        truncated[row] = reason == "length" and not terminal_token
        if reason == "length" and count != budget:
            raise ValueError(f"Rollout row {row} reports length termination at {count} tokens before its budget {budget}.")
    return truncated


def effective_training_rewards(
    raw_rewards: torch.Tensor, non_tensor_batch: Mapping[str, Any], *, enabled: bool = True
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clone scalar or token-level raw rewards and zero actual length failures.

    Call before computing GRPO group means/stds. Do not turn length failures
    into system-invalid rows: their zero reward remains in the group and the
    independent reference-KL term remains active. All-zero groups have zero PG.
    """

    if raw_rewards.ndim not in (1, 2):
        raise ValueError("Raw training rewards must have shape [rows] or [rows, tokens].")
    truncated = termination_flags(non_tensor_batch, expected_rows=int(raw_rewards.shape[0]))
    effective = raw_rewards.clone()
    selected = torch.as_tensor(truncated, dtype=torch.bool, device=raw_rewards.device)
    if enabled:
        effective[selected] = 0.0
    raw_scores = raw_rewards if raw_rewards.ndim == 1 else raw_rewards.sum(dim=-1)
    effective_scores = effective if effective.ndim == 1 else effective.sum(dim=-1)
    rows = int(raw_rewards.shape[0])
    metrics = {
        "training_reward/raw_mean": float(raw_scores.mean().item()) if rows else 0.0,
        "training_reward/effective_mean": float(effective_scores.mean().item()) if rows else 0.0,
        "training_reward/truncated_fraction": float(truncated.mean()) if rows else 0.0,
        "training_reward/penalized_fraction": float(selected.float().mean().item()) if rows and enabled else 0.0,
        "training_reward/changed_fraction": float(raw_scores.ne(effective_scores).float().mean().item()) if rows else 0.0,
    }
    return effective, metrics
