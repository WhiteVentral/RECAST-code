"""Model-side helpers for Video-t-RSI teacher scoring and snapshots.

* :func:`rmpad_response_logit_positions` locates, inside a padding-free
  (``unpad_input``) sequence, the hidden states that predict every valid
  response token.
* :func:`lm_head_logits_at` evaluates the output projection only at those
  positions and, by default, in float32.  The teacher distribution of Route A
  depends on *differences* of log probabilities between two snapshots; bf16
  output rounding of logits around magnitude 16-32 is 0.06-0.125, which would
  dominate the small changes produced by a task phase.
* :func:`copy_module_state_` copies the local parameter shards (cast to the
  destination storage dtype) and the buffers (exactly) between identically
  wrapped modules.  It is used for the frozen anchor / updated snapshots;
  under FSDP both modules hold matching flat-parameter shards, so a
  shard-local copy is a full copy.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn


__all__ = [
    "copy_module_buffers_",
    "copy_module_state_",
    "find_output_projection",
    "lm_head_logits_at",
    "padded_response_logit_positions",
    "rmpad_response_logit_positions",
]


def rmpad_response_logit_positions(
    attention_mask: torch.Tensor,
    response_length: int,
    response_lengths: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    """Indices into the unpadded token axis whose logits predict response tokens.

    The padded layout is ``[left-padded prompt | right-padded response]`` of
    width ``S``.  Response token ``t`` of row ``i`` is predicted by padded
    position ``S - R - 1 + t``.  ``unpad_input`` keeps the non-zero positions of
    the flattened attention mask in order, so the unpadded index of a flat
    position is its rank among those positions.

    Returns:
        ``(positions, counts)`` where ``positions`` is a 1-D LongTensor in
        row-major ``(i, t)`` order and ``counts[i] == response_lengths[i]``.
    """

    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, seqlen].")
    batch, seqlen = attention_mask.shape
    response_length = int(response_length)
    if response_length <= 0 or response_length >= seqlen:
        raise ValueError("response_length must be positive and shorter than the sequence.")
    flat = attention_mask.reshape(-1).to(torch.bool)
    rank = torch.cumsum(flat.to(torch.long), dim=0) - 1
    first = seqlen - response_length - 1
    pieces = []
    counts = []
    for row in range(batch):
        length = int(response_lengths[row])
        if length < 0 or length > response_length:
            raise ValueError(f"Row {row} response length {length} is outside [0, {response_length}].")
        columns = torch.arange(first, first + length, device=attention_mask.device)
        flat_index = row * seqlen + columns
        if length and not bool(flat[flat_index].all().item()):
            raise ValueError(
                f"Row {row}: positions predicting its {length} response tokens are not all attended; "
                "the prompt must be left-padded and the response right-padded."
            )
        pieces.append(rank[flat_index])
        counts.append(length)
    positions = torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.long, device=attention_mask.device)
    return positions, counts


def padded_response_logit_positions(seqlen: int, response_length: int, device=None) -> torch.Tensor:
    """Positions ``[S - R - 1, S - 1)`` of a padded ``[B, S]`` batch."""

    return torch.arange(int(seqlen) - int(response_length) - 1, int(seqlen) - 1, device=device)


def find_output_projection(model: nn.Module) -> nn.Linear:
    """Return the vocabulary projection of a (possibly FSDP-wrapped) causal LM."""

    candidate = model
    for _ in range(4):
        getter = getattr(candidate, "get_output_embeddings", None)
        if callable(getter):
            head = getter()
            if isinstance(head, nn.Linear):
                return head
        inner = getattr(candidate, "module", None)
        if isinstance(inner, nn.Module) and inner is not candidate:
            candidate = inner
            continue
        break
    head = getattr(candidate, "lm_head", None)
    if isinstance(head, nn.Linear):
        return head
    raise TypeError(f"Cannot find a linear output projection on {type(model).__name__}.")


@contextmanager
def lm_head_logits_at(
    model: nn.Module,
    positions: torch.Tensor,
    *,
    fp32: bool = True,
    vocab_chunk: int = 16384,
) -> Iterator[None]:
    """Restrict the next forward's output projection to ``positions`` (sequence dim 1).

    A forward pre-hook selects the hidden states before the projection, so
    ``output.logits`` has shape ``[batch, len(positions), vocab]``.  With
    ``fp32=True`` a forward hook recomputes the projection in float32 from the
    same (unsharded) weight, in vocabulary chunks to bound the transient copy.
    Intended for gradient-free teacher scoring.
    """

    head = find_output_projection(model)
    if int(vocab_chunk) <= 0:
        raise ValueError("vocab_chunk must be positive.")

    def select_positions(module, args):
        hidden = args[0]
        index = positions.to(device=hidden.device, dtype=torch.long)
        return (hidden.index_select(1, index),) + tuple(args[1:])

    def float32_projection(module, args, output):
        if not fp32:
            return output
        hidden = args[0].float()
        weight = module.weight
        bias = getattr(module, "bias", None)
        logits = torch.empty(*hidden.shape[:-1], weight.size(0), dtype=torch.float32, device=hidden.device)
        for start in range(0, weight.size(0), int(vocab_chunk)):
            end = min(start + int(vocab_chunk), weight.size(0))
            block = torch.matmul(hidden, weight[start:end].float().t())
            if bias is not None:
                block = block + bias[start:end].float()
            logits[..., start:end] = block
        return logits

    handles = [head.register_forward_pre_hook(select_positions), head.register_forward_hook(float32_projection)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def copy_module_buffers_(destination: nn.Module, source: nn.Module) -> int:
    """Give ``destination`` exact copies of the buffers of ``source``; returns the element count.

    Floating buffers keep the *source* dtype instead of being rounded to the
    destination's storage dtype.  The teachers are stored in bfloat16 while the
    actor keeps float32 buffers; rounding e.g. the rotary ``inv_freq`` to
    bfloat16 (relative error up to 2^-9) shifts rotary angles by
    ``position * inv_freq * 2^-9``, i.e. by whole radians at positions in the
    thousands, so an unchanged snapshot would no longer compute the student's
    function.  FSDP does not shard buffers and casts them to the
    mixed-precision ``buffer_dtype`` (float32 here) anyway.
    """

    pairs = _matched_tensors("buffer", list(destination.buffers()), list(source.buffers()))
    copied = 0
    for target, value in pairs:
        if value.is_floating_point() and target.dtype != value.dtype:
            target.data = value.data.to(device=target.device).clone()
        else:
            target.data.copy_(value.data.to(device=target.device, dtype=target.dtype))
        copied += target.numel()
    return copied


def _matched_tensors(kind: str, destination: list, source: list) -> list:
    """Pair tensors by position after checking the whole layout (no partial copy on mismatch)."""

    if len(destination) != len(source):
        raise RuntimeError(f"Snapshot {kind} layouts differ: {len(destination)} != {len(source)} tensors.")
    for index, (target, value) in enumerate(zip(destination, source)):
        if tuple(target.shape) != tuple(value.shape):
            raise RuntimeError(f"Snapshot {kind} {index} shape {tuple(target.shape)} != {tuple(value.shape)}.")
    return list(zip(destination, source))


@torch.no_grad()
def copy_module_state_(destination: nn.Module, source: nn.Module) -> int:
    """Copy local parameter shards and buffers of ``source`` into ``destination``.

    Both modules must expose identically ordered and shaped parameters and
    buffers (for FSDP: built with the same wrap policy, sharding strategy,
    device mesh and ``use_orig_params``).  Parameters are cast to the
    destination storage dtype; under bf16 mixed precision this is the same
    rounding the student's own forward applies, so the snapshot computes the
    student's function.  Buffers are copied exactly (see
    :func:`copy_module_buffers_`).  Returns the number of copied elements.
    """

    params = _matched_tensors("parameter", list(destination.parameters()), list(source.parameters()))
    _matched_tensors("buffer", list(destination.buffers()), list(source.buffers()))
    copied = 0
    for target, value in params:
        target.data.copy_(value.data.to(device=target.device, dtype=target.dtype))
        copied += target.numel()
    return copied + copy_module_buffers_(destination, source)
