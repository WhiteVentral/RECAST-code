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
"""Batch layout across data-parallel ranks, including sample-level rollout dispatch.

``Dispatch.DP_COMPUTE_PROTO`` splits a ``DataProto`` into ``world_size`` equal
chunks.  With 16 prompts per step and 32 ranks the prompt batch cannot be
split ("only support equal chunk. Got size of DataProto 16 and chunk 32").

Instead of padding with dummy prompts, the prompt batch is expanded to its
``rollout.n`` samples before generation: every prompt is repeated ``n`` times
(interleaved, which is the order in which vLLM returns the ``n`` samples of a
request) and every copy carries its sample index.  Each rank then generates
one sample per row with ``SamplingParams.n = 1``.  The rollout produces the
same ``rollout_batch_size * n`` rows in the same order as before, so the
effective batch, the GRPO groups (``uid``) and every downstream tensor are
unchanged and nothing has to be removed afterwards.

Sampling seeds: every request is seeded.  vLLM gives the ``n`` child requests
of a seeded request the seeds ``seed + j`` (``vllm/v1/engine/parallel_sampling.py``).
Repeating a prompt with ``n = 1`` and the *same* seed would make all copies
sample identical tokens, collapsing every GRPO group to zero advantage.
``per_request_sampling_params`` therefore reproduces vLLM's child seeds per row.

A seeded vLLM request draws its sampling noise from a generator seeded with its
seed, so the same seed replays the same noise at every decoding position.
With one fixed ``SamplingParams.seed`` every prompt of every step would use the
same ``n`` noise streams.  The Video-t-RSI driver therefore attaches a seed per
prompt (``ROLLOUT_SEED_KEY``, from ``rollout_request_seeds``): distinct within
a step, different for every step, and reproducible on resume.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from ..protocol import DataProto


ROLLOUT_SAMPLE_INDEX_KEY = "rollout_sample_index"
"""Non-tensor key holding the sample index ``j`` of each pre-expanded rollout row."""

ROLLOUT_N_META_KEY = "n"
"""meta_info key that the vLLM rollout applies to ``SamplingParams.n``."""

ROLLOUT_SEED_KEY = "rollout_seed"
"""Non-tensor key holding the sampling seed of each prompt row (sample ``j`` uses ``seed + j``)."""

_SEED_LIMIT = 2**31
"""Seeds stay below 2**31 (non-negative int32), including ``seed + j``."""


def rollout_request_seeds(base_seed: int, step: int, call: int, num_prompts: int, samples_per_prompt: int) -> np.ndarray:
    """Per-prompt sampling seeds of one generation call: ``offset + i * n`` for prompt ``i``.

    ``offset`` is drawn from ``numpy.random.SeedSequence([base_seed, step, call])``, so the seeds are a
    deterministic function of the run seed and the step (a resumed run draws the same ones) and differ
    between steps.  Within a call the children ``seed_i + j`` of all prompts are distinct.
    """

    num_prompts, n = int(num_prompts), int(samples_per_prompt)
    if num_prompts < 0 or n <= 0:
        raise ValueError(f"Invalid seed request: prompts={num_prompts}, samples_per_prompt={n}.")
    span = num_prompts * n
    if span >= _SEED_LIMIT:
        raise ValueError(f"{num_prompts} prompts x {n} samples do not fit the seed range.")
    entropy = [int(base_seed) % (2**32), int(step), int(call)]
    if min(entropy) < 0:
        raise ValueError(f"Seed inputs must be non-negative, got step={step}, call={call}.")
    draw = int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0])
    offset = draw % (_SEED_LIMIT - span)
    return offset + np.arange(num_prompts, dtype=np.int64) * n


@dataclass(frozen=True)
class RolloutDispatchPlan:
    """How one generation call is split over the data-parallel ranks."""

    num_prompts: int
    samples_per_prompt: int
    world_size: int
    expanded: bool
    """True: prompts are repeated ``n`` times and each row draws one sample."""
    rows_per_rank: int
    """Rows each rank receives for generation."""
    output_rows: int
    """Rows returned by generation (always ``num_prompts * samples_per_prompt``)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_rollout_dispatch(num_prompts: int, samples_per_prompt: int, world_size: int) -> RolloutDispatchPlan:
    """Choose prompt-level or sample-level dispatch for one generation call.

    Prompt-level dispatch (the original behaviour) is kept whenever the prompt
    batch divides evenly over the ranks.  Otherwise the call is dispatched at
    sample level, which needs ``num_prompts * samples_per_prompt`` to divide
    evenly.  No padding is ever introduced.
    """

    num_prompts, samples_per_prompt, world_size = int(num_prompts), int(samples_per_prompt), int(world_size)
    if num_prompts <= 0 or samples_per_prompt <= 0 or world_size <= 0:
        raise ValueError(
            f"Invalid rollout dispatch: prompts={num_prompts}, samples_per_prompt={samples_per_prompt}, "
            f"world_size={world_size}."
        )
    output_rows = num_prompts * samples_per_prompt
    if num_prompts % world_size == 0:
        return RolloutDispatchPlan(
            num_prompts=num_prompts,
            samples_per_prompt=samples_per_prompt,
            world_size=world_size,
            expanded=False,
            rows_per_rank=num_prompts // world_size,
            output_rows=output_rows,
        )
    if output_rows % world_size != 0:
        raise ValueError(
            f"{num_prompts} prompts x {samples_per_prompt} samples = {output_rows} rollout rows cannot be split "
            f"evenly over {world_size} ranks. Choose data.rollout_batch_size so that rollout_batch_size * "
            f"worker.rollout.n is a multiple of {world_size} (padding is deliberately not used because padded "
            "rows would change the GRPO groups and the effective batch)."
        )
    return RolloutDispatchPlan(
        num_prompts=num_prompts,
        samples_per_prompt=samples_per_prompt,
        world_size=world_size,
        expanded=True,
        rows_per_rank=output_rows // world_size,
        output_rows=output_rows,
    )


def expand_prompts_for_rollout(gen_batch: DataProto, samples_per_prompt: int) -> DataProto:
    """Repeat every prompt ``n`` times (interleaved) and tag each copy with its sample index."""

    n = int(samples_per_prompt)
    if n <= 0:
        raise ValueError("samples_per_prompt must be positive.")
    if ROLLOUT_SAMPLE_INDEX_KEY in gen_batch.non_tensor_batch:
        raise ValueError(f"The prompt batch already contains {ROLLOUT_SAMPLE_INDEX_KEY!r}; it was expanded twice.")
    expanded = gen_batch.repeat(repeat_times=n, interleave=True)
    # A new dict: the caller's meta_info must keep its own sampling settings.
    expanded.meta_info = {**gen_batch.meta_info, ROLLOUT_N_META_KEY: 1}
    expanded.non_tensor_batch[ROLLOUT_SAMPLE_INDEX_KEY] = np.tile(np.arange(n, dtype=np.int64), len(gen_batch))
    return expanded


def check_rollout_output(
    output: DataProto, plan: RolloutDispatchPlan, expected_prompt_ids: Optional[torch.Tensor] = None
) -> DataProto:
    """Validate generation output against the plan and restore the original meta_info contract.

    ``expected_prompt_ids`` (the prompt ``input_ids`` repeated ``n`` times,
    interleaved) are compared with the ``prompts`` tensor returned by the
    rollout, which proves that row ``k`` of the output belongs to prompt
    ``k // n`` before it is joined with the prompt-side batch.
    """

    if len(output) != plan.output_rows:
        raise RuntimeError(
            f"Generation returned {len(output)} rows; expected {plan.num_prompts} prompts x "
            f"{plan.samples_per_prompt} samples = {plan.output_rows} ({plan.to_dict()})."
        )
    if expected_prompt_ids is not None and output.batch is not None and "prompts" in output.batch.keys():
        returned = output.batch["prompts"]
        expected = expected_prompt_ids.to(device=returned.device, dtype=returned.dtype)
        if tuple(returned.shape) != tuple(expected.shape) or not torch.equal(returned, expected):
            raise RuntimeError(
                "Rollout rows are not aligned with their prompts after dispatch "
                f"(expanded={plan.expanded}); refusing to build GRPO groups from misaligned rows."
            )
    if plan.expanded:
        output.meta_info = {key: value for key, value in output.meta_info.items() if key != ROLLOUT_N_META_KEY}
    return output


def generate_rollout_samples(
    generate_fn: Callable[[DataProto], DataProto], gen_batch: DataProto, samples_per_prompt: int, world_size: int
) -> tuple[DataProto, RolloutDispatchPlan]:
    """Run ``generate_fn`` (a DP_COMPUTE_PROTO worker call) with an even split over ``world_size`` ranks.

    Returns ``num_prompts * samples_per_prompt`` rows ordered prompt by prompt
    (sample ``j`` of prompt ``i`` at row ``i * n + j``), exactly as
    prompt-level generation with ``SamplingParams.n = n`` returns them.
    """

    n = int(samples_per_prompt)
    plan = plan_rollout_dispatch(len(gen_batch), n, world_size)
    expected_prompt_ids = gen_batch.batch["input_ids"].repeat_interleave(n, dim=0)
    request = expand_prompts_for_rollout(gen_batch, n) if plan.expanded else gen_batch
    output = generate_fn(request)
    return check_rollout_output(output, plan, expected_prompt_ids=expected_prompt_ids), plan


def per_request_sampling_params(
    base: Any,
    sample_indices: Optional[Sequence[int]],
    num_requests: int,
    request_seeds: Optional[Sequence[int]] = None,
) -> Any:
    """Per-row vLLM ``SamplingParams`` (vLLM's own child seeding).

    Without per-prompt seeds, ``base`` is returned unchanged for ordinary
    prompt rows.  For pre-expanded rows each row gets a shallow copy of
    ``base`` (the copy vLLM makes for its own child requests) whose seed is
    ``seed + j`` (``seed`` = the row's entry of ``request_seeds``, else
    ``base.seed``); with no seed at all the copies sample from the global
    generator, as vLLM's unseeded children do.  With ``request_seeds`` and
    ordinary prompt rows, each prompt's copy carries its seed and vLLM derives
    the ``n`` child seeds ``seed + j`` itself, so both dispatch modes sample
    with the same noise per (prompt, sample).
    """

    seeds = None
    if request_seeds is not None:
        seeds = [int(seed) for seed in np.asarray(request_seeds).reshape(-1).tolist()]
        if len(seeds) != int(num_requests):
            raise ValueError(f"Got {len(seeds)} rollout seeds for {num_requests} requests.")
        if any(seed < 0 for seed in seeds):
            raise ValueError(f"Rollout seeds must be non-negative, got {seeds}.")
    if sample_indices is None:
        if seeds is None:
            return base
        params = []
        for seed in seeds:
            child = copy.copy(base)
            child.seed = seed
            params.append(child)
        return params
    indices = [int(index) for index in np.asarray(sample_indices).reshape(-1).tolist()]
    if len(indices) != int(num_requests):
        raise ValueError(f"Got {len(indices)} rollout sample indices for {num_requests} requests.")
    if int(getattr(base, "n", 1)) != 1:
        raise ValueError(
            "Pre-expanded rollout rows require SamplingParams.n == 1 (the driver sets meta_info['n'] = 1); "
            f"got n={getattr(base, 'n', None)}."
        )
    if any(index < 0 for index in indices):
        raise ValueError(f"Rollout sample indices must be non-negative, got {indices}.")
    base_seed = getattr(base, "seed", None)
    params = []
    for row, index in enumerate(indices):
        child = copy.copy(base)
        seed = seeds[row] if seeds is not None else base_seed
        if seed is not None:
            child.seed = int(seed) + index
        params.append(child)
    return params


@dataclass(frozen=True)
class BatchLayout:
    """Per-rank row counts of one training step (validated once at start-up)."""

    world_size: int
    rollout_rows: int
    rows_per_rank: int
    experience_micro_batches_per_rank: int
    update_rows: int
    update_rows_per_rank: int
    update_micro_batches_per_rank: int
    loader_dispatch: RolloutDispatchPlan

    @property
    def optimizer_steps_per_rollout_step(self) -> int:
        """Update mini-batches (one optimizer step each) per rollout step."""

        return self.rollout_rows // self.update_rows

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["optimizer_steps_per_rollout_step"] = self.optimizer_steps_per_rollout_step
        value["loader_dispatch"] = self.loader_dispatch.to_dict()
        return value


def validate_batch_layout(
    *,
    rollout_batch_size: int,
    loader_batch_size: int,
    rollout_n: int,
    world_size: int,
    global_batch_size: int,
    micro_batch_size_for_experience: int,
    micro_batch_size_for_update: int,
    ulysses_size: int = 1,
) -> BatchLayout:
    """Check that every DP_COMPUTE_PROTO call of a training step splits evenly on every rank.

    Calls of one step and their row counts: generation (``loader_batch_size``
    prompts or their samples), old log-probs / teacher scoring
    (``rollout_batch_size * n`` rows, split into micro-batches of
    ``micro_batch_size_for_experience``) and the actor update (mini-batches of
    ``global_batch_size * n`` rows, micro-batches of
    ``micro_batch_size_for_update``).  Identical per-rank micro-batch counts
    keep the FSDP collectives of all ranks aligned.
    """

    world_size = int(world_size)
    n = int(rollout_n)
    rollout_rows = int(rollout_batch_size) * n
    if world_size <= 0 or int(ulysses_size) <= 0 or world_size % int(ulysses_size) != 0:
        raise ValueError(f"world_size={world_size} must be a positive multiple of ulysses_size={ulysses_size}.")
    if rollout_rows % world_size != 0:
        raise ValueError(
            f"rollout_batch_size * rollout.n = {rollout_batch_size} * {n} = {rollout_rows} rows cannot be split "
            f"evenly over {world_size} ranks."
        )
    rows_per_rank = rollout_rows // world_size
    if rows_per_rank % int(micro_batch_size_for_experience) != 0:
        raise ValueError(
            f"Each rank receives {rows_per_rank} rows for log-prob / teacher scoring, which is not divisible by "
            f"micro_batch_size_per_device_for_experience={micro_batch_size_for_experience}."
        )
    dp_size = world_size // int(ulysses_size)
    update_rows = int(global_batch_size) * n
    if update_rows % dp_size != 0:
        raise ValueError(
            f"global_batch_size * rollout.n = {update_rows} rows cannot be split evenly over {dp_size} data-parallel ranks."
        )
    update_rows_per_rank = update_rows // dp_size
    if update_rows_per_rank % int(micro_batch_size_for_update) != 0:
        raise ValueError(
            f"Each rank receives {update_rows_per_rank} rows per update mini-batch, which is not divisible by "
            f"micro_batch_size_per_device_for_update={micro_batch_size_for_update}."
        )
    if int(ulysses_size) == 1 and rows_per_rank % update_rows_per_rank != 0:
        raise ValueError(
            f"Each rank's {rows_per_rank} rollout rows do not split into update mini-batches of "
            f"{update_rows_per_rank} rows."
        )
    loader_dispatch = plan_rollout_dispatch(loader_batch_size, n, world_size)
    return BatchLayout(
        world_size=world_size,
        rollout_rows=rollout_rows,
        rows_per_rank=rows_per_rank,
        experience_micro_batches_per_rank=rows_per_rank // int(micro_batch_size_for_experience),
        update_rows=update_rows,
        update_rows_per_rank=update_rows_per_rank,
        update_micro_batches_per_rank=update_rows_per_rank // int(micro_batch_size_for_update),
        loader_dispatch=loader_dispatch,
    )


__all__ = [
    "BatchLayout",
    "ROLLOUT_N_META_KEY",
    "ROLLOUT_SAMPLE_INDEX_KEY",
    "ROLLOUT_SEED_KEY",
    "RolloutDispatchPlan",
    "check_rollout_output",
    "expand_prompts_for_rollout",
    "generate_rollout_samples",
    "per_request_sampling_params",
    "plan_rollout_dispatch",
    "rollout_request_seeds",
    "validate_batch_layout",
]
