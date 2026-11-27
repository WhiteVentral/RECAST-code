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
"""FSDP state operations behind the Video-t-RSI teacher snapshots.

``FSDPTRSIWorker`` calls these functions for snapshot events, teacher
initialization and teacher checkpoints.  They import neither Ray nor vLLM, so
the CPU FSDP integration test runs exactly this code.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict

from ..trainer.trsi.torch_utils import copy_module_buffers_, copy_module_state_


__all__ = [
    "SNAPSHOT_TARGETS",
    "apply_snapshot",
    "load_teacher_shards",
    "module_fingerprint",
    "optimizer_fingerprint",
    "reshard_fsdp_root",
    "save_teacher_shards",
    "state_fingerprint",
    "sync_teacher_buffers_",
    "teacher_shard_path",
]

SNAPSHOT_TARGETS = ("anchor", "updated", "anchor_from_updated", "updated_from_anchor", "both")
_SHARD_PREFIX = {"updated": "trsi_updated", "anchor": "trsi_anchor"}


def _sharded_world() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _unsharded_storage_allocated(handle) -> bool:
    """Whether the handle still holds its padded unsharded parameter (what FSDP's free would release).

    Outside a full-precision context FSDP gathers into and frees
    ``_full_param_padded``.  The full-precision buffer
    (``_full_prec_full_param_padded``) is used only inside
    ``summon_full_params``, which frees it on exit, or with
    ``FSDP_USE_FULL_PREC_IN_EVAL=1``, which the worker rejects.
    """

    padded = getattr(handle.flat_param, "_full_param_padded", None)
    return padded is not None and padded.untyped_storage().size() > 0


def reshard_fsdp_root(module: nn.Module) -> int:
    """Leave every FSDP unit of a root module sharded; returns the number of units resharded.

    After a gradient-free forward FSDP keeps the root unit unsharded (it
    expects a backward), and with SHARD_GRAD_OP / ZeRO-2 style strategies it
    keeps every nested unit unsharded too.  Two frozen teacher roots would
    then stay unsharded at the same time during teacher scoring.  Worse, a
    snapshot copy writes local shards only, and a unit whose unsharded
    parameters are still allocated skips the next all-gather, so it would keep
    computing with the old weights.  A root that has not run yet is lazily
    initialized first: the sharded-parameter attributes are created there, and
    buffers are cast to the mixed-precision buffer dtype.

    Only units whose unsharded storage is still allocated are resharded.  Each
    FSDP reshard that frees storage queues a CUDA event for the all-gather
    rate limiter (``limit_all_gathers``), and only an all-gather takes one
    back; resharding units that are already sharded would grow that queue on
    every call and stop the limiter from ever waiting.
    """

    if not _sharded_world():
        return 0
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp._runtime_utils import _lazy_init, _reshard
    from torch.distributed.fsdp._traversal_utils import _get_fsdp_states

    if not isinstance(module, FSDP):
        return 0
    _lazy_init(module, module)
    resharded = 0
    for state in _get_fsdp_states(module):
        handle = state._handle
        if handle is not None and handle.uses_sharded_strategy and _unsharded_storage_allocated(handle):
            _reshard(state, handle, True)  # FSDP's own post-forward reshard, freeing the unsharded storage
            resharded += 1
    return resharded


@torch.no_grad()
def apply_snapshot(target: str, *, actor: nn.Module, updated: nn.Module, anchor: nn.Module) -> dict[str, int]:
    """``anchor``/``updated``/``both`` <- student, ``anchor_from_updated``: anchor <- updated, or
    ``updated_from_anchor``: updated <- anchor (the teacher health gate's fallback).

    Returns the number of copied local elements per destination.
    """

    if target not in SNAPSHOT_TARGETS:
        raise ValueError(f"Unknown Video-t-RSI snapshot target {target!r}.")
    for module in (actor, updated, anchor):
        reshard_fsdp_root(module)
    copied = {}
    if target in ("anchor", "both"):
        copied["anchor"] = copy_module_state_(anchor, actor)
    if target in ("updated", "both"):
        copied["updated"] = copy_module_state_(updated, actor)
    if target == "anchor_from_updated":
        copied["anchor"] = copy_module_state_(anchor, updated)
    if target == "updated_from_anchor":
        copied["updated"] = copy_module_state_(updated, anchor)
    return copied


@torch.no_grad()
def sync_teacher_buffers_(actor: nn.Module, *teachers: nn.Module) -> None:
    """Give the teachers the student's exact buffers (they never train).

    Teachers loaded in bf16 storage have bf16-rounded rotary ``inv_freq``
    buffers, and teacher checkpoints contain no non-persistent buffers, so
    this runs after the teachers are built and after they are restored.
    """

    for teacher in teachers:
        copy_module_buffers_(teacher, actor)


def teacher_shard_path(path: str, name: str, *, world_size: int, rank: int) -> str:
    if name not in _SHARD_PREFIX:
        raise ValueError(f"Unknown Video-t-RSI teacher {name!r}.")
    return os.path.join(path, f"{_SHARD_PREFIX[name]}_world_size_{world_size}_rank_{rank}.pt")


def save_teacher_shards(path: str, teachers: dict[str, nn.Module], *, world_size: int, rank: int) -> None:
    """Write each teacher's local parameter shards (sharded state dict, as for the student)."""

    for name, module in teachers.items():
        state = get_model_state_dict(module, options=StateDictOptions(cpu_offload=True))
        torch.save(state, teacher_shard_path(path, name, world_size=world_size, rank=rank))


def load_teacher_shards(
    path: str, teachers: dict[str, nn.Module], *, world_size: int, rank: int, device: torch.device
) -> None:
    """Restore teacher shards written by :func:`save_teacher_shards`; every rank must find its files."""

    for name, module in teachers.items():
        shard = teacher_shard_path(path, name, world_size=world_size, rank=rank)
        exists = torch.tensor(1 if os.path.isfile(shard) else 0, device=device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(exists, op=dist.ReduceOp.MIN)
        if not bool(exists.item()):
            raise FileNotFoundError(
                f"Video-t-RSI checkpoint is missing {name} teacher shards required for an exact resume; "
                f"this rank expected {shard}."
            )
        state = torch.load(shard, weights_only=False)
        set_model_state_dict(module, model_state_dict=state, options=StateDictOptions(cpu_offload=True))


# ----------------------------------------------------------------------
# State fingerprints (run records; see verl/trainer/trsi_records.py)
# ----------------------------------------------------------------------
@torch.no_grad()
def module_fingerprint(
    module: nn.Module, *, cast_dtype: torch.dtype | None = None, reshard: bool = True
) -> dict[str, float]:
    """Float64 sums of this rank's local parameter shards and of the module's floating buffers.

    The sums are taken in a fixed order, so identical tensors give identical
    values; summing the per-rank values on the driver gives a fingerprint of
    the whole module.  ``cast_dtype`` first rounds the parameters (e.g. the
    student to the teachers' bf16 storage dtype), which is exactly what a
    snapshot copy does, so a fresh snapshot matches the cast student.  The
    module is resharded first (a no-op when it already is) so that its
    parameters are the local shards; pass ``reshard=False`` when the caller
    already did that.
    """

    if reshard:
        reshard_fsdp_root(module)
    sums, abs_sums, buffer_sums = [], [], []
    numel = 0
    for parameter in module.parameters():
        value = parameter.data if cast_dtype is None else parameter.data.to(cast_dtype)
        sums.append(value.sum(dtype=torch.float64))
        abs_sums.append(value.abs().sum(dtype=torch.float64))
        numel += value.numel()
    for buffer in module.buffers():
        if buffer.is_floating_point():
            buffer_sums.append(buffer.data.sum(dtype=torch.float64))
    return {
        "sum": _stacked_sum(sums),
        "abs_sum": _stacked_sum(abs_sums),
        "numel": float(numel),
        "buffer_sum": _stacked_sum(buffer_sums),
    }


def _stacked_sum(values: list) -> float:
    if not values:
        return 0.0
    device = values[0].device
    return float(torch.stack([value.to(device) for value in values]).sum().item())


@torch.no_grad()
def optimizer_fingerprint(optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    """Adam-state sums in parameter-group order (independent of the state dict's insertion order)."""

    result = {"step": 0.0, "exp_avg_sum": 0.0, "exp_avg_sq_sum": 0.0, "states": 0.0}
    if optimizer is None:
        return result
    exp_avg, exp_avg_sq = [], []
    step = 0.0
    states = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = optimizer.state.get(parameter)
            if not state:
                continue
            states += 1
            if "step" in state:
                value = state["step"]
                step = max(step, float(value.item() if torch.is_tensor(value) else value))
            if "exp_avg" in state:
                exp_avg.append(state["exp_avg"].sum(dtype=torch.float64))
            if "exp_avg_sq" in state:
                exp_avg_sq.append(state["exp_avg_sq"].sum(dtype=torch.float64))
    result.update(
        {
            "step": step,
            "exp_avg_sum": _stacked_sum(exp_avg),
            "exp_avg_sq_sum": _stacked_sum(exp_avg_sq),
            "states": float(states),
        }
    )
    return result


def state_fingerprint(
    *, actor: nn.Module, updated: nn.Module, anchor: nn.Module, optimizer: torch.optim.Optimizer | None
) -> dict[str, dict[str, float]]:
    """Per-rank fingerprints of the student, its teacher-dtype cast, both teachers and the optimizer.

    Issues no collective; each module is resharded at most once.
    """

    for module in (actor, updated, anchor):
        reshard_fsdp_root(module)
    teacher_dtype = next(updated.parameters()).dtype
    return {
        "student": module_fingerprint(actor, reshard=False),
        "student_as_teacher": module_fingerprint(actor, cast_dtype=teacher_dtype, reshard=False),
        "updated": module_fingerprint(updated, reshard=False),
        "anchor": module_fingerprint(anchor, reshard=False),
        "optimizer": optimizer_fingerprint(optimizer),
    }
