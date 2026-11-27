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
"""Driver-side completeness check of per-rank checkpoint shards.

Every rank loads ``<prefix>_world_size_<W>_rank_<r>.pt`` itself.  If one file
is missing, that rank raises while the others enter the collective of the
state-dict load and wait until the NCCL timeout.  Checking every expected file
on the driver first (the checkpoint directory is shared storage) turns this
into an immediate, explicit error, including the case of a checkpoint written
with a different world size (shards are not resharded on load).
"""

from __future__ import annotations

import os
import re
from typing import Iterable


STUDENT_SHARD_PREFIXES = ("model", "optim", "extra_state")
"""Shards written by FSDPCheckpointManager.save_checkpoint for every rank."""

_SHARD_PATTERN = re.compile(r"^(?P<prefix>.+)_world_size_(?P<world>\d+)_rank_(?P<rank>\d+)\.pt$")


def rank_shard_name(prefix: str, world_size: int, rank: int) -> str:
    return f"{prefix}_world_size_{int(world_size)}_rank_{int(rank)}.pt"


def saved_world_sizes(directory: str, prefix: str = "model") -> list[int]:
    """World sizes for which ``directory`` holds at least one ``prefix`` shard."""

    if not os.path.isdir(directory):
        return []
    sizes = set()
    for name in os.listdir(directory):
        match = _SHARD_PATTERN.match(name)
        if match and match.group("prefix") == prefix:
            sizes.add(int(match.group("world")))
    return sorted(sizes)


def missing_rank_shards(directory: str, prefixes: Iterable[str], world_size: int) -> list[str]:
    missing = []
    for prefix in prefixes:
        for rank in range(int(world_size)):
            path = os.path.join(directory, rank_shard_name(prefix, world_size, rank))
            if not os.path.isfile(path):
                missing.append(path)
    return missing


def check_rank_shards(directory: str, prefixes: Iterable[str], world_size: int) -> None:
    """Raise ``FileNotFoundError`` unless every rank's shard of every prefix exists."""

    prefixes = tuple(prefixes)
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Checkpoint directory not found: {directory}")
    missing = missing_rank_shards(directory, prefixes, world_size)
    if missing:
        present = saved_world_sizes(directory, prefixes[0])
        hint = (
            f" Shards in this directory were written for world size(s) {present}; a checkpoint can only be "
            f"resumed with the world size it was saved with."
            if present and int(world_size) not in present
            else ""
        )
        raise FileNotFoundError(
            f"Incomplete checkpoint {directory}: {len(missing)} of {len(prefixes) * int(world_size)} rank shard "
            f"files for world size {world_size} are missing (first: {missing[:3]}).{hint}"
        )


__all__ = [
    "STUDENT_SHARD_PREFIXES",
    "check_rank_shards",
    "missing_rank_shards",
    "rank_shard_name",
    "saved_world_sizes",
]
