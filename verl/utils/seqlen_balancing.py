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

import copy
import heapq
from itertools import chain
from typing import Optional, Tuple

import torch
from tensordict import TensorDict
from torch import distributed as dist

from ..protocol import DataProto


class Set:
    def __init__(self) -> None:
        self.sum = 0
        self.items = []

    def add(self, idx: int, val: int):
        self.items.append((idx, val))
        self.sum += val

    def merge(self, other):
        for idx, val in other.items:
            self.items.append((idx, val))
            self.sum += val

    def __lt__(self, other):
        if self.sum != other.sum:
            return self.sum < other.sum
        if len(self.items) != len(other.items):
            return len(self.items) < len(other.items)
        return self.items < other.items


class State:
    def __init__(self, items: list[Tuple[int, int]], k: int) -> None:
        self.k = k
        # sets should always be decreasing order
        self.sets = [Set() for _ in range(k)]
        assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
        for i, (idx, seqlen) in enumerate(items):
            self.sets[i].add(idx=idx, val=seqlen)
        self.sets = sorted(self.sets, reverse=True)

    def get_partitions(self):
        partitions = []
        for i in range(len(self.sets)):
            cur_partition = []
            for idx, _ in self.sets[i].items:
                cur_partition.append(idx)
            partitions.append(cur_partition)
        return partitions

    def merge(self, other):
        for i in range(self.k):
            self.sets[i].merge(other.sets[self.k - 1 - i])
        self.sets = sorted(self.sets, reverse=True)

    @property
    def spread(self) -> int:
        return self.sets[0].sum - self.sets[-1].sum

    def __lt__(self, other):
        # least heap, let the state with largest spread to be popped first,
        # if the spread is the same, let the state who has the largest set
        # to be popped first.
        if self.spread != other.spread:
            return self.spread > other.spread
        return self.sets[0] > other.sets[0]

    def __repr__(self) -> str:
        repr_str = "["
        for i in range(self.k):
            if i > 0:
                repr_str += ","
            repr_str += "{"
            for j, (_, seqlen) in enumerate(self.sets[i].items):
                if j > 0:
                    repr_str += ","
                repr_str += str(seqlen)
            repr_str += "}"
        repr_str += "]"
        return repr_str


def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq: list[State] = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def greedy_partition(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    bias = sum(seqlen_list) + 1 if equal_size else 0
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    partitions = [[] for _ in range(k_partitions)]
    partition_sums = [0 for _ in range(k_partitions)]
    for seqlen, i in sorted_seqlen:
        min_idx = None
        for j in range(k_partitions):
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:
                min_idx = j
        partitions[min_idx].append(i)
        partition_sums[min_idx] += seqlen
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool) -> list[list[int]]:
    """Get order of seq lengths to make partitions balanced, this is
    used in balacing sum of seqlength across dp ranks and microbatches.

    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            resulting number of partitions
        equal_size (bool):
            if True, number of items in each partitions must be equal.
            if False, only consider balancing the sum, each partition can have variable number of items

    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


def micro_batch_balanced_order(seqlen_list: list[int], world_size: int, unit_size: int) -> list[int]:
    """Row order for data-parallel dispatch that balances every micro-batch position across the ranks.

    The dispatched batch is cut into ``world_size`` contiguous chunks (one per rank), and every rank
    splits its chunk in order into micro-batches whose sizes are multiples of ``unit_size``.  Each
    micro-batch's forward/backward is synchronized across the ranks (FSDP collectives), so a pass costs
    the sum over micro-batch positions of the slowest rank; balancing only each rank's total, as
    ``get_seqlen_balanced_partitions`` does, leaves that cost unbalanced.  Here:

    1. the rows, sorted by length (longest first, ties by index), are dealt into
       ``len(seqlen_list) // unit_size`` units in boustrophedon order, so the units have nearly equal
       totals (for ``unit_size == 2``: the i-th longest row with the i-th shortest);
    2. the units, sorted by total (largest first), form waves of ``world_size`` similar units; each wave
       fills one slot of every rank, in reverse rank order on odd slots so that the ranks' totals stay
       balanced too.  The waves take the slots in folded order (largest, smallest, second largest, ...):
       with single-row units, an update mini-batch of several slots then spans the length range instead
       of one length quantile, and micro-batches of several slots stay balanced.

    Returns a permutation of the row indices; rank ``r`` receives
    ``order[r * rows_per_rank:(r + 1) * rows_per_rank]``.  Rows within a unit keep their index order.
    """

    count = len(seqlen_list)
    if world_size <= 0 or unit_size <= 0 or count == 0 or count % (world_size * unit_size) != 0:
        raise ValueError(
            f"cannot split {count} rows into {world_size} ranks with micro-batch units of {unit_size} rows"
        )
    unit_count = count // unit_size
    ranked = sorted(range(count), key=lambda index: (-seqlen_list[index], index))
    units: list[list[int]] = [[] for _ in range(unit_count)]
    for position, index in enumerate(ranked):
        lap, offset = divmod(position, unit_count)
        units[offset if lap % 2 == 0 else unit_count - 1 - offset].append(index)
    totals = [sum(seqlen_list[index] for index in unit) for unit in units]
    by_total = sorted(range(unit_count), key=lambda unit: (-totals[unit], unit))
    waves = [by_total[start : start + world_size] for start in range(0, unit_count, world_size)]
    slots = [waves[slot // 2] if slot % 2 == 0 else waves[len(waves) - 1 - slot // 2] for slot in range(len(waves))]
    order: list[int] = []
    for rank in range(world_size):
        for slot, wave in enumerate(slots):
            order.extend(sorted(units[wave[rank if slot % 2 == 0 else world_size - 1 - rank]]))
    return order


def minibatch_group_order(
    seqlen_list: list[int], group_ids: list, world_size: int, minibatches: int, unit_size: int
) -> list[int]:
    """Row order in which every optimizer mini-batch holds whole GRPO groups, each mini-batch balanced over the ranks.

    With ``minibatches`` optimizer steps per rollout step, each rank cuts its rows into ``minibatches``
    contiguous chunks and chunk ``j`` of every rank forms optimizer step ``j``.  The group-relative advantages
    of a GRPO group sum to zero only over the whole group, so a token-mean loss gives the tokens that all
    responses of a group share (the format prefix) no net update only if the group is in one optimizer step.
    Here:

    1. the groups (rows with the same id), largest token total first, go to the mini-batch with the smallest
       total that still has room, so every mini-batch holds ``len(groups) / minibatches`` whole groups;
    2. the rows of each mini-batch are ordered over the ranks by ``micro_batch_balanced_order``, and rank
       ``r`` receives its share of mini-batch 0, then of mini-batch 1, and so on.

    Returns a permutation of the row indices; rank ``r`` receives ``order[r * rows_per_rank:(r + 1) * rows_per_rank]``.
    """

    count = len(seqlen_list)
    if len(group_ids) != count:
        raise ValueError(f"got {len(group_ids)} group ids for {count} rows")
    if world_size <= 0 or minibatches <= 0 or count == 0 or count % (world_size * minibatches) != 0:
        raise ValueError(f"cannot split {count} rows into {minibatches} mini-batches over {world_size} ranks")
    groups: dict = {}
    for index, group in enumerate(group_ids):
        groups.setdefault(group, []).append(index)
    sizes = {len(rows) for rows in groups.values()}
    rows_per_set = count // minibatches
    if len(sizes) != 1 or rows_per_set % next(iter(sizes)) != 0:
        raise ValueError(
            f"groups of sizes {sorted(sizes)} cannot fill mini-batches of {rows_per_set} rows with whole groups"
        )
    groups_per_set = rows_per_set // next(iter(sizes))
    totals = {group: sum(seqlen_list[index] for index in rows) for group, rows in groups.items()}
    first = {group: rows[0] for group, rows in groups.items()}
    sets: list[list] = [[] for _ in range(minibatches)]
    set_totals = [0] * minibatches
    for group in sorted(groups, key=lambda item: (-totals[item], first[item])):
        target = min((j for j in range(minibatches) if len(sets[j]) < groups_per_set), key=lambda j: (set_totals[j], j))
        sets[target].append(group)
        set_totals[target] += totals[group]
    rows_per_rank_in_set = rows_per_set // world_size
    per_rank: list[list[int]] = [[] for _ in range(world_size)]
    for members in sets:
        rows = sorted(index for group in members for index in groups[group])
        local = micro_batch_balanced_order([seqlen_list[index] for index in rows], world_size, unit_size)
        for rank in range(world_size):
            chunk = local[rank * rows_per_rank_in_set : (rank + 1) * rows_per_rank_in_set]
            per_rank[rank].extend(rows[position] for position in chunk)
    return [index for rows in per_rank for index in rows]


def synchronized_token_cost(seqlen_list: list[int], order: list[int], world_size: int, micro_batch_size: int) -> int:
    """Tokens of the slowest rank summed over micro-batch positions (each position is synchronized)."""

    rows_per_rank = len(order) // world_size
    cost = 0
    for start in range(0, rows_per_rank, micro_batch_size):
        cost += max(
            sum(seqlen_list[index] for index in order[rank * rows_per_rank + start : rank * rows_per_rank + start + micro_batch_size])
            for rank in range(world_size)
        )
    return cost


def log_seqlen_unbalance(seqlen_list: list[int], partitions: list[list[int]], prefix: str) -> dict[str, float]:
    """
    Calculate and log metrics related to sequence length imbalance before and after partitioning.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        partitions (List[List[int]]): A list of partitions, where each inner list contains indices
                                      from seqlen_list assigned to that partition.
        prefix (str): A prefix to be added to each metric key in the returned dictionary.

    Returns:
        dict: A dictionary containing metrics related to sequence length imbalance.
    """
    # Get the number of partitions
    k_partition = len(partitions)
    # assert len(seqlen_list) % k_partition == 0
    batch_size = len(seqlen_list) // k_partition
    min_sum_seqlen = None
    max_sum_seqlen = None
    total_sum_seqlen = 0

    # Iterate over each batch of sequence lengths
    for offset in range(0, len(seqlen_list), batch_size):
        cur_sum_seqlen = sum(seqlen_list[offset : offset + batch_size])
        if min_sum_seqlen is None or cur_sum_seqlen < min_sum_seqlen:
            min_sum_seqlen = cur_sum_seqlen
        if max_sum_seqlen is None or cur_sum_seqlen > max_sum_seqlen:
            max_sum_seqlen = cur_sum_seqlen
        total_sum_seqlen += cur_sum_seqlen

    balanced_sum_seqlen_list = []
    for partition in partitions:
        cur_sum_seqlen_balanced = sum([seqlen_list[i] for i in partition])
        balanced_sum_seqlen_list.append(cur_sum_seqlen_balanced)

    min_sum_seqlen_balanced = min(balanced_sum_seqlen_list)
    max_sum_seqlen_balanced = max(balanced_sum_seqlen_list)

    return {
        f"{prefix}/min": min_sum_seqlen,
        f"{prefix}/max": max_sum_seqlen,
        f"{prefix}/minmax_diff": max_sum_seqlen - min_sum_seqlen,
        f"{prefix}/balanced_min": min_sum_seqlen_balanced,
        f"{prefix}/balanced_max": max_sum_seqlen_balanced,
        f"{prefix}/mean": total_sum_seqlen / len(partitions),
    }


def ceildiv(a: float, b: float) -> float:
    return -(a // -b)


def rearrange_micro_batches(
    batch: TensorDict, max_token_len: int, dp_group: Optional[dist.ProcessGroup] = None
) -> Tuple[list[TensorDict], list[list[int]]]:
    """Split the batch into a list of micro_batches, where the max_token_len is smaller than max_token_len
    and the number of valid tokens in each micro batch is well balanced.
    """
    # this is per local micro_bsz
    max_seq_len = batch["attention_mask"].shape[-1]
    assert max_token_len >= max_seq_len, (
        f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"
    )
    effective_seqlen = torch.sum(batch["attention_mask"], dim=-1)
    total_seqlen = effective_seqlen.sum().item()
    num_micro_batches = min(len(effective_seqlen), ceildiv(total_seqlen, max_token_len))
    if dist.is_initialized():
        num_micro_batches = torch.tensor([num_micro_batches], device="cuda")
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = num_micro_batches.cpu().item()

    effective_seqlen = effective_seqlen.tolist()
    assert num_micro_batches <= len(effective_seqlen)
    micro_bsz_idx = get_seqlen_balanced_partitions(effective_seqlen, num_micro_batches, equal_size=False)

    # Use the sum of squared sequence lengths to approximate attention computation workload
    def compute_workload(partition: list[int]) -> Tuple[int, int]:
        return (sum(effective_seqlen[idx] ** 2 for idx in partition), min(partition) if partition else 0)

    micro_bsz_idx.sort(key=compute_workload, reverse=True)

    micro_batches = []
    for partition in micro_bsz_idx:
        curr_micro_batch = [batch[idx] for idx in partition]
        micro_batches.append(torch.stack(curr_micro_batch))

    return micro_batches, micro_bsz_idx


def get_reverse_idx(idx_map: list[int]) -> list[int]:
    """
    Build the inverse of an index mapping.

    Args:
        idx_map (Sequence[int]): Sequence where idx_map[i] = j.

    Returns:
        List[int]: Inverse mapping list such that output[j] = i for each i.
    """
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map


def prepare_dynamic_batch(data: DataProto, max_token_len: int) -> tuple[list[DataProto], list[list[int]]]:
    """
    Prepare a batch for dynamic batching.

    Args:
        data (DataProto): The input data.
        max_token_len (int): The maximum token length for dynamic batching.

    Returns:
        Tuple[List[DataProto], List[List[int]]]: A tuple containing a list of DataProto objects
        and a list of index lists.
    """
    batch, batch_idx_list = rearrange_micro_batches(data.batch, max_token_len=max_token_len)
    micro_batches = []
    for i, batch_idx in enumerate(batch_idx_list):
        tensors = dict(batch[i])
        non_tensors = {key: value[batch_idx] for key, value in data.non_tensor_batch.items()}
        micro_batches.append(DataProto.from_dict(tensors, non_tensors))

    return micro_batches, batch_idx_list


def restore_dynamic_batch(data: torch.Tensor, batch_idx_list: list[list[int]]) -> torch.Tensor:
    """
    Restore a batch from dynamic batching.

    Args:
        data (torch.Tensor): The input data.
        batch_idx_list (List[List[int]]): The list of index lists.

    Returns:
        torch.Tensor: The restored data.
    """
    indices = list(chain.from_iterable(batch_idx_list))
    revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
    return data[revert_indices]
