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

import hashlib
import os
from typing import Any, Optional, Sequence

import torch
from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import STGRDataset, collate_fn
from .config import DataConfig


class RepeatPaddedSampler(Sampler[int]):
    """Pad an epoch with its leading indices instead of dropping STGR rows.

    This mirrors DistributedSampler's default padding: 37,231 rows on 16 GPUs
    produce 2,327 full global prompt batches with one repeated row.
    """

    def __init__(self, sampler: Sampler[int], multiple: int):
        if multiple <= 0:
            raise ValueError("Sampler padding multiple must be positive.")
        self.sampler = sampler
        self.multiple = multiple

    def __len__(self) -> int:
        size = len(self.sampler)
        return ((size + self.multiple - 1) // self.multiple) * self.multiple

    def __iter__(self):
        indices = list(iter(self.sampler))
        if not indices:
            return iter(())
        padding = len(self) - len(indices)
        if padding:
            repeats = (padding + len(indices) - 1) // len(indices)
            indices.extend((indices * repeats)[:padding])
        return iter(indices)


class FixedResumeSampler(Sampler[int]):
    """Immutable epoch plan with a logical batch-aligned resume offset.

    DataLoader prefetch may advance its private sampler iterator, so that
    iterator position is deliberately not stored.  Checkpoints restore only a
    committed batch offset into this deterministic, fully materialized plan.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        indices: Sequence[int],
        *,
        batch_size: int,
        dataset_size: int,
        seed: int,
        shuffle: bool,
        dataset_sha256: str,
    ):
        if batch_size <= 0:
            raise ValueError("Logical resume batch size must be positive.")
        self._indices = tuple(int(index) for index in indices)
        if len(self._indices) % batch_size != 0:
            raise ValueError("Logical resume plan must contain complete batches.")
        self.batch_size = int(batch_size)
        self.dataset_size = int(dataset_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.dataset_sha256 = str(dataset_sha256)
        self.plan_sha256 = self._hash_indices(self._indices)
        self._start_batch = 0

    @staticmethod
    def _hash_indices(indices: Sequence[int]) -> str:
        digest = hashlib.sha256()
        for index in indices:
            digest.update(int(index).to_bytes(8, byteorder="little", signed=False))
        return digest.hexdigest()

    @property
    def total_batches(self) -> int:
        return len(self._indices) // self.batch_size

    @property
    def start_batch(self) -> int:
        return self._start_batch

    def set_start_batch(self, batch_index: int) -> None:
        batch_index = int(batch_index)
        if batch_index < 0 or batch_index > self.total_batches:
            raise ValueError(f"Logical resume batch {batch_index} is outside [0, {self.total_batches}].")
        self._start_batch = batch_index

    def __len__(self) -> int:
        # Keep the full epoch length stable so trainer.training_steps does not
        # change after a checkpoint offset is restored.
        return len(self._indices)

    def __iter__(self):
        start = self._start_batch * self.batch_size
        return iter(self._indices[start:])

    def logical_state_dict(self, *, completed_global_step: int) -> dict[str, Any]:
        completed_global_step = int(completed_global_step)
        if completed_global_step < 0 or completed_global_step > self.total_batches:
            raise ValueError(f"Completed step {completed_global_step} is outside the logical plan.")
        return {
            "schema_version": self.STATE_VERSION,
            "completed_global_step": completed_global_step,
            "next_batch_index": completed_global_step,
            "dataset_size": self.dataset_size,
            "padded_size": len(self._indices),
            "batch_size": self.batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "dataset_sha256": self.dataset_sha256,
            "plan_sha256": self.plan_sha256,
            "batch_plan": torch.tensor(self._indices, dtype=torch.int64),
        }

    def restore_logical_state(self, state: dict[str, Any], *, checkpoint_step: int) -> None:
        if int(state.get("schema_version", -1)) != self.STATE_VERSION:
            raise ValueError("Unsupported logical dataloader checkpoint schema.")

        expected = {
            "dataset_size": self.dataset_size,
            "padded_size": len(self._indices),
            "batch_size": self.batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "dataset_sha256": self.dataset_sha256,
            "plan_sha256": self.plan_sha256,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(
                    f"Logical dataloader checkpoint mismatch for {key}: saved={state.get(key)!r}, current={value!r}."
                )

        saved_plan = state.get("batch_plan")
        if not isinstance(saved_plan, torch.Tensor) or saved_plan.dtype != torch.int64:
            raise ValueError("Logical dataloader checkpoint has an invalid batch plan.")
        saved_indices = tuple(int(index) for index in saved_plan.tolist())
        if self._hash_indices(saved_indices) != state["plan_sha256"]:
            raise ValueError("Logical dataloader checkpoint batch plan is corrupt.")
        if saved_indices != self._indices:
            raise ValueError("Logical dataloader checkpoint batch plan changed.")

        completed_step = int(state.get("completed_global_step", -1))
        next_batch = int(state.get("next_batch_index", -1))
        if completed_step != int(checkpoint_step) or next_batch != completed_step:
            raise ValueError("Logical dataloader checkpoint step/cursor does not match its directory.")
        self.set_start_batch(next_batch)


def _sha256_file(data_path: str) -> str:
    source_path = data_path.split("@", 1)[0]
    if not os.path.isfile(source_path):
        raise ValueError(
            f"Logical dataloader resume currently requires a local training data file; got {data_path!r}."
        )
    digest = hashlib.sha256()
    with open(source_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]):
    train_dataset = STGRDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        video_max_frames=config.video_max_frames,
        max_prompt_length=config.max_prompt_length,
        truncation=config.truncation,
        image_min_pixels=config.image_min_pixels,
        image_max_pixels=config.image_max_pixels,
        video_min_pixels=config.video_min_pixels,
        video_max_pixels=config.video_max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        apply_chat_template_kwargs=config.apply_chat_template_kwargs,
        dataset_format=config.dataset_format,
        media_root=config.media_root,
        stgr_frame_cache_dir=config.stgr_frame_cache_dir,
        stgr_frame_cache_required=config.stgr_frame_cache_required,
    )
    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size
    sampler = RepeatPaddedSampler(sampler, multiple=train_batch_size)

    if config.dataloader_resume_mode == "logical":
        sampler = FixedResumeSampler(
            list(sampler),
            batch_size=train_batch_size,
            dataset_size=len(train_dataset),
            seed=config.seed,
            shuffle=config.shuffle,
            dataset_sha256=_sha256_file(config.train_files),
        )

    worker_kwargs = {"num_workers": config.dataloader_num_workers}
    if config.dataloader_num_workers > 0:
        worker_kwargs["persistent_workers"] = config.dataloader_persistent_workers
        if config.dataloader_prefetch_factor is not None:
            worker_kwargs["prefetch_factor"] = config.dataloader_prefetch_factor

    train_dataloader_cls = DataLoader if config.dataloader_resume_mode == "logical" else StatefulDataLoader
    train_dataloader = train_dataloader_cls(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
        **worker_kwargs,
    )

    if not config.val_files:
        print("Validation dataset disabled because data.val_files is empty.")
        return train_dataloader, None

    val_dataset = STGRDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        video_max_frames=config.video_max_frames,
        max_prompt_length=config.max_prompt_length,
        truncation=config.truncation,
        image_min_pixels=config.image_min_pixels,
        image_max_pixels=config.image_max_pixels,
        video_min_pixels=config.video_min_pixels,
        video_max_pixels=config.video_max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        apply_chat_template_kwargs=config.apply_chat_template_kwargs,
        dataset_format=config.dataset_format,
        media_root=config.media_root,
        stgr_frame_cache_dir=config.stgr_frame_cache_dir,
        stgr_frame_cache_required=config.stgr_frame_cache_required,
    )

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
        **worker_kwargs,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
