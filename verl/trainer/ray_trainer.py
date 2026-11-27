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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type, Union
import numpy as np
import ray
import torch
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.checkpoint.shards import STUDENT_SHARD_PREFIXES, check_rank_shards
from ..utils.rollout_dispatch import generate_rollout_samples, validate_batch_layout
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

from ..workers.reward import AutoRewardManager
from .config import MeSDTrainingConfig
from .data_loader import FixedResumeSampler

from .metrics import compute_length_metrics, reduce_metrics


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool
        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


class RayTrainerBase:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: MeSDTrainingConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: Union[DataLoader, StatefulDataLoader],
        val_dataloader: Optional[Union[DataLoader, StatefulDataLoader]],
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None
        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        if config.algorithm.adv_estimator != "grpo":
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")
        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")
        if (
            config.data.rollout_batch_size
            * config.worker.rollout.n
            % config.worker.actor.micro_batch_size_per_device_for_experience
            != 0
        ):
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )
        if config.worker.rollout.n <= 1:
            raise ValueError("GRPO needs `config.worker.rollout.n > 1`.")
        # Every DP_COMPUTE_PROTO call of a step must split evenly over all ranks (no padding of training rows).
        self.configured_world_size = int(config.trainer.nnodes) * int(config.trainer.n_gpus_per_node)
        self.batch_layout = validate_batch_layout(
            rollout_batch_size=config.data.rollout_batch_size,
            loader_batch_size=config.data.mini_rollout_batch_size or config.data.rollout_batch_size,
            rollout_n=config.worker.rollout.n,
            world_size=self.configured_world_size,
            global_batch_size=config.worker.actor.global_batch_size,
            micro_batch_size_for_experience=config.worker.actor.micro_batch_size_per_device_for_experience,
            micro_batch_size_for_update=config.worker.actor.micro_batch_size_per_device_for_update,
            ulysses_size=config.worker.actor.ulysses_size,
        )
        print(f"Batch layout over {self.configured_world_size} ranks: {self.batch_layout.to_dict()}")
        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs
        self.logical_data_resume = config.data.dataloader_resume_mode == "logical"
        if self.logical_data_resume:
            logical_sampler = self._get_logical_train_sampler()
            if config.data.mini_rollout_batch_size is not None:
                raise ValueError("Logical dataloader resume requires mini_rollout_batch_size=null.")
            if config.trainer.total_epochs != 1:
                raise ValueError("Logical dataloader resume currently supports exactly one epoch.")
            if logical_sampler.batch_size != config.data.rollout_batch_size:
                raise ValueError("Logical dataloader resume requires one loader batch per rollout step.")
            if self.training_steps > logical_sampler.total_batches:
                raise ValueError("Logical dataloader resume cannot run beyond its fixed one-epoch plan.")
        config.worker.actor.optim.training_steps = self.training_steps

        print(f"Total training steps: {self.training_steps}")

    def _get_logical_train_sampler(self) -> FixedResumeSampler:
        sampler = getattr(self.train_dataloader, "sampler", None)
        if not isinstance(sampler, FixedResumeSampler):
            raise TypeError(
                "Logical dataloader resume requires the ordinary DataLoader created with FixedResumeSampler."
            )
        return sampler

    @staticmethod
    def _atomic_torch_save(state: dict[str, Any], path: str) -> None:
        temporary_path = f"{path}.tmp-{uuid.uuid4().hex}"
        try:
            torch.save(state, temporary_path)
            os.replace(temporary_path, path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            self.wg_dicts.append(wg_dict)
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        if self.actor_rollout_ref_wg.world_size != self.configured_world_size:
            raise RuntimeError(
                f"The worker group has {self.actor_rollout_ref_wg.world_size} ranks, but trainer.nnodes * "
                f"trainer.n_gpus_per_node = {self.configured_world_size}; the batch layout was validated for the latter."
            )
        self.actor_rollout_ref_wg.init_model()

    def _verify_checkpoint(self, checkpoint_path: str) -> None:
        """Check on the driver that every rank's shards exist for the current world size."""

        check_rank_shards(
            os.path.join(checkpoint_path, "actor"), STUDENT_SHARD_PREFIXES, self.actor_rollout_ref_wg.world_size
        )

    def _save_extra_state(self, folder_path: str) -> None:
        """Hook for trainer-specific state; written before the checkpoint tracker."""

    def _load_extra_state(self, checkpoint_path: str) -> None:
        """Hook for trainer-specific state; called after model and data state are restored."""

    def _save_checkpoint(self) -> str:
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)
        if self.logical_data_resume:
            data_state_path = os.path.join(folder_path, "data_state.pt")
            data_state = self._get_logical_train_sampler().logical_state_dict(completed_global_step=self.global_step)
            self._atomic_torch_save(data_state, data_state_path)
        else:
            dataloader_path = os.path.join(folder_path, "dataloader.pt")
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_path)
        self._save_extra_state(folder_path)
        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)
        return folder_path

    def _load_checkpoint(self) -> Optional[str]:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None
        if load_checkpoint_path is None:
            return None
        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")
        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        # Fail on the driver before any rank enters a collective load with a missing or mismatched shard.
        self._verify_checkpoint(load_checkpoint_path)
        if self.logical_data_resume:
            data_state_path = os.path.join(load_checkpoint_path, "data_state.pt")
            if not os.path.isfile(data_state_path):
                raise FileNotFoundError(
                    "Logical dataloader checkpoint is missing data_state.pt. Legacy dataloader.pt queue snapshots are intentionally not loaded."
                )
            data_state = torch.load(data_state_path, map_location="cpu", weights_only=False)
            self._get_logical_train_sampler().restore_logical_state(data_state, checkpoint_step=self.global_step)
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if not self.logical_data_resume:
            dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
            if os.path.exists(dataloader_path):
                dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
            else:
                print(f"No dataloader state found at {dataloader_path}, will start from scratch.")
        self._load_extra_state(load_checkpoint_path)
        return load_checkpoint_path

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        sample_inputs, sample_outputs, sample_labels, sample_scores = ([], [], [], [])
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            if "raw_prompt" in test_batch.non_tensor_batch:
                test_gen_batch.non_tensor_batch["raw_prompt"] = test_batch.non_tensor_batch["raw_prompt"].copy()
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["image_min_pixels"] = self.config.data.image_min_pixels
            test_gen_batch.meta_info["image_max_pixels"] = self.config.data.image_max_pixels
            test_gen_batch.meta_info["video_min_pixels"] = self.config.data.video_min_pixels
            test_gen_batch.meta_info["video_max_pixels"] = self.config.data.video_max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps
            test_gen_batch.meta_info["video_max_frames"] = self.config.data.video_max_frames
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)
            val_reward_batch = test_batch.select(
                batch_keys=["responses", "response_mask"],
                non_tensor_batch_keys=[k for k in test_batch.non_tensor_batch if k != "multi_modal_data"],
            )
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(val_reward_batch))
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)
            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)
            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)
        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        if self.config.trainer.val_generations_to_log > 0 and sample_inputs:
            print("Sample prompt (with template):", sample_inputs[0])
            print("Sample response:", sample_outputs[0])
            print("Sample ground_truth:", sample_labels[0])
            print("Sample reward:", sample_scores[0])
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _balance_batch(
        self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen"
    ) -> torch.Tensor:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        return global_idx

    def _generate_rollout(self, gen_batch: DataProto, metrics: dict[str, Any]) -> DataProto:
        """Generate ``rollout.n`` samples per prompt with an even split over all rollout ranks.

        When the prompt batch is smaller than (or not a multiple of) the
        number of ranks, prompts are dispatched at sample level: each prompt is
        repeated ``n`` times in vLLM's own sample order and every rank draws
        one sample per row with vLLM's per-sample seeds.  The output has the
        same rows, order and size as prompt-level generation; no padding is
        added (see ``verl.utils.rollout_dispatch``).
        """

        output, plan = generate_rollout_samples(
            self.actor_rollout_ref_wg.generate_sequences,
            gen_batch,
            samples_per_prompt=self.config.worker.rollout.n,
            world_size=self.actor_rollout_ref_wg.world_size,
        )
        metrics["rollout/sample_level_dispatch"] = float(plan.expanded)
        metrics["rollout/rows_per_rank"] = float(plan.rows_per_rank)
        return output

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        num_try_make_batch = 0
        print("Start generating batch...")
        timing_raw = {}
        total_gen_time = 0.0
        total_reward_time = 0.0
        while True:
            num_try_make_batch += 1
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                if self.logical_data_resume:
                    raise RuntimeError("Logical dataloader plan was exhausted before training completed.")
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)
            meta_info = {
                "image_min_pixels": self.config.data.image_min_pixels,
                "image_max_pixels": self.config.data.image_max_pixels,
                "video_min_pixels": self.config.data.video_min_pixels,
                "video_max_pixels": self.config.data.video_max_pixels,
                "video_fps": self.config.data.video_fps,
                "video_max_frames": self.config.data.video_max_frames,
                "global_step": self.global_step,
                "total_training_steps": self.training_steps,
                "step_percent": self.global_step / max(self.training_steps, 1),
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=[
                    "raw_prompt_ids",
                    "multi_modal_data",
                ],
                meta_info_keys=[
                    "image_min_pixels",
                    "image_max_pixels",
                    "video_min_pixels",
                    "video_max_pixels",
                    "video_fps",
                    "video_max_frames",
                ],
            )
            if "raw_prompt" in new_batch.non_tensor_batch:
                gen_batch.non_tensor_batch["raw_prompt"] = new_batch.non_tensor_batch["raw_prompt"].copy()
            t_start = time.time()
            gen_batch_output = self._generate_rollout(gen_batch, metrics)
            total_gen_time += time.time() - t_start
            new_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            new_batch = new_batch.union(gen_batch_output)
            batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"current_batch_size={current_batch_size!r} < rollout_batch_size={rollout_batch_size!r}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"num_try_make_batch={num_try_make_batch!r}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"num_try_make_batch={num_try_make_batch!r} >= max_try_make_batch={max_try_make_batch!r}. Generated too many. Please check your data."
                    )
            else:
                print(
                    f"current_batch_size={current_batch_size!r} >= rollout_batch_size={rollout_batch_size!r}. Finish generating."
                )
                return batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]
