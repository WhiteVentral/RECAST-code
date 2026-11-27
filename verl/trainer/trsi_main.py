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
Entry point for Video-t-RSI training (token-level recursive teacher updates).

Usage:
    python3 -m verl.trainer.trsi_main config=configs/train_video_trsi_fast.yaml [overrides...]
"""

import json
import os

import ray

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_trsi_workers import FSDPTRSIWorker
from ..workers.reward import AutoRewardManager
from .data_loader import create_dataloader
from .mesd_main import _build_ray_runtime_env
from .ray_trainer import ResourcePoolManager, Role
from .trsi_config import VideoTRSITrainingConfig, load_trsi_config
from .trsi_trainer import RayVideoTRSITrainer


@ray.remote(num_cpus=1)
class VideoTRSIRunner:
    def run(self, config: VideoTRSITrainingConfig):
        print("[Video-t-RSI] Starting the training runner")
        print(json.dumps(config.to_dict(), indent=2))

        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.data.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        role_worker_mapping = {Role.ActorRolloutRef: ray.remote(FSDPTRSIWorker)}
        global_pool_id = "global_pool"
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        mapping = {Role.ActorRolloutRef: global_pool_id}
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        RemoteRewardManager = ray.remote(AutoRewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer) if config.data.val_files else None

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)
        trainer = RayVideoTRSITrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def _runner_options() -> dict:
    """Pin the training runner (dataloader, rewards bookkeeping, checkpoint metadata, step records) to one node.

    Set ``TRSI_DRIVER_NODE_IP`` to a Ray node address to keep the runner's CPU
    work and its writes on that node. Otherwise Ray may place it on any node.
    """

    node_ip = os.getenv("TRSI_DRIVER_NODE_IP", "").strip()
    if not node_ip:
        return {}
    resource = f"node:{node_ip}"
    if resource not in ray.cluster_resources():
        # An unknown resource would leave the runner pending forever; run unpinned instead.
        print(f"[Video-t-RSI] WARNING: Ray has no resource {resource!r}; the training runner is not pinned to it.")
        return {}
    return {"resources": {resource: 0.001}}


def main():
    training_config = load_trsi_config()
    if not ray.is_initialized():
        ray.init(runtime_env=_build_ray_runtime_env())
    runner = VideoTRSIRunner.options(**_runner_options()).remote()
    ray.get(runner.run.remote(training_config))
    if training_config.trainer.ray_timeline is not None:
        ray.timeline(filename=training_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
