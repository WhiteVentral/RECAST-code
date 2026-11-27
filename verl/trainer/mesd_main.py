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
Entry point for Direct RKL (On-Policy Self-Distillation) training.

Usage:
    python3 -m verl.trainer.mesd_main config=path/to/config.yaml [overrides...]
"""

import json
import os

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import MeSDTrainingConfig
from .data_loader import create_dataloader
from .ray_trainer import ResourcePoolManager, Role
from .mesd_trainer import RayMeSDTrainer


_DISTRIBUTED_ENV_NAMES = (
    "NCCL_NET",
    "NCCL_NET_PLUGIN",
    "NCCL_IB_DISABLE",
    "NCCL_IB_HCA",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_TIMEOUT",
    "NCCL_IB_RETRY_CNT",
    "NCCL_IB_ADDR_FAMILY",
    "NCCL_SOCKET_IFNAME",
    "NCCL_SOCKET_FAMILY",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_TIMEOUT",
    "NCCL_CONF_FILE",
    "NCCL_OOB_NET_ENABLE",
    "NCCL_OOB_NET_IFNAME",
    "GLOO_SOCKET_IFNAME",
)


def _build_ray_runtime_env() -> dict[str, dict[str, str]]:
    actor_env = {
        "TOKENIZERS_PARALLELISM": "true",
        "VLLM_LOGGING_LEVEL": "WARN",
        # vLLM 0.11's external-launcher executor is already managed by the
        # Ray/FSDP ranks and requires nested V1 multiprocessing to be disabled.
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "TENSORBOARD_DIR": os.getenv("TENSORBOARD_DIR", "tensorboard_log"),
    }
    for name in _DISTRIBUTED_ENV_NAMES:
        value = os.getenv(name)
        if value is not None and value != "":
            actor_env[name] = value
    actor_env.setdefault("NCCL_DEBUG", "WARN")
    return {"env_vars": actor_env}


@ray.remote(num_cpus=1)
class MeSDRunner:
    """A runner for Direct RKL training."""

    def run(self, config: MeSDTrainingConfig):
        print("[Direct RKL] Starting Direct RKL Training Runner")
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

        ray_worker_group_cls = RayWorkerGroup
        role_worker_mapping = {
            Role.ActorRolloutRef: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRolloutRef: global_pool_id,
        }
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        RemoteRewardManager = ray.remote(AutoRewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer) if config.data.val_files else None

        train_dataloader, val_dataloader = create_dataloader(config.data, tokenizer, processor)

        trainer = RayMeSDTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.init_workers()
        trainer.fit()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(MeSDTrainingConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    training_config = OmegaConf.merge(default_config, cli_args)
    training_config: MeSDTrainingConfig = OmegaConf.to_object(training_config)
    training_config.deep_post_init()

    if not ray.is_initialized():
        ray.init(runtime_env=_build_ray_runtime_env())

    runner = MeSDRunner.remote()
    ray.get(runner.run.remote(training_config))

    if training_config.trainer.ray_timeline is not None:
        ray.timeline(filename=training_config.trainer.ray_timeline)


if __name__ == "__main__":
    main()
