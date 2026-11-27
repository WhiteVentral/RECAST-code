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
Direct RKL training configuration.
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple

from ..utils.py_functional import get_abs_path
from ..workers.config import WorkerConfig
from .mesd_config import MeSDConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    video_max_frames: int = 128
    max_prompt_length: int = 512
    max_response_length: int = 512
    truncation: str = "right"
    """Prompt truncation side: left | right | error (Open-o3/MeSD uses left)."""
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    override_chat_template: Optional[str] = None
    apply_chat_template_kwargs: Optional[dict] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    image_min_pixels: Optional[int] = None
    image_max_pixels: Optional[int] = None
    video_min_pixels: Optional[int] = None
    video_max_pixels: Optional[int] = None
    filter_overlong_prompts: bool = True
    dataset_format: str = "stgr"
    """Local STGR schema with the Open-o3 prompt/media adapter."""
    media_root: Optional[str] = None
    """Root containing the Open-o3 ``videos/`` source directories."""
    stgr_frame_cache_dir: Optional[str] = None
    """Optional directory containing offline Open-o3 multi-image frame caches."""
    stgr_frame_cache_required: bool = False
    """Require a complete offline cache and never decode STGR media during training."""
    dataloader_num_workers: int = 8
    dataloader_prefetch_factor: Optional[int] = 2
    dataloader_persistent_workers: bool = True
    dataloader_resume_mode: str = "stateful"
    """Data resume strategy: stateful | logical."""

    def post_init(self):
        self.image_dir = get_abs_path(self.image_dir, prompt="Image directory")
        self.override_chat_template = get_abs_path(self.override_chat_template, prompt="Chat template file")
        self.media_root = get_abs_path(self.media_root, prompt="STGR media root")
        self.stgr_frame_cache_dir = get_abs_path(self.stgr_frame_cache_dir, prompt="STGR frame cache directory")
        if self.stgr_frame_cache_required and self.stgr_frame_cache_dir is None:
            raise ValueError("data.stgr_frame_cache_required=true requires data.stgr_frame_cache_dir.")
        self.dataset_format = str(self.dataset_format or "stgr").lower()
        if self.dataset_format != "stgr":
            raise ValueError(f"Unsupported data.dataset_format={self.dataset_format!r}.")
        if self.truncation not in {"left", "right", "error"}:
            raise ValueError(f"Unsupported data.truncation={self.truncation!r}.")
        if self.dataloader_num_workers < 0:
            raise ValueError("data.dataloader_num_workers must be >= 0.")
        if self.dataloader_prefetch_factor is not None and self.dataloader_prefetch_factor <= 0:
            raise ValueError("data.dataloader_prefetch_factor must be > 0 when set.")
        self.dataloader_resume_mode = str(self.dataloader_resume_mode or "stateful").lower()
        if self.dataloader_resume_mode not in {"stateful", "logical"}:
            raise ValueError("data.dataloader_resume_mode must be either 'stateful' or 'logical'.")
        if self.image_min_pixels is None:
            self.image_min_pixels = self.min_pixels
        if self.image_max_pixels is None:
            self.image_max_pixels = self.max_pixels
        if self.video_min_pixels is None:
            self.video_min_pixels = self.min_pixels
        if self.video_max_pixels is None:
            self.video_max_pixels = self.max_pixels


@dataclass
class AlgorithmConfig:
    grpo_eps: float = 1e-4
    """Epsilon used by group-relative reward normalization (MeSD uses 1e-4)."""
    adv_estimator: str = "grpo"
    """Only group-relative GRPO is supported."""
    disable_kl: bool = True
    """disable reference model"""
    use_kl_loss: bool = False
    """use kl loss instead of kl in reward"""
    kl_coef: float = 0.0
    """kl coefficient"""
    online_filtering: bool = False
    """use online filtering"""
    pipeline_reward: bool = False
    """Pipeline reward computation: submit reward jobs during rollout generation to reduce waiting time."""

    def post_init(self):
        if self.grpo_eps <= 0:
            raise ValueError("algorithm.grpo_eps must be positive.")


@dataclass
class TrainerConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "easy_r1"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 2
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """Maximum attempts to assemble a rollout batch; -1 means no limit."""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # may be not exist
        self.load_checkpoint_path = get_abs_path(self.load_checkpoint_path, prompt="Model checkpoint")


@dataclass
class MeSDTrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    mesd: MeSDConfig = field(default_factory=MeSDConfig)

    def post_init(self):
        if self.data.dataset_format != "stgr":
            raise ValueError("This release trains on the STGR schema; set data.dataset_format=stgr.")
        if self.algorithm.adv_estimator != "grpo" or self.algorithm.grpo_eps != 1e-4:
            raise ValueError("Direct RKL requires GRPO with grpo_eps=1e-4.")
        if not self.algorithm.disable_kl or self.algorithm.use_kl_loss or self.algorithm.kl_coef != 0:
            raise ValueError("Direct RKL uses its own routed loss; reference-policy KL must remain disabled.")
        if self.algorithm.online_filtering or self.algorithm.pipeline_reward:
            raise ValueError("Direct RKL requires synchronous, unfiltered rewards for outcome routing.")
        if self.worker.actor.ppo_epochs != 1 or self.worker.actor.dynamic_batching:
            raise ValueError("Direct RKL requires ppo_epochs=1 and dynamic_batching=false.")
        if self.worker.ref.offload.offload_params:
            raise ValueError("EMA replay requires worker.ref.offload.offload_params=false.")
        if (self.trainer.nnodes, self.trainer.n_gpus_per_node) != (2, 8):
            raise ValueError("This release provides a two-node, eight-GPU-per-node training configuration.")
        self.worker.actor.mesd_enabled = True
        self.worker.actor.freeze_teacher_model = True
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
