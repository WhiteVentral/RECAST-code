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
FSDP worker for the Direct RKL actor, EMA teacher, and vLLM rollout.
"""

import os
from datetime import timedelta
from typing import Literal, Optional, cast

import numpy as np
import psutil
import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from codetiming import Timer
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    GenerationConfig,
    PreTrainedModel,
)
from transformers.modeling_utils import no_init_weights

from ..models.monkey_patch import apply_ulysses_patch
from ..protocol import DataProto
from ..single_controller.base import Worker
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.dataset import process_image, process_video
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_fn,
    load_fsdp_model,
    load_fsdp_optimizer,
    offload_fsdp_model,
    offload_fsdp_optimizer,
)
from ..utils.model_utils import print_gpu_memory_usage, print_model_size
from ..utils.tokenizer import get_processor, get_tokenizer
from ..utils.torch_dtypes import PrecisionType
from ..utils.torch_functional import (
    AnyPrecisionAdamW,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from .config import ActorConfig, FSDPConfig, ModelConfig, OptimConfig, WorkerConfig
from .rollout.vllm_rollout_spmd import vLLMRollout
from .sharding_manager import FSDPVLLMShardingManager
from .sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager


def process_group_timeout() -> Optional[timedelta]:
    """Collective timeout from ``NCCL_TIMEOUT`` (milliseconds), which the launchers export.

    NCCL and PyTorch do not read ``NCCL_TIMEOUT`` themselves; without this the
    process group used PyTorch's default (10 minutes for NCCL), so e.g. a
    checkpoint barrier waiting for a slow shared-storage write could abort a
    32-rank run although the launcher asked for 30 minutes.
    """

    value = os.getenv("NCCL_TIMEOUT", "").strip()
    if not value:
        return None
    milliseconds = int(value)
    if milliseconds <= 0:
        raise ValueError(f"NCCL_TIMEOUT must be a positive number of milliseconds, got {value!r}.")
    return timedelta(milliseconds=milliseconds)


class FSDPWorker(Worker):
    def __init__(self, config: WorkerConfig, role: Literal["actor_rollout_ref"]):
        super().__init__()
        self.config = config
        if role != "actor_rollout_ref":
            raise ValueError("Direct RKL workers require role=actor_rollout_ref.")
        self.role = role
        self._cache = {}
        if not dist.is_initialized():
            timeout = process_group_timeout()
            if timeout is None:
                dist.init_process_group(backend="nccl")
            else:
                dist.init_process_group(backend="nccl", timeout=timeout)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        self._has_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._has_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._has_ref = self.role in ["ref", "actor_rollout_ref"]
        self._use_param_offload = False
        self._use_optimizer_offload = False
        self._use_ref_param_offload = False
        if self._has_actor:
            self._use_param_offload = self.config.actor.offload.offload_params
            self._use_optimizer_offload = self.config.actor.offload.offload_optimizer
            self._init_dist_mesh(self.config.actor, "actor")
        if self._has_ref:
            self._use_ref_param_offload = self.config.ref.offload.offload_params

    def _init_dist_mesh(self, config: ActorConfig, role: Literal["actor"]):
        world_size = dist.get_world_size()
        fsdp_size = config.fsdp.fsdp_size
        if fsdp_size <= 0 or fsdp_size >= world_size:
            self.device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
        else:
            self.device_mesh = init_device_mesh(
                "cuda", mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp")
            )
        if config.ulysses_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                "cuda",
                mesh_shape=(world_size // config.ulysses_size, config.ulysses_size),
                mesh_dim_names=("dp", "sp"),
            )
        else:
            self.ulysses_device_mesh = None
        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        if self.config.rollout.n > 1:
            config.global_batch_size *= self.config.rollout.n
            self.print_rank0(f"{role} will use global batch size {config.global_batch_size}.")
        config.global_batch_size_per_device = config.global_batch_size // (world_size // config.ulysses_size)
        if config.global_batch_size_per_device == 0:
            raise ValueError(f"{role} global batch size * ulysses size must be larger than num gpus.")
        if config.global_batch_size_per_device % config.micro_batch_size_per_device_for_update != 0:
            raise ValueError(f"{role} global batch size per device must be divisible by the micro batch size.")
        if (
            config.fsdp.enable_cpu_offload
            and config.global_batch_size_per_device != config.micro_batch_size_per_device_for_update
        ):
            raise ValueError(f"{role} cannot use FSDP's CPU offload when gradient accumulation is enabled.")

    def _build_model_optimizer(
        self,
        model_config: ModelConfig,
        fsdp_config: FSDPConfig,
        optim_config: Optional[OptimConfig],
        padding_free: bool,
        role: Literal["actor", "ref"],
    ) -> None:
        if role != "ref":
            self.tokenizer = get_tokenizer(
                model_config.tokenizer_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
            )
            self.processor = get_processor(
                model_config.tokenizer_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
            )
            self.model_config = AutoConfig.from_pretrained(
                model_config.model_path,
                trust_remote_code=model_config.trust_remote_code,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **model_config.override_config,
            )
            try:
                self.generation_config = GenerationConfig.from_pretrained(model_config.model_path)
            except Exception:
                self.generation_config = GenerationConfig.from_model_config(self.model_config)
            self.print_rank0(f"Model config: {self.model_config}")
        if padding_free:
            apply_ulysses_patch(self.model_config.model_type)
            self.print_rank0("Ulysses patch applied!")
        if fsdp_config.torch_dtype is None:
            torch_dtype = torch.float32 if role != "ref" else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(fsdp_config.torch_dtype)
        if type(self.model_config) in AutoModelForImageTextToText._model_mapping.keys():
            AutoClass = AutoModelForImageTextToText
        else:
            AutoClass = AutoModelForCausalLM
        if not fsdp_config.enable_rank0_init or self.device_mesh.get_local_rank("fsdp") == 0:
            model = AutoClass.from_pretrained(
                model_config.model_path,
                config=self.model_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                device_map="cpu" if fsdp_config.enable_rank0_init else "cuda",
                low_cpu_mem_usage=True,
                trust_remote_code=model_config.trust_remote_code,
            )
        else:
            with no_init_weights(), init_empty_weights():
                model = AutoClass.from_config(
                    self.model_config,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    trust_remote_code=model_config.trust_remote_code,
                )
        model = cast(PreTrainedModel, model)
        model.tie_weights()
        model = model.to(torch_dtype)
        if model_config.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if role == "ref":
            model.requires_grad_(False)
        if model_config.freeze_vision_tower:
            if hasattr(model, "model") and hasattr(model.model, "visual"):
                model.model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            elif hasattr(model, "visual"):
                model.visual.requires_grad_(False)
                fsdp_config.use_orig_params = True
                self.print_rank0("Vision tower is set to not trainable.")
            else:
                self.print_rank0("No vision tower found.")
        dist.barrier()
        print_model_size(model)
        print_gpu_memory_usage("After huggingface model init")
        mixed_precision = MixedPrecision(
            param_dtype=PrecisionType.to_dtype(fsdp_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(fsdp_config.mp_reduce_dtype),
            buffer_dtype=PrecisionType.to_dtype(fsdp_config.mp_buffer_dtype),
        )
        auto_wrap_policy = get_fsdp_wrap_policy(model)
        self.print_rank0(f"FSDP wrap policy: {auto_wrap_policy}.")
        if self.device_mesh.ndim == 2:
            if fsdp_config.enable_full_shard:
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            else:
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
        elif fsdp_config.enable_full_shard:
            sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        if fsdp_config.enable_cpu_offload:
            cpu_offload = CPUOffload(offload_params=True)
        else:
            cpu_offload = None
        if fsdp_config.enable_rank0_init:
            sync_module_states = True
            param_init_fn = get_init_fn(model, device="cuda") if self.rank != 0 else None
        else:
            sync_module_states = False
            param_init_fn = None
        fsdp_module = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            param_init_fn=param_init_fn,
            device_id=torch.cuda.current_device(),
            sync_module_states=sync_module_states,
            forward_prefetch=False,
            use_orig_params=fsdp_config.use_orig_params,
            device_mesh=self.device_mesh,
        )
        print_gpu_memory_usage("After FSDP module init")
        if role == "actor":
            self.fsdp_module = fsdp_module
            if optim_config.strategy == "adamw":
                self.optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                    fused=True,
                )
            elif optim_config.strategy == "adamw_bf16":
                self.optimizer = AnyPrecisionAdamW(
                    filter(lambda p: p.requires_grad, self.fsdp_module.parameters()),
                    lr=optim_config.lr,
                    betas=optim_config.betas,
                    weight_decay=optim_config.weight_decay,
                )
            else:
                raise NotImplementedError(f"Optimizer {optim_config.strategy} not supported.")
            if optim_config.lr_warmup_steps is not None:
                num_warmup_steps = optim_config.lr_warmup_steps
            else:
                num_warmup_steps = int(optim_config.lr_warmup_ratio * optim_config.training_steps)
            if optim_config.lr_scheduler_type == "constant":
                self.lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=self.optimizer, num_warmup_steps=num_warmup_steps
                )
            elif optim_config.lr_scheduler_type == "cosine":
                total_steps = optim_config.training_steps
                min_lr_ratio = optim_config.min_lr_ratio
                num_cycles = 0.5
                self.lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=self.optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(f"LR scheduler type {optim_config.lr_scheduler_type} is not supported")
            print_gpu_memory_usage("After optimizer init")
            if self._use_param_offload:
                offload_fsdp_model(self.fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")
            if self._use_optimizer_offload:
                offload_fsdp_optimizer(optimizer=self.optimizer)
                print_gpu_memory_usage(f"After offload {role} optimizer during init")
        else:
            self.ref_fsdp_module = fsdp_module
            if self._use_ref_param_offload:
                offload_fsdp_model(self.ref_fsdp_module)
                print_gpu_memory_usage(f"After offload {role} model during init")

    def _build_rollout(self) -> None:
        tp_size = self.config.rollout.tensor_parallel_size
        dp_size = self.world_size // tp_size
        if self.world_size % tp_size != 0:
            raise ValueError(f"rollout world size {self.world_size} is not divisible by tp size {tp_size}.")
        rollout_device_mesh = init_device_mesh("cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        self.rollout = vLLMRollout(
            model_path=self.config.actor.model.model_path,
            config=self.config.rollout,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        self.rollout_sharding_manager = FSDPVLLMShardingManager(
            module=self.fsdp_module,
            inference_engine=self.rollout.inference_engine,
            device_mesh=rollout_device_mesh,
            use_param_offload=self._use_param_offload,
        )
        print_gpu_memory_usage("After vllm init")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self._has_actor:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.actor.fsdp,
                optim_config=self.config.actor.optim,
                padding_free=self.config.actor.padding_free,
                role="actor",
            )
        if self._has_ref:
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.ref.fsdp,
                optim_config=None,
                padding_free=self.config.ref.padding_free,
                role="ref",
            )
        if self._has_actor:
            from .actor.dp_mesd_actor import DataParallelMeSDActor

            self.actor = DataParallelMeSDActor(
                config=self.config.actor,
                actor_module=self.fsdp_module,
                actor_optimizer=self.optimizer,
                teacher_module=self.ref_fsdp_module
                if getattr(self.config.actor, "freeze_teacher_model", False) and self._has_ref
                else None,
            )
        if self._has_rollout:
            self._build_rollout()
        if self._has_actor:
            self.flops_counter = FlopsCounter(self.model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.fsdp_module,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                processing_class=self.processor or self.tokenizer,
            )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, save_model_only: bool = False):
        assert self._has_actor
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        self.checkpoint_manager.save_checkpoint(path, save_model_only)
        if self._has_actor and self._has_ref and getattr(self.config.actor, "freeze_teacher_model", False):
            if self._use_ref_param_offload:
                load_fsdp_model(self.ref_fsdp_module)
            teacher_path = os.path.join(path, f"teacher_world_size_{self.world_size}_rank_{self.rank}.pt")
            teacher_state = get_model_state_dict(self.ref_fsdp_module, options=StateDictOptions(cpu_offload=True))
            self.print_rank0(f"Saving EMA teacher shards under {os.path.abspath(path)}.")
            torch.save(teacher_state, teacher_path)
            dist.barrier()
            if self._use_ref_param_offload:
                offload_fsdp_model(self.ref_fsdp_module)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        assert self._has_actor
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        self.checkpoint_manager.load_checkpoint(path)
        if self._has_actor and self._has_ref and getattr(self.config.actor, "freeze_teacher_model", False):
            teacher_path = os.path.join(path, f"teacher_world_size_{self.world_size}_rank_{self.rank}.pt")
            teacher_exists = torch.tensor(
                1 if os.path.isfile(teacher_path) else 0, device=torch.cuda.current_device(), dtype=torch.int32
            )
            dist.all_reduce(teacher_exists, op=dist.ReduceOp.MIN)
            if not bool(teacher_exists.item()):
                if getattr(self.config.actor, "mesd_enabled", False):
                    raise FileNotFoundError(
                        f"MeSD checkpoint is missing at least one EMA teacher shard required for an exact resume; this rank expected {teacher_path}"
                    )
                self.print_rank0(
                    "Frozen-teacher checkpoint was not found; initializing the teacher from the loaded actor."
                )
                self.actor._sync_teacher_from_actor()
            else:
                if self._use_ref_param_offload:
                    load_fsdp_model(self.ref_fsdp_module)
                teacher_state = torch.load(teacher_path, weights_only=False)
                set_model_state_dict(
                    self.ref_fsdp_module, model_state_dict=teacher_state, options=StateDictOptions(cpu_offload=True)
                )
                self.print_rank0(f"Loaded EMA teacher shards from {os.path.abspath(path)}.")
                dist.barrier()
                if self._use_ref_param_offload:
                    offload_fsdp_model(self.ref_fsdp_module)
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        if self._use_optimizer_offload:
            offload_fsdp_optimizer(self.optimizer)

    def _build_multi_modal_inputs(self, data: DataProto, uid_key: str, data_key: str, out_key: str):
        if data_key not in data.non_tensor_batch or uid_key not in data.non_tensor_batch:
            return
        cache_uid_key = f"{out_key}_uid"
        if cache_uid_key in self._cache and (not np.all(data.non_tensor_batch[uid_key] == self._cache[cache_uid_key])):
            self._cache.pop(cache_uid_key, None)
            self._cache.pop(out_key, None)
        if out_key not in self._cache:
            image_min_pixels = data.meta_info["image_min_pixels"]
            image_max_pixels = data.meta_info["image_max_pixels"]
            video_min_pixels = data.meta_info["video_min_pixels"]
            video_max_pixels = data.meta_info["video_max_pixels"]
            video_fps = data.meta_info["video_fps"]
            video_max_frames = data.meta_info["video_max_frames"]
            batch_multi_modal_inputs = []
            multi_modal_inputs_cache = {}
            for index, multi_modal_data in zip(data.non_tensor_batch[uid_key], data.non_tensor_batch[data_key]):
                if index not in multi_modal_inputs_cache:
                    images, videos = ([], [])
                    video_metadatas = None
                    if "preprocessed_video_path" in multi_modal_data:
                        pt_path = multi_modal_data["preprocessed_video_path"]
                        preprocessed_data = torch.load(pt_path, map_location="cpu", weights_only=False)
                        videos = [preprocessed_data["frames"]]
                        video_metadatas = [preprocessed_data["metadata"]]
                    elif "images" in multi_modal_data:
                        for image in multi_modal_data["images"]:
                            images.append(process_image(image, image_min_pixels, image_max_pixels))
                    elif "video" in multi_modal_data:
                        videos = multi_modal_data["video"]
                        video_metadatas = multi_modal_data.get("video_metadatas", None)
                    elif "videos" in multi_modal_data:
                        video_metadatas = []
                        for video in multi_modal_data["videos"]:
                            result = process_video(
                                video,
                                min_pixels=video_min_pixels,
                                max_pixels=video_max_pixels,
                                max_frames=video_max_frames,
                                video_fps=video_fps,
                                return_fps=True,
                            )
                            if isinstance(result, tuple) and len(result) == 2:
                                video_data, _ = result
                                if isinstance(video_data, tuple) and len(video_data) == 2:
                                    frames, metadata = video_data
                                    videos.append(frames)
                                    video_metadatas.append(metadata)
                                else:
                                    videos.append(video_data)
                                    video_metadatas = None
                            else:
                                videos.append(result)
                                video_metadatas = None
                    if len(images) != 0:
                        multi_modal_inputs = dict(self.processor.image_processor(images=images, return_tensors="pt"))
                    elif len(videos) != 0:
                        processor_kwargs = {
                            "videos": videos,
                            "return_tensors": "pt",
                            "do_resize": False,
                            "do_sample_frames": False,
                        }
                        if video_metadatas is not None and len(video_metadatas) > 0:
                            processor_kwargs["video_metadata"] = video_metadatas
                        if hasattr(self.processor, "video_processor") and self.processor.video_processor is not None:
                            multi_modal_inputs = dict(self.processor.video_processor(**processor_kwargs))
                        else:
                            processor_kwargs["images"] = None
                            multi_modal_inputs = dict(self.processor.image_processor(**processor_kwargs))
                    else:
                        multi_modal_inputs = {}
                    multi_modal_inputs_cache[index] = multi_modal_inputs
                batch_multi_modal_inputs.append(multi_modal_inputs_cache[index])
            self._cache[cache_uid_key] = data.non_tensor_batch[uid_key]
            self._cache[out_key] = np.array(batch_multi_modal_inputs, dtype=object)
        data.non_tensor_batch[out_key] = self._cache[out_key]

    def _process_multi_modal_inputs(self, data: DataProto):
        if "multi_modal_data" not in data.non_tensor_batch:
            return
        self._build_multi_modal_inputs(data, "uid", "multi_modal_data", "multi_modal_inputs")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        assert self._has_actor
        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        if self._use_optimizer_offload:
            load_fsdp_optimizer(optimizer=self.optimizer)
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu_actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / (promised_flops * self.world_size)
            )
            metrics["perf/max_memory_allocated_gb"] = (
                torch.cuda.max_memory_allocated() - self.rollout_sharding_manager.freed_bytes
            ) / 1024**3
            metrics["perf/max_memory_reserved_gb"] = (
                torch.cuda.max_memory_reserved() - self.rollout_sharding_manager.freed_bytes
            ) / 1024**3
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / 1024**3
            lr = self.lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.lr_scheduler.step()
            output = DataProto(
                non_tensor_batch={
                    key: np.array([value] if np.isscalar(value) else value) for key, value in metrics.items()
                }
            )
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        if self._use_optimizer_offload:
            offload_fsdp_optimizer(optimizer=self.optimizer)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_rollout_engine(self):
        self.rollout_sharding_manager.load_vllm_and_sync_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_rollout_engine(self):
        self.rollout_sharding_manager.offload_vllm()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        assert self._has_rollout
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        prompts = self.rollout_sharding_manager.preprocess_data(prompts)
        output = self.rollout.generate_sequences(prompts=prompts)
        output = self.rollout_sharding_manager.postprocess_data(output)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs(self, data: DataProto):
        assert self._has_actor
        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(
                tensors={"old_log_probs": output}, meta_info={"temperature": self.config.rollout.temperature}
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)
        if self.world_size > 1:
            if dist.is_initialized():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            self.fsdp_module._handle.reshard(True)
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        output = output.to("cpu")
        return output
