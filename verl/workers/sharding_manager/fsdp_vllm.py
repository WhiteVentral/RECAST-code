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

import inspect
import re
from typing import Iterable, Union

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel
from vllm import LLM
from vllm.distributed import parallel_state as vllm_ps

from ...protocol import DataProto, all_gather_data_proto
from ...utils.fsdp_utils import load_fsdp_model, offload_fsdp_model
from ...utils.model_utils import print_gpu_memory_usage
from .base import BaseShardingManager


class FSDPVLLMShardingManager(BaseShardingManager):
    def __init__(
        self,
        module: FSDP,
        inference_engine: LLM,
        device_mesh: DeviceMesh,
        use_param_offload: bool,
    ):
        self.module = module
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        self.use_param_offload = use_param_offload
        self.loaded = False

        self.world_size = dist.get_world_size()
        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()
        self.tp_group = vllm_ps.get_tensor_model_parallel_group().device_group

        # Record freed bytes to estimate memory usage correctly
        # https://github.com/vllm-project/vllm/pull/11743#issuecomment-2754338119
        self.freed_bytes = 0

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        gen_dp_rank = self.device_mesh["dp"].get_local_rank()
        torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(self.torch_random_states)

    def _rename_weight_keys(self, actor_weights: dict[str, Union[torch.Tensor, DTensor]], model: PreTrainedModel):
        # convert state dict keys: https://github.com/huggingface/transformers/pull/38385
        if not hasattr(model, "_checkpoint_conversion_mapping"):
            return actor_weights

        reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
        original_weights = {}
        for key, value in actor_weights.items():
            for pattern, replacement in reverse_key_mapping.items():
                replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
                replacement = re.sub(r"\(.*\)", "", replacement)
                key, n_replace = re.subn(pattern, replacement, key)
                # Early exit of the loop
                if n_replace > 0:
                    break

            original_weights[key] = value

        return original_weights

    def _make_weight_iterator(
        self, actor_weights: dict[str, Union[torch.Tensor, DTensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        for name, tensor in actor_weights.items():
            yield name, tensor.full_tensor() if self.world_size != 1 else tensor

    def _sync_weight_to_vllm(self):
        if self.use_param_offload:
            load_fsdp_model(self.module)

        actor_weights = get_model_state_dict(self.module)
        actor_weights = self._rename_weight_keys(actor_weights, self.module._fsdp_wrapped_module)
        print_gpu_memory_usage("After gather model weights in sharding manager")

        model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
        model.load_weights(self._make_weight_iterator(actor_weights))

        del actor_weights
        if self.use_param_offload:
            offload_fsdp_model(self.module)

        torch.cuda.empty_cache()
        print_gpu_memory_usage("After sync model weights in sharding manager")

    def load_vllm_and_sync_weights(self):
        """Load vllm engine and sync model weights to vllm model."""
        # NOTE: Basically, we only need `torch.cuda.empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        torch.cuda.empty_cache()
        assert self.loaded is False, "vllm engine has already been loaded"
        self.loaded = True

        print_gpu_memory_usage("Before vllm wake up in sharding manager")
        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["weights"])
        else:
            self.inference_engine.wake_up()

        self._sync_weight_to_vllm()

        if "tags" in inspect.signature(self.inference_engine.wake_up).parameters:
            self.inference_engine.wake_up(tags=["kv_cache"])

        print_gpu_memory_usage("After vllm wake up in sharding manager")
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def offload_vllm(self):
        """Offload vllm engine."""
        assert self.loaded is True, "vllm engine has not been loaded"
        self.loaded = False

        print_gpu_memory_usage("Before vllm offload in sharding manager")
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]
        self.inference_engine.sleep(level=1)
        free_bytes_after_sleep = torch.cuda.mem_get_info()[0]
        self.freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        print_gpu_memory_usage("After vllm offload in sharding manager")

        self.module.train()
        # The vision tower is trained: encoder outputs cached by this generation must not be reused after the
        # next weight update (and are released before empty_cache).
        self._drain_vllm_encoder_cache()
        torch.cuda.empty_cache()  # add empty cache after each compute

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def _drain_vllm_encoder_cache(self) -> None:
        """Drop vLLM's multimodal encoder cache after generation (best effort, no-op on other vLLM layouts).

        vLLM V1 keeps the vision encoder output of every image (keyed by a hash of its content) in
        ``GPUModelRunner.encoder_cache``; the scheduler's ``EncoderCacheManager`` shares an entry with later
        requests for the same image and evicts entries only under capacity pressure.  Neither ``sleep``,
        ``wake_up`` nor ``reset_prefix_cache`` clears it (vLLM 0.11.x).  Frames seen in an earlier step
        (another question on the same video) would then be encoded by an older vision tower, and the
        entries hold GPU memory across steps.  Generation has finished when this runs, so the runner's
        tensors and the manager's accounting are both reset to empty (as EasyR1 07cae10 does).
        """

        drained = 0
        try:
            # Resolve the runner's cache and the scheduler's accounting first and change them only together: a
            # runner cache emptied while the scheduler still lists its entries makes vLLM assert on the next
            # request for the same image.
            runner = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner
            engine_core = self.inference_engine.llm_engine.engine_core
            engine_core = getattr(engine_core, "engine_core", engine_core)  # the in-process client's core
            scheduler = engine_core.scheduler
            encoder_cache = runner.encoder_cache
            manager = scheduler.encoder_cache_manager
            missing = [
                name
                for name in ("cache_size", "num_free_slots", "num_freeable_slots", "cached", "freed")
                if not hasattr(manager, name)
            ]
            if not isinstance(encoder_cache, dict) or missing:
                raise AttributeError(f"unexpected encoder cache layout (runner cache {type(encoder_cache).__name__}, "
                                     f"manager lacks {missing})")
            if scheduler.has_unfinished_requests():
                return  # never the case after a completed generate call; keep runner and scheduler consistent
            drained = len(encoder_cache)
            encoder_cache.clear()
            manager.num_free_slots = manager.cache_size
            manager.num_freeable_slots = manager.cache_size
            manager.cached.clear()
            if hasattr(manager, "freeable"):
                manager.freeable.clear()
            manager.freed.clear()
        except Exception as error:  # noqa: BLE001 - a different vLLM layout must not stop training
            if not getattr(self, "_encoder_cache_warned", False):
                self._encoder_cache_warned = True
                print(f"[sharding manager] could not drain vLLM's encoder cache ({type(error).__name__}: {error}).")
            return
        if drained and not getattr(self, "_encoder_cache_reported", False):
            self._encoder_cache_reported = True
            print(
                f"[sharding manager] drained {drained} cached vision encoder outputs after generation "
                "(done after every generation, reported once)."
            )

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        all_gather_data_proto(data, size=self.tp_size, group=self.tp_group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size > 1:
            data = data.chunk(chunks=self.tp_size)[self.tp_rank]

        return data
