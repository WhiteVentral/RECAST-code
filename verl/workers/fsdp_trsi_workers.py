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
"""FSDP worker for Video-t-RSI: actor + vLLM rollout + two frozen teacher snapshots.

Compared with the MeSD worker, the single EMA teacher is replaced by two
frozen, gradient-free FSDP modules with identical sharding:

* ``updated`` -- theta^+ (base teacher of the current version),
* ``anchor``  -- theta^- (Route-A reference snapshot).

Both are initialized from the same Open-o3-Video-SFT-7B weights as the actor,
so before the first snapshot event they equal the initialization model.
Snapshots are local-shard copies scheduled by the driver (parameters cast to
the teacher storage dtype, buffers exact); they are saved with every
checkpoint so a resumed run restores the matching teacher state.  No EMA is
applied.

With ``rsi.ref_kl_coef > 0`` (or ``rsi.ref_kl_monitor``) a third frozen module,
``reference``, keeps the initialization model for the KL penalty; it is never
written and is rebuilt from the model path on resume (not checkpointed).
"""

import os

import numpy as np
import torch
import torch.distributed as dist

from ..protocol import DataProto
from ..single_controller.base.decorator import Dispatch, register
from ..utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from ..utils.flops_counter import FlopsCounter
from ..utils.fsdp_utils import load_fsdp_model, offload_fsdp_model, offload_fsdp_optimizer
from ..utils.peak_memory import StagePeakMemory
from .fsdp_workers import FSDPWorker
from .trsi_snapshots import (
    SNAPSHOT_TARGETS,
    apply_snapshot,
    load_teacher_shards,
    reshard_fsdp_root,
    save_teacher_shards,
    state_fingerprint,
    sync_teacher_buffers_,
)


TRSI_SNAPSHOT_TARGETS = SNAPSHOT_TARGETS


class FSDPTRSIWorker(FSDPWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if not (self._has_actor and self._has_ref and self._has_rollout):
            raise RuntimeError("Video-t-RSI requires the colocated actor_rollout_ref worker role.")
        if self._use_ref_param_offload:
            raise ValueError("Video-t-RSI teacher snapshots require worker.ref.offload.offload_params=false.")
        if os.environ.get("FSDP_USE_FULL_PREC_IN_EVAL", "") == "1":
            raise ValueError(
                "FSDP_USE_FULL_PREC_IN_EVAL=1 makes FSDP recast the teachers' buffers to their bf16 storage dtype in "
                "eval forwards (rounding the rotary inv_freq); unset it for Video-t-RSI."
            )
        self._build_model_optimizer(
            model_config=self.config.actor.model,
            fsdp_config=self.config.actor.fsdp,
            optim_config=self.config.actor.optim,
            padding_free=self.config.actor.padding_free,
            role="actor",
        )
        # Snapshots must share the actor's flat-parameter layout.
        self.config.ref.fsdp.use_orig_params = self.config.actor.fsdp.use_orig_params
        self._build_model_optimizer(
            model_config=self.config.actor.model,
            fsdp_config=self.config.ref.fsdp,
            optim_config=None,
            padding_free=self.config.ref.padding_free,
            role="ref",
        )
        updated_module = self.ref_fsdp_module
        self._build_model_optimizer(
            model_config=self.config.actor.model,
            fsdp_config=self.config.ref.fsdp,
            optim_config=None,
            padding_free=self.config.ref.padding_free,
            role="ref",
        )
        self.anchor_fsdp_module = self.ref_fsdp_module
        self.reference_fsdp_module = None
        if getattr(self.config.actor, "trsi_reference", False):
            self._build_model_optimizer(
                model_config=self.config.actor.model,
                fsdp_config=self.config.ref.fsdp,
                optim_config=None,
                padding_free=self.config.ref.padding_free,
                role="ref",
            )
            self.reference_fsdp_module = self.ref_fsdp_module
        self.ref_fsdp_module = updated_module
        self.updated_fsdp_module = updated_module
        frozen = [self.updated_fsdp_module, self.anchor_fsdp_module]
        if self.reference_fsdp_module is not None:
            frozen.append(self.reference_fsdp_module)
        for module in frozen:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        # Loading in the ref storage dtype (bf16) also rounds floating buffers such as the rotary
        # inv_freq; start from the actor's exact buffers (snapshots keep them exact afterwards).
        sync_teacher_buffers_(self.fsdp_module, *frozen)

        from .actor.dp_trsi_actor import DataParallelTRSIActor

        self.actor = DataParallelTRSIActor(
            config=self.config.actor,
            actor_module=self.fsdp_module,
            actor_optimizer=self.optimizer,
            updated_teacher=self.updated_fsdp_module,
            anchor_teacher=self.anchor_fsdp_module,
        )
        self.reference_policy = None
        if self.reference_fsdp_module is not None:
            # The base PPO actor is abstract (update_policy is not implemented).
            # The concrete TRSI actor inherits the same no-grad log-probability
            # scorer and needs neither an optimizer nor teachers for reference use.
            self.reference_policy = DataParallelTRSIActor(
                config=self.config.actor, actor_module=self.reference_fsdp_module
            )
        self._build_rollout()
        self.flops_counter = FlopsCounter(self.model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.fsdp_module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.processor or self.tokenizer,
        )

    # ------------------------------------------------------------------
    # Teacher-state events
    # ------------------------------------------------------------------
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def trsi_snapshot(self, target: str):
        """Freeze a snapshot: ``anchor``/``updated``/``both`` <- actor, or anchor <- updated."""

        if target not in TRSI_SNAPSHOT_TARGETS:
            raise ValueError(f"Unknown Video-t-RSI snapshot target {target!r}.")
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        # Local-shard copies after resharding every root (fresh roots are lazily initialized first).
        copied = apply_snapshot(
            target, actor=self.fsdp_module, updated=self.updated_fsdp_module, anchor=self.anchor_fsdp_module
        )
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        return copied

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def trsi_state_fingerprint(self):
        """This rank's parameter/optimizer sums for the run records (no collectives, no state change)."""

        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        result = state_fingerprint(
            actor=self.fsdp_module,
            updated=self.updated_fsdp_module,
            anchor=self.anchor_fsdp_module,
            optimizer=self.optimizer,
        )
        result["lr"] = {"value": float(self.lr_scheduler.get_last_lr()[0])}
        result["rank"] = {"value": float(self.rank)}
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        return result

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_probs(self, data: DataProto):
        """Old log-probabilities; on Route C steps (``meta_info["trsi_candidates"] = k``, package v6) also the
        policy's top-k tokens at every response position with their log-probabilities, from the same forward."""

        candidates = int(data.meta_info.get("trsi_candidates") or 0)
        if not candidates:
            return super().compute_log_probs(data)
        assert self._has_actor
        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            log_probs, ids, values = self.actor.compute_log_prob_with_candidates(
                data, candidates=candidates, position_chunk=int(data.meta_info["trsi_candidate_chunk"])
            )
            output = DataProto.from_dict(
                tensors={"old_log_probs": log_probs, "trsi_candidate_ids": ids, "trsi_candidate_log_probs": values},
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)
        if self.world_size > 1:
            if dist.is_initialized():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            self.fsdp_module._handle.reshard(True)
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        return output.to("cpu")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_trsi_reference_log_probs(self, data: DataProto):
        """log pi_ref of the sampled response tokens under the student's prompt (KL penalty / health/ref_kl)."""

        if self.reference_policy is None:
            raise RuntimeError("The KL reference was not built (rsi.ref_kl_coef = 0 and rsi.ref_kl_monitor = false).")
        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())
        data.meta_info["temperature"] = self.config.rollout.temperature
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            log_probs = self.reference_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={"ref_log_probs": log_probs})
            output = self.ulysses_sharding_manager.postprocess_data(output)
        if self.world_size > 1:
            dist.barrier(device_ids=[torch.cuda.current_device()])
        reshard_fsdp_root(self.reference_fsdp_module)
        return output.to("cpu")

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_trsi_teacher_log_probs(self, data: DataProto):
        self._process_multi_modal_inputs(data)
        data = data.to(torch.cuda.current_device())
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            tensors = self.actor.compute_teacher_log_probs(data)
            output = DataProto.from_dict(tensors=tensors)
            output = self.ulysses_sharding_manager.postprocess_data(output)
        if self.world_size > 1:
            dist.barrier(device_ids=[torch.cuda.current_device()])
            for module in (self.updated_fsdp_module, self.anchor_fsdp_module):
                reshard_fsdp_root(module)
        torch.cuda.empty_cache()
        return output.to("cpu")

    # ------------------------------------------------------------------
    # Peak GPU memory per stage (step records: metrics.perf)
    # ------------------------------------------------------------------
    def _stage_peaks(self) -> StagePeakMemory:
        if getattr(self, "_peak_memory", None) is None:
            self._peak_memory = StagePeakMemory()
        return self._peak_memory

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_rollout_engine(self):
        super().release_rollout_engine()
        # Generation ends here (vLLM asleep); the next window covers old log-probs and teacher scoring.
        self._stage_peaks().start_window()
        self._route_c_experience_peak = None

    # ------------------------------------------------------------------
    # Route C continuations (package v6)
    # ------------------------------------------------------------------
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def prepare_route_c_engine(self):
        """Wake vLLM again for the Route C continuations, after teacher scoring and before the update.

        The actor has not been updated since this step's rollout, so the weights synced here are the ones that
        sampled the rollout (the continuations sample from the same policy).  The training-stage window so far
        (old log-probs, teacher scoring) is kept and joins the experience peak reported by ``update_actor``.
        """

        self._route_c_experience_peak = self._stage_peaks().start_window()
        super().prepare_rollout_engine()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_trsi_continuations(self, requests: DataProto):
        assert self._has_rollout
        requests.meta_info.update(
            {
                "eos_token_id": self.generation_config.eos_token_id
                if self.generation_config is not None
                else self.tokenizer.eos_token_id,
                "pad_token_id": self.generation_config.pad_token_id
                if self.generation_config is not None
                else self.tokenizer.pad_token_id,
            }
        )
        requests = self.rollout_sharding_manager.preprocess_data(requests)
        output = self.rollout.generate_continuations(requests)
        output = self.rollout_sharding_manager.postprocess_data(output)
        return output.to("cpu")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def release_route_c_engine(self):
        FSDPWorker.release_rollout_engine(self)  # vLLM asleep again; this generation is not a training stage
        self._stage_peaks().start_window()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        peaks = self._stage_peaks()
        experience_allocated, experience_reserved = peaks.start_window()
        before_route_c = getattr(self, "_route_c_experience_peak", None)
        if before_route_c is not None:  # old log-probs and teacher scoring ran before the Route C generation
            experience_allocated = max(experience_allocated, before_route_c[0])
            experience_reserved = max(experience_reserved, before_route_c[1])
            self._route_c_experience_peak = None
        output = super().update_actor(data)
        update_allocated, update_reserved = peaks.window_peak()
        process_allocated, process_reserved = peaks.process_peak()
        freed = self.rollout_sharding_manager.freed_bytes
        gib = float(1024**3)
        values = {
            # FSDPWorker's metrics with their meaning kept (the stage windows reset torch's peak counters):
            # the peak since the process started, minus the memory that vLLM's sleep frees.
            "perf/max_memory_allocated_gb": (process_allocated - freed) / gib,
            "perf/max_memory_reserved_gb": (process_reserved - freed) / gib,
            # This rollout step's training stages, vLLM asleep: old log-probs + teacher scoring, then the update.
            "perf/experience_peak_allocated_gb": experience_allocated / gib,
            "perf/update_peak_allocated_gb": update_allocated / gib,
            "perf/train_peak_reserved_gb": max(experience_reserved, update_reserved) / gib,
        }
        for key, value in values.items():
            output.non_tensor_batch[key] = np.array([value])
        return output

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _teachers(self) -> dict:
        return {"updated": self.updated_fsdp_module, "anchor": self.anchor_fsdp_module}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, path: str, save_model_only: bool = False):
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        self.checkpoint_manager.save_checkpoint(path, save_model_only)
        save_teacher_shards(path, self._teachers(), world_size=self.world_size, rank=self.rank)
        self.print_rank0(f"Saved Video-t-RSI anchor/updated teacher shards under {os.path.abspath(path)}.")
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path: str):
        if self._use_param_offload:
            load_fsdp_model(self.fsdp_module)
        self.checkpoint_manager.load_checkpoint(path)
        load_teacher_shards(
            path,
            self._teachers(),
            world_size=self.world_size,
            rank=self.rank,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        # Teacher checkpoints hold no non-persistent buffers; keep them equal to the student's.
        teachers = [self.updated_fsdp_module, self.anchor_fsdp_module]
        if self.reference_fsdp_module is not None:
            teachers.append(self.reference_fsdp_module)
        sync_teacher_buffers_(self.fsdp_module, *teachers)
        self.print_rank0(f"Loaded Video-t-RSI anchor/updated teacher shards from {os.path.abspath(path)}.")
        dist.barrier()
        if self._use_param_offload:
            offload_fsdp_model(self.fsdp_module)
        if self._use_optimizer_offload:
            offload_fsdp_optimizer(self.optimizer)


__all__ = ["FSDPTRSIWorker", "TRSI_SNAPSHOT_TARGETS"]
