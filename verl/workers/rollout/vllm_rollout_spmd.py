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
import os
from contextlib import contextmanager
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from transformers.video_utils import VideoMetadata
from vllm import LLM, RequestOutput, SamplingParams

from ...protocol import DataProto
from ...trainer.trsi.route_c import prompt_digest
from ...trainer.trsi.training_reward import completion_metadata, generation_budget, metadata_to_arrays
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.rollout_dispatch import ROLLOUT_SAMPLE_INDEX_KEY, ROLLOUT_SEED_KEY, per_request_sampling_params
from ...utils.torch_dtypes import PrecisionType
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray, list], repeats: int) -> Union[torch.Tensor, np.ndarray, list]:
    # repeat the elements, supports tensor, numpy array and list
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    elif isinstance(value, np.ndarray):
        return np.repeat(value, repeats, axis=0)
    elif isinstance(value, list):
        out = []
        for v in value:
            out.extend([v] * repeats)
        return out
    else:
        return np.repeat(value, repeats, axis=0)


_MIN_LOG_PROB = -1.0e4


def _engine_accepts(name: str) -> bool:
    """Whether the installed vLLM's ``EngineArgs`` (which ``LLM(**kwargs)`` builds) has the field ``name``."""

    try:
        import dataclasses

        from vllm.engine.arg_utils import EngineArgs

        return name in {field.name for field in dataclasses.fields(EngineArgs)}
    except Exception:  # noqa: BLE001 - an older or stand-in vLLM: do not pass the argument
        return False


def sampled_token_log_probs(output: Any) -> list[float]:
    """vLLM's log-probability of every sampled token of one ``CompletionOutput`` (``SamplingParams.logprobs=0``).

    ``output.logprobs`` holds one entry per generated token that maps token ids to ``Logprob`` objects (a list
    of dicts, or vLLM's flat container with the same indexing); with ``logprobs=0`` the entry holds the
    sampled token.
    """

    token_ids = list(output.token_ids)
    logprobs = getattr(output, "logprobs", None)
    if logprobs is None:
        raise RuntimeError("vLLM returned no logprobs; the rollout must sample with SamplingParams.logprobs=0.")
    if len(logprobs) != len(token_ids):
        raise RuntimeError(f"vLLM returned {len(logprobs)} logprob entries for {len(token_ids)} sampled tokens.")
    values = []
    for position, token in enumerate(token_ids):
        entry = logprobs[position]
        item = entry.get(token) if hasattr(entry, "get") else None
        if item is None:
            raise RuntimeError(f"vLLM's logprobs at position {position} do not contain the sampled token {token}.")
        value = float(getattr(item, "logprob", item))
        if value != value:
            raise RuntimeError(f"vLLM returned a NaN log-probability at position {position}.")
        # -inf (a token vLLM considers impossible) stays finite so that the mismatch metrics flag it.
        values.append(min(max(value, _MIN_LOG_PROB), 0.0))
    return values


def _processed_prompt(completion: Any) -> list[int]:
    """The prompt token ids vLLM generated from (placeholders expanded), from a ``RequestOutput``."""

    seen = getattr(completion, "prompt_token_ids", None)
    if seen is None:
        raise RuntimeError("vLLM returned no prompt token ids; the Route C inner loop compares them with the trainer's prompt.")
    return [int(value) for value in seen]


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output vision special tokens (image/video placeholders)
    if processor is None:
        return None

    logit_bias = {}
    if hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        logit_bias[image_token_id] = -100
    if hasattr(processor, "video_token"):
        video_token_id = processor.tokenizer.convert_tokens_to_ids(processor.video_token)
        logit_bias[video_token_id] = -100

    return logit_bias if logit_bias else None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any],
    image_min_pixels: int,
    image_max_pixels: int,
    video_min_pixels: int,
    video_max_pixels: int,
    video_max_frames: int,
    video_fps: float,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """
    Process multi-modal data and return (mm_data, mm_kwargs).
    - For images: ({"image": [processed_images]}, None)
    - For videos: ({"video": [processed_videos]}, {"do_sample_frames": False})

    Handles both:
    - "video" key: tensors already processed in dataset stage
    - "videos" key: paths that need processing (backward compatibility)
    """
    images, videos = [], []
    mm_kwargs = None

    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, image_min_pixels, image_max_pixels))

    # Check for "video" key (tensors from dataset) first
    if "video" in multi_modal_data:
        # Video tensors already processed in dataset stage
        video_tensors = multi_modal_data["video"]
        video_metadatas = multi_modal_data.get("video_metadatas", None)

        # Combine tensors with metadata for vLLM
        if video_metadatas is not None and len(video_metadatas) == len(video_tensors):
            for tensor, metadata in zip(video_tensors, video_metadatas):
                videos.append((tensor, metadata))
        else:
            # Fallback: create minimal metadata
            for tensor in video_tensors:
                num_frames = tensor.shape[0] if hasattr(tensor, 'shape') else len(tensor)
                videos.append((tensor, {"fps": 2.0, "frames_indices": list(range(num_frames)), "total_num_frames": num_frames}))

        if len(videos) > 0:
            mm_kwargs = {"do_sample_frames": False, "do_resize": False}
    elif "videos" in multi_modal_data:
        # Backward compatibility: process video paths
        for video in multi_modal_data["videos"]:
            # process_video returns ((frames, metadata), fps) when return_fps=True
            # vLLM Qwen3-VL expects each video to be (video_array, metadata) tuple
            # Use the same min/max pixels as training to ensure consistency
            processed, _ = process_video(
                video,
                min_pixels=video_min_pixels,
                max_pixels=video_max_pixels,
                max_frames=video_max_frames,
                video_fps=video_fps,
                return_fps=True
            )
            # Handle the case where processed is (frames, metadata) tuple
            if isinstance(processed, tuple) and len(processed) == 2:
                frames, metadata = processed
                # Keep the (frames, metadata) tuple for vLLM Qwen3-VL
                videos.append((frames, metadata))
            else:
                # Fallback: if no metadata, create a minimal one
                videos.append((processed, {"fps": 2.0, "frames_indices": list(range(len(processed))), "total_num_frames": len(processed)}))

        if len(videos) > 0:
            # do_sample_frames=False: we already sampled frames in fetch_video
            # do_resize=False: we already resized in fetch_video
            mm_kwargs = {"do_sample_frames": False, "do_resize": False}

    if len(images) != 0:
        return {"image": images}, None

    if len(videos) != 0:
        return {"video": videos}, mm_kwargs

    return None, None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)

        # Mix-policy configuration
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        engine_kwargs = {}
        if getattr(config, "disable_cascade_attn", False) and _engine_accepts("disable_cascade_attn"):
            # Cascade attention (shared-prefix kernels) gave wrong outputs on Ampere GPUs with the flash-attention
            # build of vLLM 0.10.2 / 0.11.0 (vLLM issue 22103); it only speeds up batches with a long common prefix.
            engine_kwargs["disable_cascade_attn"] = True
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images, "video": 1}

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)
        if config.calculate_log_probs:
            sampling_kwargs["logprobs"] = 0  # the sampled token's log-probability only

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def _request_for_row(
        self, raw_prompt_ids: Sequence[int], raw_prompt: Optional[str], multi_modal_data: dict[str, Any], meta_info: dict
    ) -> dict[str, Any]:
        """The vLLM input of one multimodal prompt row (shared by the rollout and the Route C continuations)."""

        if "preprocessed_video_path" in multi_modal_data:
            # Load the preprocessed .pt file locally.
            pt_path = multi_modal_data["preprocessed_video_path"]
            preprocessed_data = torch.load(pt_path, map_location="cpu", weights_only=False)
            video_tensors = [preprocessed_data["frames"]]
            video_metadatas_raw = [preprocessed_data["metadata"]]

            if raw_prompt is not None:
                item = {"prompt": raw_prompt}
            else:
                item = {"prompt_token_ids": list(raw_prompt_ids)}

            # Format for vLLM: list of (tensor, VideoMetadata) tuples
            videos = []
            for tensor, metadata in zip(video_tensors, video_metadatas_raw):
                if isinstance(metadata, dict):
                    metadata_obj = VideoMetadata(
                        total_num_frames=metadata.get("total_num_frames", tensor.shape[0] if hasattr(tensor, 'shape') else len(tensor)),
                        fps=metadata.get("fps"),
                        frames_indices=metadata.get("frames_indices"),
                        video_backend=metadata.get("video_backend"),
                        width=metadata.get("width"),
                        height=metadata.get("height"),
                        duration=metadata.get("duration"),
                    )
                else:
                    metadata_obj = metadata
                videos.append((tensor, metadata_obj))

            item["multi_modal_data"] = {"video": videos}
            item["mm_processor_kwargs"] = {"do_sample_frames": False, "do_resize": False}
        elif "video" in multi_modal_data:
            # Video tensors already processed in dataset stage
            # Pass text prompt to vLLM (required for finding <video> placeholder)
            if raw_prompt is not None:
                item = {"prompt": raw_prompt}  # Use the text prompt containing the <video> placeholder.
            else:
                item = {"prompt_token_ids": list(raw_prompt_ids)}  # Fallback.

            video_tensors = multi_modal_data["video"]
            video_metadatas = multi_modal_data.get("video_metadatas", None)

            # Format for vLLM: list of (tensor, metadata) tuples for Qwen3-VL
            # Convert dict metadata to VideoMetadata object
            if video_metadatas is not None and len(video_metadatas) == len(video_tensors):
                videos = []
                for tensor, metadata in zip(video_tensors, video_metadatas):
                    # Convert dict to VideoMetadata if needed
                    if isinstance(metadata, dict):
                        metadata_obj = VideoMetadata(
                            total_num_frames=metadata.get("total_num_frames", tensor.shape[0] if hasattr(tensor, 'shape') else len(tensor)),
                            fps=metadata.get("fps"),
                            frames_indices=metadata.get("frames_indices"),
                            video_backend=metadata.get("video_backend"),
                            width=metadata.get("width"),
                            height=metadata.get("height"),
                            duration=metadata.get("duration"),
                        )
                    else:
                        metadata_obj = metadata
                    videos.append((tensor, metadata_obj))
            else:
                # Fallback: create minimal metadata
                videos = []
                for tensor in video_tensors:
                    num_frames = tensor.shape[0] if hasattr(tensor, 'shape') else len(tensor)
                    metadata_obj = VideoMetadata(
                        total_num_frames=num_frames,
                        fps=2.0,
                        frames_indices=list(range(num_frames)),
                    )
                    videos.append((tensor, metadata_obj))

            item["multi_modal_data"] = {"video": videos}
            item["mm_processor_kwargs"] = {"do_sample_frames": False, "do_resize": False}
        else:
            # Backward compatibility: process video paths or images
            # For multimodal data (images/videos), use text prompts instead of token IDs.
            # This allows vLLM to locate the <image>/<video> placeholders.
            if raw_prompt is not None and ("images" in multi_modal_data or "videos" in multi_modal_data):
                item = {"prompt": raw_prompt}  # Use the text prompt containing <image>/<video> placeholders.
            else:
                item = {"prompt_token_ids": list(raw_prompt_ids)}  # Plain text or fallback.

            mm_data, mm_kwargs = _process_multi_modal_data(
                multi_modal_data,
                meta_info["image_min_pixels"],
                meta_info["image_max_pixels"],
                meta_info["video_min_pixels"],
                meta_info["video_max_pixels"],
                meta_info["video_max_frames"],
                meta_info["video_fps"],
            )
            if mm_data is not None:
                item["multi_modal_data"] = mm_data
                # Inject mm_processor_kwargs for video processing (e.g., do_sample_frames)
                if mm_kwargs is not None:
                    item["mm_processor_kwargs"] = mm_kwargs
        return item

    def _completion_termination(self, completion: Any, output: Any, params: Any, eos_token_ids: Any) -> dict[str, Any]:
        """Capture actual request settings while any sampling overrides are active."""

        requested = getattr(params, "max_tokens", None)
        seen = getattr(completion, "prompt_token_ids", None)
        engine = getattr(self.inference_engine, "llm_engine", None)
        model_config = getattr(engine, "model_config", None)
        model_limit = getattr(model_config, "max_model_len", None) or getattr(self.config, "max_model_len", None)
        if model_limit is None:
            prompt_limit = getattr(self.config, "prompt_length", None)
            if prompt_limit is not None and int(prompt_limit) > 0:
                model_limit = int(prompt_limit) + int(self.config.response_length)
        budget = generation_budget(
            requested, prompt_tokens=None if seen is None else len(seen), max_model_len=model_limit
        )
        return completion_metadata(
            output,
            max_tokens=budget,
            requested_max_tokens=requested,
            eos_token_ids=eos_token_ids,
            ignore_eos=bool(getattr(params, "ignore_eos", False)),
            stop_token_ids=getattr(params, "stop_token_ids", None),
        )

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_raw_prompt = non_tensor_batch.pop("raw_prompt", None)  # Extract the original text prompt.
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        batch_preprocessed_video = non_tensor_batch.pop("preprocessed_video", None)  # Not needed here; already used in the dataset stage.
        # Present when the driver dispatched at sample level (one sample per row, see utils/rollout_dispatch.py).
        batch_sample_indices = non_tensor_batch.pop(ROLLOUT_SAMPLE_INDEX_KEY, None)
        # Per-prompt sampling seeds chosen by the driver (rsi.rollout_seed_mode=per_step); else the fixed seed.
        batch_request_seeds = non_tensor_batch.pop(ROLLOUT_SEED_KEY, None)

        # Mix-policy: retrieve information about precollected trajectories.

        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            raw_prompts = batch_raw_prompt if batch_raw_prompt is not None else [None] * len(batch_raw_prompt_ids)
            vllm_inputs = [
                self._request_for_row(raw_prompt_ids, raw_prompt, multi_modal_data, prompts.meta_info)
                for raw_prompt_ids, raw_prompt, multi_modal_data in zip(
                    batch_raw_prompt_ids, raw_prompts, batch_multi_modal_data
                )
            ]
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            # Pre-expanded rows: one SamplingParams per row with vLLM's child seed (seed + sample index), so
            # copies of a prompt do not sample identical responses under the fixed rollout seed.
            request_sampling_params = per_request_sampling_params(
                self.sampling_params,
                batch_sample_indices,
                num_requests=len(vllm_inputs),
                request_seeds=batch_request_seeds,
            )
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs, sampling_params=request_sampling_params, use_tqdm=self.use_tqdm
            )
            if len(completions) != len(vllm_inputs):
                raise RuntimeError(f"vLLM returned {len(completions)} completions for {len(vllm_inputs)} requests.")
            params_by_request = (
                request_sampling_params
                if isinstance(request_sampling_params, (list, tuple))
                else [request_sampling_params] * len(completions)
            )
            termination_records = [
                self._completion_termination(completion, output, params, eos_token_id)
                for completion, params in zip(completions, params_by_request)
                for output in completion.outputs
            ]
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            if any(len(tokens) > int(self.config.response_length) for tokens in response_ids):
                raise RuntimeError("vLLM output exceeds the configured response tensor width; do not silently truncate it.")
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)
            rollout_log_probs = None
            if getattr(self.config, "calculate_log_probs", False):
                rollout_log_probs = [
                    sampled_token_log_probs(output) for completion in completions for output in completion.outputs
                ]
                rollout_log_probs = VF.pad_2d_list_to_length(
                    rollout_log_probs, 0.0, max_length=self.config.response_length
                ).to(device=input_ids.device, dtype=torch.float32)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

            # Mix-policy: replace the last online generation with an offline trajectory for samples with precollected data.
            # Keep all n online generations for samples without offline trajectories.

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        # Stop strings/tokens need not end in EOS. Mask actual generated tokens,
        # not the fixed-width padding. With ignore_eos, generated EOS is ordinary
        # content and must not discard later tokens that vLLM actually sampled.
        generated_counts = torch.tensor(
            [item["rollout_generated_tokens"] for item in termination_records], device=response_ids.device
        )
        ignored_eos = torch.tensor(
            [item["rollout_ignore_eos"] for item in termination_records], device=response_ids.device
        ).unsqueeze(-1)
        response_mask = torch.where(ignored_eos, torch.ones_like(response_mask), response_mask)
        response_mask *= (
            torch.arange(response_length, device=response_ids.device).unsqueeze(0) < generated_counts.unsqueeze(-1)
        ).to(response_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        tensors = {
            "prompts": input_ids,
            "responses": response_ids,
            "input_ids": sequence_ids,  # here input_ids become the whole sentences
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
        }
        if rollout_log_probs is not None:
            # vLLM's log-probabilities of the sampled tokens (0 after the end of the response).
            tensors["rollout_log_probs"] = rollout_log_probs * response_mask.to(torch.float32)
        batch = TensorDict(tensors, batch_size=batch_size)
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}
        non_tensor_batch.update(metadata_to_arrays(termination_records))

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    # ------------------------------------------------------------------
    # Route C (package v6)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_continuations(self, requests: DataProto) -> DataProto:
        """One sampled continuation per active row from ``raw_prompt_ids + trsi_rc_tail`` (Route C, package v6).

        The rows carry their rollout row's prompt (``raw_prompt_ids``, ``raw_prompt``, ``multi_modal_data``; rows with
        the same ``trsi_rc_media`` share these, and their media are prepared once), the response prefix plus
        candidate (``trsi_rc_tail``) and per-row ``trsi_rc_seed`` / ``trsi_rc_max_tokens``.
        Sampling uses the rollout's settings (``meta_info`` overrides as for ``generate_sequences``).  The prompt is
        given as token ids, so the prefix is exact.  vLLM's processed prompt (placeholders expanded) is checked: it
        must end with the tail (``trsi_rc_tail_ok``) and the part before it must be the prompt the trainer scores the
        rollout row with (``trsi_rc_prompt_ok``: same ``trsi_rc_prompt_digest`` and ``trsi_rc_prompt_length``).
        Rows with ``trsi_rc_active == 0`` (padding of the dispatch) are not generated.
        """

        length = int(self.config.response_length)
        size = len(requests)
        non_tensor = requests.non_tensor_batch
        raw_prompt_ids = non_tensor["raw_prompt_ids"]
        raw_prompts = non_tensor.get("raw_prompt")
        multi_modal = non_tensor.get("multi_modal_data")
        tails = non_tensor["trsi_rc_tail"]
        active = requests.batch["trsi_rc_active"].bool().tolist()
        seeds = requests.batch["trsi_rc_seed"].tolist()
        max_tokens = requests.batch["trsi_rc_max_tokens"].tolist()
        media = requests.batch["trsi_rc_media"].tolist()
        inputs, params, rows = [], [], []
        media_items: dict[int, dict[str, Any]] = {}
        with self.update_sampling_params(**requests.meta_info):
            for row in range(size):
                if not active[row]:
                    continue
                tail = [int(value) for value in tails[row]]
                if not tail:
                    raise ValueError(f"Continuation row {row} has an empty tail.")
                if int(max_tokens[row]) < 1:
                    raise ValueError(f"Continuation row {row} asks for {max_tokens[row]} tokens.")
                owner = int(media[row])
                if owner not in media_items:  # the media of a rollout prompt are prepared once per call
                    data = None if multi_modal is None else multi_modal[row]
                    media_items[owner] = (
                        {}
                        if data is None
                        else self._request_for_row(
                            raw_prompt_ids[row], None if raw_prompts is None else raw_prompts[row], data, requests.meta_info
                        )
                    )
                item = {key: value for key, value in media_items[owner].items() if key not in ("prompt", "prompt_token_ids")}
                item["prompt_token_ids"] = [int(value) for value in raw_prompt_ids[row]] + tail
                child = copy.copy(self.sampling_params)
                child.n = 1
                child.seed = int(seeds[row])
                child.max_tokens = int(max_tokens[row])
                child.logprobs = None  # the continuation's tokens only
                inputs.append(item)
                params.append(child)
                rows.append(row)
            completions = (
                self.inference_engine.generate(prompts=inputs, sampling_params=params, use_tqdm=self.use_tqdm)
                if inputs
                else []
            )
        if len(completions) != len(rows):
            raise RuntimeError(f"vLLM returned {len(completions)} completions for {len(rows)} continuation requests.")
        tokens = torch.full((size, length), int(self.pad_token_id), dtype=torch.long)
        counts = torch.zeros(size, dtype=torch.long)
        tail_ok = torch.zeros(size, dtype=torch.bool)
        prompt_ok = torch.zeros(size, dtype=torch.bool)
        termination_records = [{} for _ in range(size)]  # inactive dispatch padding; never used as a scored branch
        expected_digest = requests.batch["trsi_rc_prompt_digest"].tolist()
        expected_length = requests.batch["trsi_rc_prompt_length"].tolist()
        for row, completion, params_for_row in zip(rows, completions, params):
            seen = _processed_prompt(completion)
            tail = [int(value) for value in tails[row]]
            ends = len(seen) >= len(tail) and seen[len(seen) - len(tail) :] == tail
            head = seen[: len(seen) - len(tail)]
            tail_ok[row] = ends
            prompt_ok[row] = bool(
                ends and len(head) == int(expected_length[row]) and prompt_digest(head) == int(expected_digest[row])
            )
            if len(completion.outputs) != 1:
                raise RuntimeError(f"vLLM returned {len(completion.outputs)} outputs for continuation row {row}; expected one.")
            output = completion.outputs[0]
            termination_records[row] = self._completion_termination(
                completion, output, params_for_row, requests.meta_info.get("eos_token_id")
            )
            generated = [int(value) for value in output.token_ids]
            if len(generated) > length:
                raise RuntimeError(f"Continuation row {row} exceeds the response tensor width; cannot truncate metadata safely.")
            counts[row] = len(generated)
            if generated:
                tokens[row, : len(generated)] = torch.tensor(generated, dtype=torch.long)
        batch = TensorDict(
            {"trsi_rc_tokens": tokens, "trsi_rc_length": counts, "trsi_rc_tail_ok": tail_ok, "trsi_rc_prompt_ok": prompt_ok},
            batch_size=size,
        )
        return DataProto(batch=batch, non_tensor_batch=metadata_to_arrays(termination_records), meta_info=requests.meta_info)

