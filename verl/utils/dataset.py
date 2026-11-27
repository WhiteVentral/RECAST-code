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

import math
from io import BytesIO
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep every column aligned with sample order, including optional fields."""
    batch = {}
    keys = dict.fromkeys(key for feature in features for key in feature)
    for key in keys:
        values = [feature.get(key) for feature in features]
        is_tensor = [isinstance(value, torch.Tensor) for value in values]
        if any(is_tensor):
            if not all(is_tensor):
                invalid_rows = [index for index, valid in enumerate(is_tensor) if not valid]
                raise ValueError(f"Tensor field {key!r} is missing or non-tensor at rows {invalid_rows}.")
            batch[key] = torch.stack(values, dim=0)
        else:
            # Assign individual objects so sequences stay within one batch axis.
            column = np.empty(len(features), dtype=object)
            for index, value in enumerate(values):
                column[index] = value
            batch[key] = column
    return batch


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    elif isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(f"Expected a rank-3 image tensor, got shape {tuple(tensor.shape)}")
        if tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and (float(array.max()) if array.size else 0.0) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array.squeeze(-1) if array.shape[-1] == 1 else array)
    elif isinstance(image, np.ndarray):
        array = image
        if array.ndim != 3:
            raise ValueError(f"Expected a rank-3 image array, got shape {array.shape}")
        if array.shape[0] in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and (float(array.max()) if array.size else 0.0) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array.squeeze(-1) if array.shape[-1] == 1 else array)

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str,
    min_pixels: int = 4 * 32 * 32,
    max_pixels: int = 64 * 32 * 32,
    max_frames: int = 128,
    video_fps: float = 2.0,
    return_fps: bool = False,
) -> Union[list[ImageObject], tuple[list[ImageObject], list[float]]]:
    vision_info = {
        "video": video,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "max_frames": max_frames,
        "fps": video_fps,
    }
    return fetch_video(
        vision_info,
        image_patch_size=16,
        return_video_sample_fps=return_fps,
        return_video_metadata=return_fps,
    )


class STGRDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "question",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        video_max_frames: int = 16,
        max_prompt_length: int = 16384,
        truncation: str = "left",
        image_min_pixels: Optional[int] = None,
        image_max_pixels: Optional[int] = None,
        video_min_pixels: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = False,
        apply_chat_template_kwargs: Optional[dict] = None,
        dataset_format: str = "stgr",
        media_root: Optional[str] = None,
        stgr_frame_cache_dir: Optional[str] = None,
        stgr_frame_cache_required: bool = False,
    ):
        from mesd.schema import load_stgr_json_records

        if dataset_format != "stgr":
            raise ValueError("Only local STGR JSON/JSONL annotations are supported.")
        if media_root is None:
            raise ValueError("STGR data requires data.media_root (DATA_ROOT containing videos/).")
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.media_root = media_root
        self.stgr_frame_cache_dir = stgr_frame_cache_dir
        self.stgr_frame_cache_required = stgr_frame_cache_required
        # Sparse key_items must remain Python dictionaries, not an Arrow union schema.
        self.dataset = load_stgr_json_records(data_path)
        print(f"Loaded {len(self.dataset)} STGR rows with the sparse Python JSON backend.")
        if filter_overlong_prompts:
            print("STGR overlength handling is deferred to exact item preprocessing.")

    def _prepare_stgr_item(self, row: dict[str, Any]):
        """Build Open-o3's timestamped multi-image representation once.

        The same processed PIL images are reused by vLLM rollout and the FSDP
        actor/teacher forwards, avoiding the native-video semantic mismatch.
        """
        if self.processor is None:
            raise ValueError("STGR video/image preprocessing requires a multimodal processor.")

        from mesd.preprocess import (
            VIDEO_PLACEHOLDER,
            STGRVisionError,
            build_frame_prompt,
            prepare_stgr_example,
        )
        from mesd.prompts import build_student_messages
        from mesd.schema import coerce_record, resolve_media_path

        record = coerce_record(row)
        vision_error = None
        try:
            prepared = prepare_stgr_example(
                record,
                self.media_root,
                frame_cache_dir=self.stgr_frame_cache_dir,
                frame_cache_required=getattr(self, "stgr_frame_cache_required", False),
            )
            prompt = prepared.render_prompt(self.processor)
            source_images = prepared.images
            media_path = prepared.media_path
            reward_metadata = prepared.reward_metadata
            frame_times = list(prepared.frame_times)
            source_frame_count = prepared.source_frame_count
            sample_fps = prepared.sample_fps
        except (STGRVisionError, OSError) as exc:
            # MeSD keeps all ranks aligned on recoverable media failures.  The
            # central veRL dataloader can do this more simply: emit a valid
            # dummy visual payload, mark it, and force every reward component
            # (therefore the GRPO advantage) to zero.  We never optimize a
            # guessed answer from a broken video.
            vision_error = exc
            dummy = Image.new("RGB", (56, 56), color=(0, 0, 0))
            messages = build_student_messages(record)
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if record.task == "visual QA":
                source_images = [dummy]
                frame_times = []
                source_frame_count = 0
                sample_fps = None
            else:
                frame_times = [0.0, 0.5, 1.0, 1.5]
                frame_prompt = build_frame_prompt(frame_times, duration_seconds=2)
                if VIDEO_PLACEHOLDER not in prompt:
                    raise ValueError("Chat template did not emit the expected Qwen video placeholder")
                prompt = prompt.replace(VIDEO_PLACEHOLDER, frame_prompt, 1)
                source_images = [dummy.copy() for _ in frame_times]
                source_frame_count = len(source_images)
                sample_fps = 2.0
            media_path = resolve_media_path(record, self.media_root)
            reward_metadata = record.to_reward_metadata(processed_image_size=dummy.size)
            print(
                f"[STGR] recoverable media failure for id={record.id}: "
                f"{type(exc).__name__}: {exc}; emitting a zero-reward skip sample."
            )

        processed_images = [
            process_image(image, self.image_min_pixels, self.image_max_pixels) for image in source_images
        ]
        model_inputs = self.processor(
            images=processed_images,
            videos=None,
            text=[prompt],
            add_special_tokens=False,
            return_tensors="pt",
        )

        example = dict(row)
        # Start from the task adapter's canonical reward payload, then update
        # the coordinate-space size after the final dataset-side resize.  This
        # is the exact image space seen by both vLLM and the actor/teacher.
        example.update(reward_metadata)
        processed_size = list(processed_images[0].size)
        if record.task == "visual QA":
            example["image_size_refine"] = processed_size
            example["processed_image_size"] = processed_size
            # Keep optional video metadata present on every STGR row.  The
            # collator batches non-tensor values by key, so omitting these on
            # image rows would produce a shorter array whenever a batch mixes
            # visual QA and video tasks.
            example["video_sample_fps"] = None
            example["video_duration"] = 0.0
        else:
            example["image_size"] = processed_size
            example["processed_image_size"] = processed_size
            example["video_sample_fps"] = sample_fps
            example["video_duration"] = source_frame_count / sample_fps if sample_fps else 0.0
        example["frame_times"] = frame_times
        example["prompt_text_final"] = prompt
        example["resolved_media_path"] = str(media_path)
        example["vision_read_failed"] = vision_error is not None
        example["vision_error_type"] = None if vision_error is None else type(vision_error).__name__
        example["vision_error_message"] = None if vision_error is None else str(vision_error)[:500]
        example["multi_modal_data"] = {"images": processed_images}
        return example, prompt, model_inputs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        teacher_question = example[self.prompt_key]

        example, prompt, model_inputs = self._prepare_stgr_item(example)
        prompt_text = prompt
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]

        # Clean up images/videos keys if they still exist
        if self.image_key in example:
            example.pop(self.image_key, None)
        if self.video_key in example:
            example.pop(self.video_key, None)

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["raw_prompt"] = prompt  # Preserve the original text prompt for vLLM video preprocessing.
        example["prompt_text"] = prompt_text  # Text before applying the chat template, used for teacher replay.
        example["teacher_question"] = teacher_question  # Original question text for teacher-specific templates.
        example["ground_truth"] = example.pop(self.answer_key)

        return example
