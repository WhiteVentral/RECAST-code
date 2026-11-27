from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch

from ..protocol import DataProto
from ..utils import torch_functional as VF
from ..utils.dataset import process_image, process_video
from .mesd_algos import (
    MESD_PRIVILEGED_VIEW_NAMES,
    build_mesd_privileged_views,
    build_mesd_view_hint,
    inject_mesd_hindsight_hint,
    mesd_view_available,
)


def _compute_position_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model_inputs: dict[str, Any],
    processor: Any,
) -> torch.Tensor:
    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        if "Qwen3VLProcessor" in processor.__class__.__name__:
            from ..models.transformers.qwen3_vl import get_rope_index
        else:
            from ..models.transformers.qwen2_vl import get_rope_index

        vision_position_ids = get_rope_index(
            processor,
            input_ids=input_ids,
            image_grid_thw=model_inputs.get("image_grid_thw", None),
            video_grid_thw=model_inputs.get("video_grid_thw", None),
            second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
            attention_mask=attention_mask,
        )
        text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
    else:
        position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)
    return position_ids


def _tokenize_prompt(
    prompt: str,
    tokenizer: Any,
    processor: Any,
    multi_modal_item: Optional[dict[str, Any]],
    max_prompt_length: int,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
    video_min_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
    video_fps: Optional[float] = None,
    video_max_frames: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], str]:
    multi_modal_item = multi_modal_item or {}

    if processor is not None and "images" in multi_modal_item and len(multi_modal_item["images"]) > 0:
        processed_images = [
            process_image(image, min_pixels=image_min_pixels, max_pixels=image_max_pixels)
            for image in multi_modal_item["images"]
        ]
        model_inputs = processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
    elif processor is not None and "preprocessed_video_path" in multi_modal_item:
        preprocessed_data = torch.load(
            multi_modal_item["preprocessed_video_path"],
            map_location="cpu",
            weights_only=False,
        )
        model_inputs = processor(
            [prompt],
            [preprocessed_data["frames"]],
            add_special_tokens=False,
            video_metadata=[preprocessed_data["metadata"]],
            return_tensors="pt",
            do_resize=False,
            do_sample_frames=False,
        )
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
    elif processor is not None and "video" in multi_modal_item and len(multi_modal_item["video"]) > 0:
        processor_kwargs: dict[str, Any] = {
            "text": [prompt],
            "videos": multi_modal_item["video"],
            "add_special_tokens": False,
            "return_tensors": "pt",
            "do_resize": False,
            "do_sample_frames": False,
        }
        video_metadatas = multi_modal_item.get("video_metadatas", None)
        if video_metadatas is not None and len(video_metadatas) > 0:
            processor_kwargs["video_metadata"] = video_metadatas
        model_inputs = processor(**processor_kwargs)
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
    elif processor is not None and "videos" in multi_modal_item and len(multi_modal_item["videos"]) > 0:
        processed_video_frames = []
        video_metadatas = []
        for video in multi_modal_item["videos"]:
            result = process_video(
                video,
                video_min_pixels if video_min_pixels else 4096,
                video_max_pixels if video_max_pixels else 65536,
                video_max_frames,
                video_fps,
                True,
            )
            if isinstance(result, tuple) and len(result) == 2:
                video_data, _ = result
                if isinstance(video_data, tuple) and len(video_data) == 2:
                    frames, metadata = video_data
                    processed_video_frames.append(frames)
                    video_metadatas.append(metadata)
                else:
                    processed_video_frames.append(video_data)
                    video_metadatas = None
                    break
            else:
                processed_video_frames.append(result)
                video_metadatas = None

        processor_kwargs = {
            "text": [prompt],
            "videos": processed_video_frames,
            "add_special_tokens": False,
            "return_tensors": "pt",
            "do_sample_frames": False,
        }
        if video_metadatas is not None and len(video_metadatas) > 0:
            processor_kwargs["video_metadata"] = video_metadatas
            processor_kwargs["do_resize"] = False
        model_inputs = processor(**processor_kwargs)
        input_ids = model_inputs.pop("input_ids")[0]
        attention_mask = model_inputs.pop("attention_mask")[0]
    else:
        model_inputs = {}
        model_inputs = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]

    position_ids = _compute_position_ids(input_ids, attention_mask, model_inputs, processor)
    input_ids, attention_mask, position_ids = VF.postprocess_data(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_length=max_prompt_length,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation="left",
    )

    raw_prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(raw_prompt_ids) > max_prompt_length:
        raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]

    return input_ids, attention_mask, position_ids, raw_prompt_ids, prompt


def construct_mesd_multiview_teacher_prompts(
    student_raw_prompts: list[Any] | np.ndarray,
    privileged_infos: list[Mapping[str, Any]] | np.ndarray,
    tokenizer: Any,
    processor: Any,
    max_prompt_length: int,
    multi_modal_data: Optional[list[dict[str, Any]] | np.ndarray] = None,
    image_min_pixels: Optional[int] = None,
    image_max_pixels: Optional[int] = None,
    video_min_pixels: Optional[int] = None,
    video_max_pixels: Optional[int] = None,
    video_fps: Optional[float] = None,
    video_max_frames: Optional[int] = None,
    cache_keys: Optional[list[Any] | np.ndarray] = None,
) -> DataProto:
    """Build the A/A+T/A+S privileged prompt views without expanding batch ``B``.

    The fixed view order is ``("answer", "temporal", "spatial")``.  The
    returned tensor shapes are::

        input_ids:        [B, 3, prompt_length]
        attention_mask:   [B, 3, prompt_length]
        position_ids:     [B, 3, prompt_length] or [B, 3, C, prompt_length]
        view_available:   [B, 3]

    ``view_names`` is stored in ``DataProto.meta_info``.  The original
    multi-modal item is retained once per row in ``non_tensor_batch`` and must
    be reused for every view by the actor.  This keeps the DataProto batch axis
    aligned with student completions, rewards, and GRPO advantages.

    A missing temporal or spatial view does not fail prompt construction.  It
    reuses the answer-view hint when available, otherwise the unmodified
    student prompt, while ``view_available`` remains false.  Cache entries are
    scoped by ``(cache_key, view_name)`` so different views of one rollout
    group can never alias.
    """

    batch_size = len(student_raw_prompts)
    if len(privileged_infos) != batch_size:
        raise ValueError(
            "MeSD student prompts and privileged rows must have identical length, "
            f"got {batch_size} and {len(privileged_infos)}."
        )
    if multi_modal_data is not None and len(multi_modal_data) != batch_size:
        raise ValueError(
            f"MeSD multi-modal rows must align with student prompts, got {len(multi_modal_data)} and {batch_size}."
        )
    if cache_keys is not None and len(cache_keys) != batch_size:
        raise ValueError(
            f"MeSD teacher prompt cache keys must align with the batch, got {len(cache_keys)} and {batch_size}."
        )
    if batch_size == 0:
        raise ValueError("MeSD multi-view prompt construction requires a non-empty batch.")

    view_names = MESD_PRIVILEGED_VIEW_NAMES
    all_input_ids: list[list[torch.Tensor]] = []
    all_attention_mask: list[list[torch.Tensor]] = []
    all_position_ids: list[list[torch.Tensor]] = []
    all_raw_prompt_ids = np.empty((batch_size, len(view_names)), dtype=object)
    all_raw_prompt = np.empty((batch_size, len(view_names)), dtype=object)
    all_multi_modal = np.empty((batch_size,), dtype=object)
    all_view_available: list[list[bool]] = []
    tokenized_by_key: dict[
        Any,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], str],
    ] = {}

    for index in range(batch_size):
        student_prompt = student_raw_prompts[index]
        if not isinstance(student_prompt, str):
            student_prompt = str(student_prompt)
        mm_item = multi_modal_data[index] if multi_modal_data is not None else None
        all_multi_modal[index] = mm_item if mm_item is not None else {}

        view_info = build_mesd_privileged_views(privileged_infos[index])
        rendered_hints: dict[str, str] = {}
        availability: dict[str, bool] = {}
        for view_name in view_names:
            rendered = build_mesd_view_hint(view_info, view_name)
            hint = rendered if isinstance(rendered, str) else str(rendered or "")
            rendered_hints[view_name] = hint
            # Treat an inconsistent "available but empty" renderer result as a
            # missing view.  This preserves the no-failure fallback contract.
            availability[view_name] = bool(mesd_view_available(view_info, view_name) and hint.strip())

        answer_fallback = rendered_hints["answer"] if availability["answer"] else ""
        row_input_ids: list[torch.Tensor] = []
        row_attention_mask: list[torch.Tensor] = []
        row_position_ids: list[torch.Tensor] = []
        row_availability: list[bool] = []

        for view_index, view_name in enumerate(view_names):
            is_available = availability[view_name]
            hint = rendered_hints[view_name] if is_available else answer_fallback
            teacher_prompt = inject_mesd_hindsight_hint(student_prompt, hint) if hint.strip() else student_prompt

            cache_key = None
            if cache_keys is not None:
                candidate = (cache_keys[index], view_name)
                try:
                    hash(candidate)
                except TypeError:
                    # Correctness does not rely on caching; custom callers may
                    # supply an unhashable row identifier.
                    pass
                else:
                    cache_key = candidate

            cached = None if cache_key is None else tokenized_by_key.get(cache_key)
            if cached is None:
                cached = _tokenize_prompt(
                    teacher_prompt,
                    tokenizer,
                    processor,
                    mm_item,
                    max_prompt_length,
                    image_min_pixels,
                    image_max_pixels,
                    video_min_pixels,
                    video_max_pixels,
                    video_fps,
                    video_max_frames,
                )
                if cache_key is not None:
                    tokenized_by_key[cache_key] = cached

            input_ids, attention_mask, position_ids, raw_prompt_ids, raw_prompt = cached
            row_input_ids.append(input_ids)
            row_attention_mask.append(attention_mask)
            row_position_ids.append(position_ids)
            all_raw_prompt_ids[index, view_index] = raw_prompt_ids
            all_raw_prompt[index, view_index] = raw_prompt
            row_availability.append(is_available)

        all_input_ids.append(row_input_ids)
        all_attention_mask.append(row_attention_mask)
        all_position_ids.append(row_position_ids)
        all_view_available.append(row_availability)

    tensors = {
        "input_ids": torch.stack([torch.stack(row, dim=0) for row in all_input_ids], dim=0),
        "attention_mask": torch.stack([torch.stack(row, dim=0) for row in all_attention_mask], dim=0),
        "position_ids": torch.stack([torch.stack(row, dim=0) for row in all_position_ids], dim=0),
        "view_available": torch.tensor(all_view_available, dtype=torch.bool),
    }
    non_tensors = {
        "raw_prompt_ids": all_raw_prompt_ids,
        "raw_prompt": all_raw_prompt,
        "multi_modal_data": all_multi_modal,
    }
    return DataProto.from_dict(
        tensors=tensors,
        non_tensors=non_tensors,
        meta_info={
            "view_names": list(view_names),
            "view_axis": 1,
        },
    )
