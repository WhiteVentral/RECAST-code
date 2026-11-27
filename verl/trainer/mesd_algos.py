"""Pure helpers for MeSD multiview refinement and Direct RKL.

This module intentionally has no Ray/Transformers dependency.  Keeping the
schedule and reweighting math here makes it possible to test the algorithmic
contract on CPU without constructing a training worker.
"""

from __future__ import annotations

import ast
import json
import math
import re
from copy import deepcopy
from typing import Any, Mapping, Optional

import torch


QWEN_ASSISTANT_MARKER = "<|im_end|>\n<|im_start|>assistant\n"
MESD_PRIVILEGED_VIEW_NAMES = ("answer", "temporal", "spatial")
_TEMPORAL_TASKS = frozenset({"temporal QA", "temporal QA (MCQ)", "temporal-spatial free-form QA"})
_SPATIAL_TASKS = frozenset({"visual QA", "temporal-spatial free-form QA"})


def _require_finite_tensor(name: str, value: torch.Tensor, *, valid_mask: Optional[torch.Tensor] = None) -> None:
    """Fail fast before non-finite teacher evidence can poison an RL update."""

    finite = torch.isfinite(value)
    if valid_mask is not None:
        finite = finite | ~valid_mask.to(device=value.device, dtype=torch.bool)
    if not bool(finite.all().item()):
        raise FloatingPointError(f"{name} contains NaN or infinity on active entries.")


def _normalize_time_window(value: Any) -> Optional[list[float]]:
    """Read the two endpoint values used by the temporal hindsight view."""
    endpoints = value
    if isinstance(value, str):
        try:
            endpoints = ast.literal_eval(value.strip())
        except (SyntaxError, ValueError):
            endpoints = None
        if not isinstance(endpoints, (tuple, list)):
            number = r"(\d+(?:\.\d+)?)"
            interval = re.search(rf"<t>{number}</t>s to <t>{number}</t>s", value)
            endpoints = interval.groups() if interval else None

    if not isinstance(endpoints, (tuple, list)) or len(endpoints) != 2:
        return None
    try:
        return list(map(float, endpoints))
    except (TypeError, ValueError):
        return None


def _normalize_answer_window(task: Any, answer: Any) -> Optional[list[float]]:
    if str(task or "") not in ("temporal QA", "temporal QA (MCQ)"):
        return None
    if isinstance(answer, str):
        answer = answer.strip()
        if "\n[" in answer:
            _, _, answer = answer.partition("\n")
    return _normalize_time_window(answer)


def _as_finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _normalize_frame_index(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None
    return number


def _extract_bboxes(value: Any) -> list[list[float]]:
    """Extract strict four-coordinate boxes from released key-item variants."""

    if not isinstance(value, (str, bytes, Mapping, list, tuple)) and hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping):
        boxes: list[list[float]] = []
        for key in ("bbox", "box", "bboxes", "boxes"):
            if key in value:
                boxes.extend(_extract_bboxes(value[key]))
        return boxes
    if not isinstance(value, (list, tuple)):
        return []
    if len(value) == 4:
        coordinates = [_as_finite_float(item) for item in value]
        if all(item is not None for item in coordinates):
            return [[float(item) for item in coordinates]]
    boxes = []
    for item in value:
        boxes.extend(_extract_bboxes(item))
    return boxes


def _normalize_image_size(value: Any) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    if not isinstance(value, (str, bytes, Mapping, list, tuple)) and hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    width = _as_finite_float(value[0])
    height = _as_finite_float(value[1])
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    return width, height


def _first_image_size(*values: Any) -> Optional[tuple[float, float]]:
    for value in values:
        size = _normalize_image_size(value)
        if size is not None:
            return size
    return None


def _scale_objects(objects: list[dict[str, Any]], *, scale_x: float, scale_y: float) -> list[dict[str, Any]]:
    scaled = []
    for obj in objects:
        boxes = []
        for box in obj.get("bboxes") or []:
            if isinstance(box, (list, tuple)) and len(box) == 4:
                boxes.append(
                    [
                        float(box[0]) * scale_x,
                        float(box[1]) * scale_y,
                        float(box[2]) * scale_x,
                        float(box[3]) * scale_y,
                    ]
                )
        if boxes:
            scaled.append({"name": obj["name"], "bboxes": boxes})
    return scaled


def _filter_xyxy_objects(
    objects: list[dict[str, Any]],
    *,
    max_x: float,
    max_y: float,
    tolerance: float = 0.0,
) -> list[dict[str, Any]]:
    """Keep only ordered xyxy boxes inside the declared coordinate space."""

    if not (math.isfinite(max_x) and math.isfinite(max_y) and max_x > 0.0 and max_y > 0.0):
        return []
    filtered = []
    for obj in objects:
        boxes = []
        for box in obj.get("bboxes") or []:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            x1, y1, x2, y2 = (float(value) for value in box)
            if -tolerance <= x1 <= x2 <= max_x + tolerance and -tolerance <= y1 <= y2 <= max_y + tolerance:
                boxes.append(
                    [
                        min(max(x1, 0.0), max_x),
                        min(max(y1, 0.0), max_y),
                        min(max(x2, 0.0), max_x),
                        min(max(y2, 0.0), max_y),
                    ]
                )
        if boxes:
            filtered.append({"name": obj["name"], "bboxes": boxes})
    return filtered


def _redact_answer_time_expressions(text: str) -> str:
    """Remove explicit timestamps from a joint task's semantic answer view."""

    value = re.sub(
        r"\b(?:from|between)\s+\d+(?:\.\d+)?\s+(?:to|and)\s+"
        r"\d+(?:\.\d+)?\s*(?:seconds?|secs?|s)?\b",
        "[TIME]",
        text,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\b\d+(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)*\s*"
        r"(?:-\s*)?(?:seconds?|secs?|s)\b(?:\s*mark)?",
        "[TIME]",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\b(?:timestamp|time)\s*(?:of|=|:)?\s*\d+(?:\.\d+)?\b",
        "[TIME]",
        value,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", value).strip()


def _parse_visual_answer(answer: Any) -> tuple[str, list[dict[str, Any]]]:
    text = "" if answer is None else str(answer).strip()
    matches = re.findall(
        r"<obj>\s*(.*?)\s*</obj>\s*<box>\s*(.*?)\s*</box>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    objects: list[dict[str, Any]] = []
    names = []
    for raw_name, raw_box in matches:
        name = re.sub(r"\s+", " ", raw_name).strip()
        try:
            parsed_box = ast.literal_eval(raw_box)
        except (SyntaxError, ValueError):
            parsed_box = None
        boxes = _extract_bboxes(parsed_box)
        if name:
            names.append(name)
        if name and boxes:
            objects.append({"name": name, "bboxes": boxes})
    if names:
        return "; ".join(dict.fromkeys(names)), objects

    # If an annotation is malformed, retain semantic text but never expose the
    # contents of spatial/time tags in the answer-only view.
    semantic = re.sub(r"<box>.*?</box>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    semantic = re.sub(r"</?obj>", " ", semantic, flags=re.IGNORECASE)
    semantic = re.sub(r"<t>.*?</t>s?", " ", semantic, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\s+", " ", semantic).strip(), []


def _semantic_answer_and_window(task: str, answer: Any) -> tuple[str, Optional[list[float]]]:
    window = _normalize_answer_window(task, answer)
    if answer is None:
        return "", window
    if not isinstance(answer, str):
        if window is not None:
            return "", window
        return str(answer).strip(), None

    text = answer.strip()
    if task in {"temporal QA", "temporal QA (MCQ)"} and window is not None:
        lines = text.splitlines()
        kept = []
        for line in lines:
            parsed = _normalize_time_window(line.strip())
            if parsed is not None and parsed == window:
                continue
            kept.append(line)
        text = "\n".join(kept).strip()
        if _normalize_time_window(text) is not None:
            text = ""

    # Structured privileged values must not leak into the answer anchor.
    text = re.sub(r"<box>.*?</box>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<t>.*?</t>s?", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"</?obj>", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip(), window


def _key_item_objects(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping):
        return []
    objects = []
    for raw_name, raw_boxes in value.items():
        name = str(raw_name).strip()
        boxes = _extract_bboxes(raw_boxes)
        if name and boxes:
            objects.append({"name": name, "bboxes": boxes})
    return objects


def build_mesd_privileged_views(example: Mapping[str, Any] | None) -> dict[str, Any]:
    """Parse strict answer, temporal, and spatial privileged views.

    Temporal and spatial *hindsight fields* are intentionally kept disjoint.
    The common Open-o3 visual prompt is not changed here and may already
    contain the dataset-inserted gold keyframe and timestamp.  For joint video
    samples, spatial evidence is admitted only when ``key_items`` can be joined
    to an annotated key frame through the shared frame ``idx``.  The matching
    timestamp is retained only in ``spatiotemporal_evidence``; the spatial
    renderer never serializes it as an additional hidden field.
    """

    row = dict(example or {})
    if {
        "answer_text",
        "temporal_keyframes",
        "spatial_evidence",
        "availability",
    }.issubset(row):
        # Prompt construction may receive rows that were parsed centrally by
        # the trainer.  Preserve that representation instead of treating it as
        # a raw STGR row and silently dropping every view.
        return deepcopy(row)
    task = str(row.get("task") or "")
    raw_answer = row.get("ground_truth")
    if raw_answer is None:
        raw_answer = row.get("answer")

    if task == "visual QA":
        answer_text, visual_objects = _parse_visual_answer(raw_answer)
        annotation_size = _first_image_size(row.get("image_size"), row.get("annotation_image_size"))
        processed_size = _first_image_size(row.get("image_size_refine"), row.get("processed_image_size"))
        if annotation_size is not None and processed_size is not None:
            visual_objects = _filter_xyxy_objects(
                visual_objects,
                max_x=annotation_size[0],
                max_y=annotation_size[1],
                # Released GQA boxes contain a handful of 1--3 pixel
                # boundary overflows.  Clamp those annotation round-off
                # artifacts while still rejecting materially wrong boxes.
                tolerance=3.0,
            )
            visual_objects = _scale_objects(
                visual_objects,
                scale_x=processed_size[0] / annotation_size[0],
                scale_y=processed_size[1] / annotation_size[1],
            )
            visual_coordinate_space = "processed_pixel_xyxy"
        else:
            # A box in the annotation coordinate system is not valid
            # supervision for the resized image actually seen by the model.
            visual_objects = []
            visual_coordinate_space = None
        window = None
    else:
        answer_text, window = _semantic_answer_and_window(task, raw_answer)
        visual_objects = []
        visual_coordinate_space = None
    if task in {"temporal QA", "temporal QA (MCQ)"}:
        explicit_window = _normalize_time_window(row.get("ground_truth_window"))
        if explicit_window is not None:
            window = explicit_window

    raw_frames = row.get("key_frames")
    if raw_frames is None:
        raw_frames = row.get("gold_keyframes")
    if raw_frames is None:
        raw_frames = []
    elif not isinstance(raw_frames, (str, bytes, Mapping)) and hasattr(raw_frames, "tolist"):
        raw_frames = raw_frames.tolist()

    temporal_keyframes = []
    frame_by_index: dict[str, dict[str, Any]] = {}
    if task in _TEMPORAL_TASKS:
        for raw_frame in raw_frames:
            if not isinstance(raw_frame, Mapping):
                continue
            frame_idx = _normalize_frame_index(raw_frame.get("idx"))
            timestamp = _as_finite_float(raw_frame.get("time", raw_frame.get("timestamp")))
            if frame_idx is None or timestamp is None:
                continue
            frame = {"frame_idx": frame_idx, "timestamp": timestamp}
            temporal_keyframes.append(frame)
            frame_by_index[str(frame_idx)] = frame
    if task == "temporal-spatial free-form QA" and answer_text:
        answer_text = _redact_answer_time_expressions(answer_text)

    spatial_evidence = []
    spatiotemporal_evidence = []
    if task == "visual QA" and visual_objects:
        spatial_evidence.append(
            {
                "anchor": "input_image",
                "coordinate_space": visual_coordinate_space,
                "objects": visual_objects,
            }
        )
    elif task == "temporal-spatial free-form QA":
        processed_size = _first_image_size(row.get("processed_image_size"), row.get("image_size"))
        raw_items = row.get("key_items")
        if isinstance(raw_items, Mapping):
            for frame_key, frame in frame_by_index.items():
                item_value = raw_items.get(frame_key)
                if item_value is None:
                    try:
                        item_value = raw_items.get(int(frame_key))
                    except ValueError:
                        item_value = None
                objects = _filter_xyxy_objects(
                    _key_item_objects(item_value),
                    max_x=1.0,
                    max_y=1.0,
                    tolerance=1.0e-6,
                )
                if not objects:
                    continue
                if processed_size is not None:
                    objects = _scale_objects(
                        objects,
                        scale_x=processed_size[0],
                        scale_y=processed_size[1],
                    )
                    coordinate_space = "processed_pixel_xyxy"
                else:
                    # Joint key_items are normalized, while grounded model
                    # outputs/rewards use processed-image pixels.  Do not emit
                    # a spatial view when that conversion is unavailable.
                    continue
                spatial = {
                    # The released annotation idx is only a join key; it is
                    # not the merged prompt's visible ``Frame N``.  Name the
                    # actual semantic anchor without inventing a frame map.
                    "anchor": "annotated_keyframe_image",
                    "coordinate_space": coordinate_space,
                    "objects": objects,
                }
                spatial_evidence.append(spatial)
                spatiotemporal_evidence.append(
                    {
                        "annotation_idx": frame["frame_idx"],
                        "timestamp": frame["timestamp"],
                        "coordinate_space": coordinate_space,
                        "objects": deepcopy(objects),
                    }
                )

    temporal_available = bool(task in _TEMPORAL_TASKS and (window is not None or temporal_keyframes))
    spatial_available = bool(task in _SPATIAL_TASKS and spatial_evidence)
    return {
        "question": row.get("question") or row.get("teacher_question"),
        "task": task,
        "sample_id": row.get("id") or row.get("sample_id"),
        "answer_text": answer_text,
        "temporal_window": window,
        "temporal_keyframes": temporal_keyframes,
        "spatial_evidence": spatial_evidence,
        "spatiotemporal_evidence": spatiotemporal_evidence,
        "availability": {
            "answer": bool(answer_text),
            "temporal": temporal_available,
            "spatial": spatial_available,
        },
    }


def mesd_view_available(info: Mapping[str, Any] | None, view_name: str) -> bool:
    if view_name not in MESD_PRIVILEGED_VIEW_NAMES:
        raise ValueError(f"Unknown MeSD privileged view: {view_name!r}.")
    availability = (info or {}).get("availability") or {}
    return bool(availability.get(view_name, False))


def _format_box(box: list[float]) -> str:
    return json.dumps(box, ensure_ascii=False, separators=(",", ":"))


def build_mesd_view_hint(info: Mapping[str, Any] | None, view_name: str) -> str:
    """Render one strict nested context: A, A+T, or A+S."""

    if view_name not in MESD_PRIVILEGED_VIEW_NAMES:
        raise ValueError(f"Unknown MeSD privileged view: {view_name!r}.")
    data = info or {}
    parts = [
        f"[Hidden hindsight evidence: {view_name} view]",
        "Use only the verified fields listed below as hidden supervision.",
    ]
    answer = str(data.get("answer_text") or "").strip()
    if answer:
        parts.append(f"Verified semantic answer: {answer}")

    if view_name == "temporal":
        window = data.get("temporal_window")
        if isinstance(window, (list, tuple)) and len(window) == 2:
            parts.append(f"Verified temporal window: [{window[0]}, {window[1]}]")
        for frame in data.get("temporal_keyframes") or []:
            if isinstance(frame, Mapping):
                parts.append(f"Verified keyframe timestamp: {frame.get('timestamp')}s")
    elif view_name == "spatial":
        for frame in data.get("spatial_evidence") or []:
            if not isinstance(frame, Mapping):
                continue
            anchor = frame.get("anchor")
            coordinate_space = frame.get("coordinate_space")
            for obj in frame.get("objects") or []:
                if not isinstance(obj, Mapping):
                    continue
                name = str(obj.get("name") or "").strip()
                for box in obj.get("bboxes") or []:
                    if name and isinstance(box, list):
                        parts.append(
                            "Verified spatial target: "
                            f"anchor={anchor}, coordinate_space={coordinate_space}, "
                            f"object={name}, bbox={_format_box(box)}"
                        )

    # An empty answer anchor is represented by the original student context,
    # not by a metadata-only hidden block.
    if len(parts) == 2:
        return ""
    return "\n" + "\n".join(parts) + "\n"


def inject_mesd_hindsight_hint(
    student_prompt: str,
    hint: str,
    *,
    assistant_marker: str = QWEN_ASSISTANT_MARKER,
) -> str:
    """Inject hindsight immediately before the final assistant turn."""

    if not hint.strip():
        return student_prompt
    marker_pos = student_prompt.rfind(assistant_marker)
    if marker_pos < 0:
        raise ValueError(
            "MeSD requires a preformatted Qwen chat prompt ending in an assistant turn; "
            f"assistant marker {assistant_marker!r} was not found."
        )
    return student_prompt[:marker_pos] + hint + student_prompt[marker_pos:]


def compute_mesd_mixing_lambda(*, initial_lambda: float, anneal_steps: int, step: int) -> float:
    """MeSD's 1-indexed linear schedule.

    Step 1 uses the full initial lambda, step ``anneal_steps`` uses
    ``initial_lambda / anneal_steps``, and the following step is pure GRPO.
    """

    if anneal_steps <= 0 or step <= 0:
        return float(initial_lambda)
    if step > anneal_steps:
        return 0.0
    remaining = max(float(anneal_steps - (step - 1)), 0.0) / float(anneal_steps)
    return float(initial_lambda) * remaining


def mesd_teacher_enabled(*, disable_after_step: int, step: int) -> bool:
    return not (disable_after_step > 0 and step > disable_after_step)


def compute_mesd_common_support_log_probs(
    *,
    student_logits: torch.Tensor,
    answer_teacher_logits: torch.Tensor,
    temporal_teacher_logits: torch.Tensor,
    spatial_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    view_availability: Optional[torch.Tensor] = None,
    top_k: int = 16,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Normalize Student and all teacher views on one deduplicated support.

    The support is the union of Student top-k, each *available* teacher's
    top-k, and the realized Student token. ``view_availability`` is ``[B, 3]``
    in ``(A, T, S)`` order. Sorting plus a validity mask removes duplicate
    token ids without allocating a dense ``[B, R, vocabulary]`` boolean
    tensor. Returned log probabilities and all support metadata are
    stop-gradient.
    """

    distributions = {
        "student": student_logits,
        "answer": answer_teacher_logits,
        "temporal": temporal_teacher_logits,
        "spatial": spatial_teacher_logits,
    }
    expected = tuple(student_logits.shape)
    if student_logits.ndim != 3:
        raise ValueError("MeSD logits must have shape [batch, response, vocabulary].")
    for name, logits in distributions.items():
        if tuple(logits.shape) != expected:
            raise ValueError(f"{name} logits must have shape {expected}, got {tuple(logits.shape)}.")
    if tuple(sampled_token_ids.shape) != expected[:2]:
        raise ValueError(
            "sampled_token_ids must match the first two logits dimensions, "
            f"got {tuple(sampled_token_ids.shape)} and {expected[:2]}."
        )
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")

    detached = {name: logits.detach() for name, logits in distributions.items()}
    k = min(int(top_k), expected[-1])
    sampled = sampled_token_ids.detach().to(device=student_logits.device, dtype=torch.long)
    if view_availability is None:
        availability = torch.ones(
            (expected[0], len(MESD_PRIVILEGED_VIEW_NAMES)),
            dtype=torch.bool,
            device=student_logits.device,
        )
    else:
        expected_availability_shape = (expected[0], len(MESD_PRIVILEGED_VIEW_NAMES))
        if tuple(view_availability.shape) != expected_availability_shape:
            raise ValueError(
                "view_availability must have shape "
                f"{expected_availability_shape} in (A, T, S) order, "
                f"got {tuple(view_availability.shape)}."
            )
        availability = view_availability.detach().to(device=student_logits.device) > 0

    candidates = [torch.topk(detached["student"], k=k, dim=-1).indices]
    for view_index, name in enumerate(MESD_PRIVILEGED_VIEW_NAMES):
        view_ids = torch.topk(detached[name], k=k, dim=-1).indices
        # Replacing an unavailable view with the already-mandatory sampled id
        # preserves fixed tensor shapes without admitting any of its Top-K ids.
        view_ids = torch.where(
            availability[:, view_index].view(expected[0], 1, 1),
            view_ids,
            sampled.unsqueeze(-1).expand_as(view_ids),
        )
        candidates.append(view_ids)
    candidates.append(sampled.unsqueeze(-1))
    support_ids = torch.sort(torch.cat(candidates, dim=-1), dim=-1).values
    support_mask = torch.ones_like(support_ids, dtype=torch.bool)
    support_mask[..., 1:] = support_ids[..., 1:] != support_ids[..., :-1]

    log_probs: dict[str, torch.Tensor] = {}
    for name, logits in detached.items():
        selected = torch.gather(logits, dim=-1, index=support_ids)
        selected = selected.masked_fill(~support_mask, float("-inf"))
        _require_finite_tensor(f"{name}_support_logits", selected, valid_mask=support_mask)
        log_probs[name] = selected - torch.logsumexp(selected, dim=-1, keepdim=True)

    sampled_matches = support_mask & support_ids.eq(sampled.unsqueeze(-1))
    if not bool(sampled_matches.any(dim=-1).all().item()):
        raise RuntimeError("The realized token is missing from the common MeSD support.")
    sampled_support_index = sampled_matches.to(torch.long).argmax(dim=-1)
    return log_probs, {
        "support_ids": support_ids.detach(),
        "support_mask": support_mask.detach(),
        "sampled_support_index": sampled_support_index.detach(),
        "view_availability": availability.detach(),
    }


def compute_mesd_answer_anchored_avsd_log_probs(
    *,
    answer_log_probs: torch.Tensor,
    temporal_log_probs: torch.Tensor,
    spatial_log_probs: torch.Tensor,
    view_availability: Optional[torch.Tensor] = None,
    token_role_mask: Optional[torch.Tensor] = None,
    support_mask: Optional[torch.Tensor] = None,
    evidence_alpha: float = 1.0,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fuse answer-anchored likelihood lifts with AVSD-inspired aggregation.

    ``view_availability`` has shape ``[B, 2]`` in ``(T, S)`` order.  An
    optional role mask may have shape ``[B, R, 2]`` (or ``[B, 2]`` for
    broadcast routing).  At ``evidence_alpha=1``, the function has exact
    M=0/1/2 degeneracies on the supplied common support; alpha zero recovers
    the answer anchor.  Every input is detached before doing arithmetic.
    """

    expected = tuple(answer_log_probs.shape)
    if answer_log_probs.ndim != 3:
        raise ValueError("Teacher log probabilities must have shape [batch, response, support].")
    for name, value in {
        "temporal_log_probs": temporal_log_probs,
        "spatial_log_probs": spatial_log_probs,
    }.items():
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}.")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")
    if not math.isfinite(float(evidence_alpha)) or not 0.0 <= float(evidence_alpha) <= 1.0:
        raise ValueError(f"evidence_alpha must be in [0, 1], got {evidence_alpha}.")

    answer = answer_log_probs.detach()
    temporal = temporal_log_probs.detach()
    spatial = spatial_log_probs.detach()
    batch, response, _ = expected
    dtype, device = answer.dtype, answer.device

    if support_mask is None:
        valid_support = torch.ones(expected, dtype=torch.bool, device=device)
    else:
        if tuple(support_mask.shape) != expected:
            raise ValueError(f"support_mask must have shape {expected}, got {tuple(support_mask.shape)}.")
        valid_support = support_mask.detach().to(device=device, dtype=torch.bool)
    if not bool(valid_support.any(dim=-1).all().item()):
        raise ValueError("Every MeSD response token must have a non-empty support.")
    for name, value in {
        "answer_log_probs": answer,
        "temporal_log_probs": temporal,
        "spatial_log_probs": spatial,
    }.items():
        _require_finite_tensor(name, value, valid_mask=valid_support)

    if view_availability is None:
        availability = torch.ones((batch, 2), dtype=torch.bool, device=device)
    else:
        if tuple(view_availability.shape) != (batch, 2):
            raise ValueError(f"view_availability must have shape {(batch, 2)}, got {tuple(view_availability.shape)}.")
        availability = view_availability.detach().to(device=device) > 0

    if token_role_mask is None:
        roles = torch.ones((batch, response, 2), dtype=torch.bool, device=device)
    else:
        role_value = token_role_mask.detach().to(device=device)
        if tuple(role_value.shape) == (batch, 2):
            role_value = role_value.unsqueeze(1).expand(batch, response, 2)
        if tuple(role_value.shape) != (batch, response, 2):
            raise ValueError(
                "token_role_mask must have shape "
                f"{(batch, response, 2)} or {(batch, 2)}, got {tuple(token_role_mask.shape)}."
            )
        roles = role_value > 0

    active = availability.unsqueeze(1) & roles
    effective_view_count = active.sum(dim=-1)
    has_specialist = effective_view_count > 0
    count = effective_view_count.to(dtype=dtype).clamp_min(1.0)
    weights = active.to(dtype=dtype) / count.unsqueeze(-1)

    deltas = torch.stack(
        [
            torch.where(valid_support, temporal - answer, torch.zeros_like(answer)),
            torch.where(valid_support, spatial - answer, torch.zeros_like(answer)),
        ],
        dim=2,
    )
    geometric = (weights.unsqueeze(-1) * deltas).sum(dim=2)
    log_weights = torch.where(
        active,
        -torch.log(count).unsqueeze(-1).expand_as(weights),
        torch.full_like(weights, float("-inf")),
    )
    arithmetic_log_lift_raw = torch.logsumexp(deltas + log_weights.unsqueeze(-1), dim=2)
    arithmetic_log_lift = torch.where(has_specialist.unsqueeze(-1), arithmetic_log_lift_raw, geometric)
    residual = (arithmetic_log_lift - geometric).clamp_min(0.0)

    average_abs = (weights.unsqueeze(-1) * deltas.abs()).sum(dim=2)
    alignment = geometric.abs() / (average_abs + float(epsilon))
    magnitude = geometric.abs() / (geometric.abs() + residual + float(epsilon))
    residual_gate = alignment * magnitude
    fused_lift = geometric + residual_gate * residual

    specialist_mask = has_specialist.unsqueeze(-1) & valid_support
    geometric = torch.where(specialist_mask, geometric, torch.zeros_like(geometric))
    residual = torch.where(specialist_mask, residual, torch.zeros_like(residual))
    alignment = torch.where(specialist_mask, alignment, torch.zeros_like(alignment))
    magnitude = torch.where(specialist_mask, magnitude, torch.zeros_like(magnitude))
    residual_gate = torch.where(specialist_mask, residual_gate, torch.zeros_like(residual_gate))
    fused_lift = torch.where(specialist_mask, fused_lift, torch.zeros_like(fused_lift))

    # Evidence scaling belongs only to target reconstruction.  In particular,
    # it must not change AVSD's C/R/lambda calculation or its residual bound.
    fused_logits = (answer + float(evidence_alpha) * fused_lift).masked_fill(~valid_support, float("-inf"))
    fused_log_probs = fused_logits - torch.logsumexp(fused_logits, dim=-1, keepdim=True)
    _require_finite_tensor("fused_teacher_log_probs", fused_log_probs, valid_mask=valid_support)
    return fused_log_probs.detach(), {
        "effective_view_count": effective_view_count.detach(),
        "geometric": geometric.detach(),
        "residual": residual.detach(),
        "alignment": alignment.detach(),
        "magnitude": magnitude.detach(),
        "residual_gate": residual_gate.detach(),
        "fused_lift": fused_lift.detach(),
    }


def build_mesd_answer_anchored_avsd_gap(
    *,
    student_logits: torch.Tensor,
    answer_teacher_logits: torch.Tensor,
    temporal_teacher_logits: torch.Tensor,
    spatial_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    view_availability: Optional[torch.Tensor] = None,
    token_role_mask: Optional[torch.Tensor] = None,
    response_mask: Optional[torch.Tensor] = None,
    top_k: int = 16,
    evidence_alpha: float = 1.0,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the sampled-token gap from task-aware three-view AVSD.

    ``view_availability`` is ``[B, 3]`` in ``(A, T, S)`` order.  When A is
    available, T/S likelihood lifts are fused relative to that answer anchor.
    Without A, a single routed specialist is used exactly (a pure T or S
    bypass); two routed specialists use their normalized geometric consensus.
    If neither the answer nor a routed specialist is available, the gap is
    exactly zero.
    """

    batch, response = tuple(student_logits.shape[:2])
    if view_availability is None:
        full_availability = torch.ones(
            (batch, len(MESD_PRIVILEGED_VIEW_NAMES)),
            dtype=torch.bool,
            device=student_logits.device,
        )
    else:
        expected_availability_shape = (batch, len(MESD_PRIVILEGED_VIEW_NAMES))
        if tuple(view_availability.shape) != expected_availability_shape:
            raise ValueError(
                "view_availability must have shape "
                f"{expected_availability_shape} in (A, T, S) order, "
                f"got {tuple(view_availability.shape)}."
            )
        full_availability = view_availability.detach().to(device=student_logits.device) > 0

    log_probs, support = compute_mesd_common_support_log_probs(
        student_logits=student_logits,
        answer_teacher_logits=answer_teacher_logits,
        temporal_teacher_logits=temporal_teacher_logits,
        spatial_teacher_logits=spatial_teacher_logits,
        sampled_token_ids=sampled_token_ids,
        view_availability=full_availability,
        top_k=top_k,
    )
    fused_log_probs, avsd = compute_mesd_answer_anchored_avsd_log_probs(
        answer_log_probs=log_probs["answer"],
        temporal_log_probs=log_probs["temporal"],
        spatial_log_probs=log_probs["spatial"],
        view_availability=full_availability[:, 1:],
        token_role_mask=token_role_mask,
        support_mask=support["support_mask"],
        evidence_alpha=evidence_alpha,
        epsilon=epsilon,
    )

    if token_role_mask is None:
        roles = torch.ones((batch, response, 2), dtype=torch.bool, device=student_logits.device)
    else:
        role_value = token_role_mask.detach().to(device=student_logits.device)
        if tuple(role_value.shape) == (batch, 2):
            role_value = role_value.unsqueeze(1).expand(batch, response, 2)
        if tuple(role_value.shape) != (batch, response, 2):
            raise ValueError(
                "token_role_mask must have shape "
                f"{(batch, response, 2)} or {(batch, 2)}, got {tuple(token_role_mask.shape)}."
            )
        roles = role_value > 0

    active_specialists = full_availability[:, None, 1:] & roles
    specialist_count = active_specialists.sum(dim=-1)
    specialist_weights = active_specialists.to(log_probs["student"].dtype) / (
        specialist_count.clamp_min(1).to(log_probs["student"].dtype).unsqueeze(-1)
    )
    direct_logits = (
        specialist_weights[..., 0, None] * log_probs["temporal"]
        + specialist_weights[..., 1, None] * log_probs["spatial"]
    ).masked_fill(~support["support_mask"], float("-inf"))
    direct_log_probs = direct_logits - torch.logsumexp(direct_logits, dim=-1, keepdim=True)

    answer_available = full_availability[:, 0].view(batch, 1, 1)
    has_specialist = specialist_count.gt(0).unsqueeze(-1)
    # Student is a numerically well-defined placeholder for M=0. The final
    # gap is also set explicitly to zero below, so no unavailable teacher can
    # create supervision through this branch.
    unanchored_log_probs = torch.where(has_specialist, direct_log_probs, log_probs["student"])
    selected_log_probs = torch.where(answer_available, fused_log_probs, unanchored_log_probs)

    sampled_index = support["sampled_support_index"].unsqueeze(-1)
    fused_sampled_logp = torch.gather(selected_log_probs, dim=-1, index=sampled_index).squeeze(-1)
    student_sampled_logp = torch.gather(log_probs["student"], dim=-1, index=sampled_index).squeeze(-1)
    no_teacher_evidence = (~full_availability[:, 0].unsqueeze(-1)) & specialist_count.eq(0)
    teacher_gap = torch.where(
        no_teacher_evidence,
        torch.zeros_like(student_sampled_logp),
        fused_sampled_logp - student_sampled_logp,
    ).detach()
    _require_finite_tensor("fused_teacher_gap", teacher_gap)

    batch_response = teacher_gap.shape
    if response_mask is None:
        token_mask = torch.ones(batch_response, dtype=teacher_gap.dtype, device=teacher_gap.device)
    else:
        if tuple(response_mask.shape) != tuple(batch_response):
            raise ValueError(
                f"response_mask must have shape {tuple(batch_response)}, got {tuple(response_mask.shape)}."
            )
        token_mask = response_mask.detach().to(device=teacher_gap.device, dtype=teacher_gap.dtype)
    token_denom = token_mask.sum().clamp_min(1.0)
    view_count = specialist_count
    response_cells = token_mask.to(torch.bool)
    support_cells = support["support_mask"] & response_cells.unsqueeze(-1)
    anchored_support_cells = support_cells & full_availability[:, 0].view(batch, 1, 1)
    anchored_support_denom = anchored_support_cells.sum().clamp_min(1)

    def _support_mean(value: torch.Tensor) -> float:
        return float((value * anchored_support_cells.to(value.dtype)).sum().div(anchored_support_denom).item())

    bound_excess = (avsd["residual_gate"] * avsd["residual"] - avsd["geometric"].abs()).clamp_min(0.0)
    metrics = {
        "mesd/avsd_m0_fraction": float(
            (((view_count == 0).to(token_mask.dtype) * token_mask).sum() / token_denom).item()
        ),
        "mesd/avsd_m1_fraction": float(
            (((view_count == 1).to(token_mask.dtype) * token_mask).sum() / token_denom).item()
        ),
        "mesd/avsd_m2_fraction": float(
            (((view_count == 2).to(token_mask.dtype) * token_mask).sum() / token_denom).item()
        ),
        "mesd/avsd_residual_mean": _support_mean(avsd["residual"]),
        "mesd/avsd_alignment_mean": _support_mean(avsd["alignment"]),
        "mesd/avsd_gate_mean": _support_mean(avsd["residual_gate"]),
        "mesd/avsd_bound_excess_max": float(
            bound_excess.masked_fill(~anchored_support_cells, 0.0).max().item()
        ),
        "mesd/avsd_answer_anchor_fraction": float(
            (full_availability[:, 0].to(token_mask.dtype).unsqueeze(-1) * token_mask).sum().div(token_denom).item()
        ),
        "mesd/avsd_unanchored_bypass_fraction": float(
            (
                ((~full_availability[:, 0].unsqueeze(-1)) & specialist_count.gt(0))
                .to(token_mask.dtype)
                .mul(token_mask)
                .sum()
                .div(token_denom)
                .item()
            )
        ),
        "mesd/avsd_no_evidence_fraction": float(
            (no_teacher_evidence.to(token_mask.dtype).mul(token_mask).sum().div(token_denom).item())
        ),
        "mesd/common_support_size_mean": float(
            (support["support_mask"].sum(dim=-1).to(token_mask.dtype) * token_mask).sum().div(token_denom).item()
        ),
    }
    return teacher_gap, metrics


def _coarse_distribution_from_logits(
    *,
    logits: torch.Tensor,
    support_ids: torch.Tensor,
    support_mask: torch.Tensor,
    keep_gradient: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project a full-vocabulary distribution onto common top-K plus tail.

    The full-vocabulary normalizer is retained.  Duplicate support ids are
    masked, and the final category stores all probability outside the explicit
    union.  The live Student path is never detached or probability-floored.
    """

    source = logits if keep_gradient else logits.detach()
    source = source.float()
    selected_logits = torch.gather(source, dim=-1, index=support_ids)
    full_log_normalizer = torch.logsumexp(source, dim=-1, keepdim=True)
    selected_log_probs = (selected_logits - full_log_normalizer).masked_fill(~support_mask, float("-inf"))
    selected_probs = torch.where(support_mask, selected_log_probs.exp(), torch.zeros_like(selected_log_probs))
    tail_probs = 1.0 - selected_probs.sum(dim=-1, keepdim=True)
    numerical_tolerance = 32.0 * torch.finfo(selected_probs.dtype).eps
    if bool((tail_probs < -numerical_tolerance).any().item()):
        raise FloatingPointError("Common-support probability mass exceeds one.")
    # This clamp only removes round-off below zero.  It is not a probability
    # floor and therefore does not alter ordinary low-probability Student mass.
    tail_probs = tail_probs.clamp_min(0.0)
    coarse_mask = torch.cat(
        [
            support_mask,
            torch.ones(
                (*support_mask.shape[:-1], 1),
                dtype=torch.bool,
                device=support_mask.device,
            ),
        ],
        dim=-1,
    )
    probabilities = torch.cat([selected_probs, tail_probs], dim=-1)
    if not keep_gradient:
        # Detached teacher paths need finite logs for AVSD arithmetic.  This is
        # only an underflow guard; the configured target floor is applied later
        # to q_mv/reference on the finished coarse alphabet.
        probabilities = torch.where(
            coarse_mask,
            probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny),
            torch.zeros_like(probabilities),
        )
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )
    positive_support = coarse_mask & (probabilities > 0)
    # Never evaluate log(0), even on a later-masked branch: autograd can turn
    # that false branch into 0/0 during backward.  Replacing non-positive
    # entries by one is not a Student probability floor: their exact zero mass
    # is restored as -inf below and excluded from the KL expectation.
    safe_probabilities = torch.where(positive_support, probabilities, torch.ones_like(probabilities))
    log_probabilities = torch.log(safe_probabilities)
    log_probabilities = log_probabilities.masked_fill(~positive_support, float("-inf"))
    _require_finite_tensor("coarse_distribution_probabilities", probabilities, valid_mask=coarse_mask)
    return probabilities, log_probabilities


def _floor_stop_gradient_target(
    log_probs: torch.Tensor,
    *,
    support_mask: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Apply target-only floor and renormalization on the coarse alphabet."""

    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError(f"target epsilon must be finite and positive, got {epsilon}.")
    target_probs = torch.where(
        support_mask,
        log_probs.detach().float().exp(),
        torch.zeros_like(log_probs, dtype=torch.float32),
    )
    floored = torch.where(
        support_mask,
        target_probs.clamp_min(float(epsilon)),
        torch.zeros_like(target_probs),
    )
    floored = floored / floored.sum(dim=-1, keepdim=True).clamp_min(float(epsilon))
    return torch.where(
        support_mask,
        torch.log(floored),
        torch.full_like(floored, float("-inf")),
    ).detach()


def build_mesd_multiview_direction_token_loss(
    *,
    student_logits: torch.Tensor,
    answer_teacher_logits: torch.Tensor,
    temporal_teacher_logits: torch.Tensor,
    spatial_teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    view_availability: torch.Tensor,
    response_mask: torch.Tensor,
    direction_mask: torch.Tensor,
    top_k: int = 32,
    evidence_alpha: float = 1.0,
    avsd_epsilon: float = 1e-06,
    target_epsilon: float = 1e-08,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Build Student-to-multiview-teacher reverse KL on a common top-K+tail alphabet. Only Student probabilities retain gradients."""
    expected = tuple(student_logits.shape)
    if student_logits.ndim != 3:
        raise ValueError("Direction-control logits must have shape [batch, response, vocabulary].")
    teacher_logits = {
        "answer": answer_teacher_logits,
        "temporal": temporal_teacher_logits,
        "spatial": spatial_teacher_logits,
    }
    for name, value in teacher_logits.items():
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} teacher logits must have shape {expected}.")
    if tuple(sampled_token_ids.shape) != expected[:2]:
        raise ValueError("sampled_token_ids must have shape [batch, response].")
    batch, response, vocabulary = expected
    if tuple(view_availability.shape) != (batch, len(MESD_PRIVILEGED_VIEW_NAMES)):
        raise ValueError("view_availability must have shape [batch, 3].")
    if tuple(response_mask.shape) != (batch, response):
        raise ValueError("response_mask must have shape [batch, response].")
    if tuple(direction_mask.shape) != (batch,):
        raise ValueError("direction_mask must have shape [batch].")
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")
    k = min(int(top_k), int(vocabulary))
    sampled = sampled_token_ids.detach().to(device=student_logits.device, dtype=torch.long)
    availability = view_availability.detach().to(device=student_logits.device) > 0
    routed = direction_mask.detach().to(device=student_logits.device) > 0
    if bool((routed & ~availability.any(dim=-1)).any().item()):
        raise ValueError("Every direction-routed row must have at least one privileged teacher view.")
    candidates = [torch.topk(student_logits.detach(), k=k, dim=-1).indices]
    for view_index, view_name in enumerate(MESD_PRIVILEGED_VIEW_NAMES):
        view_ids = torch.topk(teacher_logits[view_name].detach(), k=k, dim=-1).indices
        view_ids = torch.where(
            availability[:, view_index].view(batch, 1, 1), view_ids, sampled.unsqueeze(-1).expand_as(view_ids)
        )
        candidates.append(view_ids)
    candidates.append(sampled.unsqueeze(-1))
    support_ids = torch.sort(torch.cat(candidates, dim=-1), dim=-1).values
    support_mask = torch.ones_like(support_ids, dtype=torch.bool)
    support_mask[..., 1:] = support_ids[..., 1:] != support_ids[..., :-1]
    coarse_mask = torch.cat(
        [support_mask, torch.ones((*support_mask.shape[:-1], 1), dtype=torch.bool, device=support_mask.device)], dim=-1
    )
    student_probs, student_log_probs = _coarse_distribution_from_logits(
        logits=student_logits, support_ids=support_ids, support_mask=support_mask, keep_gradient=True
    )
    teacher_log_probs: dict[str, torch.Tensor] = {}
    for name, value in teacher_logits.items():
        _, teacher_log_probs[name] = _coarse_distribution_from_logits(
            logits=value, support_ids=support_ids, support_mask=support_mask, keep_gradient=False
        )
    fused_log_probs, avsd = compute_mesd_answer_anchored_avsd_log_probs(
        answer_log_probs=teacher_log_probs["answer"],
        temporal_log_probs=teacher_log_probs["temporal"],
        spatial_log_probs=teacher_log_probs["spatial"],
        view_availability=availability[:, 1:],
        support_mask=coarse_mask,
        evidence_alpha=evidence_alpha,
        epsilon=avsd_epsilon,
    )
    specialist_count = availability[:, 1:].sum(dim=-1).view(batch, 1)
    specialist_weights = availability[:, 1:].to(torch.float32) / specialist_count.clamp_min(1).to(torch.float32)
    temporal_safe = torch.where(
        coarse_mask, teacher_log_probs["temporal"], torch.zeros_like(teacher_log_probs["temporal"])
    )
    spatial_safe = torch.where(
        coarse_mask, teacher_log_probs["spatial"], torch.zeros_like(teacher_log_probs["spatial"])
    )
    direct_logits = (
        specialist_weights[:, None, 0, None] * temporal_safe + specialist_weights[:, None, 1, None] * spatial_safe
    ).masked_fill(~coarse_mask, float("-inf"))
    direct_log_probs = direct_logits - torch.logsumexp(direct_logits, dim=-1, keepdim=True)
    answer_available = availability[:, 0].view(batch, 1, 1)
    has_specialist = specialist_count.gt(0).view(batch, 1, 1)
    unanchored = torch.where(has_specialist, direct_log_probs, student_log_probs.detach())
    multiview_log_probs = torch.where(answer_available, fused_log_probs, unanchored).detach()
    multiview_target = _floor_stop_gradient_target(
        multiview_log_probs, support_mask=coarse_mask, epsilon=target_epsilon
    )
    target_log_probs = multiview_target
    positive_student = coarse_mask & (student_probs > 0)
    safe_student_log_probs = torch.where(positive_student, student_log_probs, torch.zeros_like(student_log_probs))
    safe_target_log_probs = torch.where(coarse_mask, target_log_probs, torch.zeros_like(target_log_probs))
    kl_terms = torch.where(
        positive_student,
        student_probs * (safe_student_log_probs - safe_target_log_probs),
        torch.zeros_like(student_probs),
    )
    token_kl = kl_terms.sum(dim=-1)
    _require_finite_tensor("failure_direction_token_kl", token_kl)
    token_mask = response_mask.to(dtype=torch.float32) * direction_mask.to(
        device=response_mask.device, dtype=torch.float32
    ).unsqueeze(-1)
    denom = token_mask.sum().clamp_min(1.0)
    support_size = coarse_mask.sum(dim=-1).to(torch.float32)
    metrics = {
        "mesd/direction_kl_mean": float((token_kl.detach() * token_mask).sum().div(denom).item()),
        "mesd/direction_support_size_mean": float((support_size * token_mask).sum().div(denom).item()),
        "mesd/direction_student_tail_mean": float(
            (student_probs[..., -1].detach() * token_mask).sum().div(denom).item()
        ),
        "mesd/direction_target_tail_mean": float(
            (target_log_probs[..., -1].exp() * token_mask).sum().div(denom).item()
        ),
        "mesd/direction_avsd_gate_mean": float(
            (avsd["residual_gate"].mean(dim=-1).detach() * token_mask).sum().div(denom).item()
        ),
    }
    return (token_kl, metrics)


def apply_mesd_advantage_reweighting(
    *,
    advantages: torch.Tensor,
    teacher_gap: torch.Tensor,
    response_mask: torch.Tensor,
    mixing_lambda: float,
    weight_clip: Optional[float] = 0.2,
    target_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Apply MeSD/RLSD direction-magnitude credit reweighting."""

    expected = tuple(advantages.shape)
    for name, value in {"teacher_gap": teacher_gap, "response_mask": response_mask}.items():
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}.")

    dtype, device = advantages.dtype, advantages.device
    if not advantages.is_floating_point():
        raise TypeError("advantages must be a floating-point tensor.")
    _require_finite_tensor("advantages", advantages)
    response_mask = response_mask.to(device=device, dtype=dtype)
    teacher_gap = teacher_gap.detach().to(device=device, dtype=dtype)
    _require_finite_tensor("response_mask", response_mask)
    _require_finite_tensor("teacher_gap", teacher_gap)
    if target_mask is None:
        token_mask = response_mask
    else:
        if target_mask.ndim != 1 or target_mask.size(0) != advantages.size(0):
            raise ValueError("target_mask must have shape [batch].")
        token_mask = response_mask * target_mask.to(device=device, dtype=dtype).unsqueeze(-1)

    signed_gap = torch.sign(advantages.detach()) * teacher_gap
    # Bound the exponent in the destination dtype.  This does not affect the
    # configured clipped objective, but also keeps the optional unclipped mode
    # strictly positive and finite under extreme teacher gaps.
    exponent_limit = min(60.0, math.log(torch.finfo(dtype).max) - 2.0)
    raw_weights = torch.exp(signed_gap.clamp(-exponent_limit, exponent_limit))
    if weight_clip is None:
        clipped_weights = raw_weights
    else:
        clip = float(weight_clip)
        if not math.isfinite(clip) or clip < 0:
            raise ValueError(f"weight_clip must be non-negative, got {clip}.")
        clipped_weights = raw_weights.clamp(min=1.0 - clip, max=1.0 + clip)

    if not math.isfinite(float(mixing_lambda)):
        raise ValueError(f"mixing_lambda must be finite, got {mixing_lambda}.")
    mixing = min(max(float(mixing_lambda), 0.0), 1.0)
    selected_weights = (1.0 - mixing) + mixing * clipped_weights
    effective_weights = (torch.ones_like(advantages) + token_mask * (selected_weights - 1.0)).detach()
    _require_finite_tensor("effective_mesd_weights", effective_weights)
    if not bool((effective_weights > 0).all().item()):
        raise FloatingPointError("MeSD effective weights must remain strictly positive.")
    reweighted = advantages * effective_weights

    denom = token_mask.sum()
    if denom.item() > 0:
        weight_mean = float(((effective_weights * token_mask).sum() / denom).detach().item())
        clipped_fraction = float(
            (((raw_weights != clipped_weights).to(dtype) * token_mask).sum() / denom).detach().item()
        )
    else:
        weight_mean = 1.0
        clipped_fraction = 0.0
    return (
        reweighted,
        effective_weights,
        {
            "mesd/weight_mean": weight_mean,
            "mesd/weight_clipped_fraction": clipped_fraction,
            "mesd/effective_token_fraction": float(
                (token_mask.sum() / response_mask.sum().clamp(min=1.0)).detach().item()
            ),
        },
    )


__all__ = [
    "QWEN_ASSISTANT_MARKER",
    "MESD_PRIVILEGED_VIEW_NAMES",
    "apply_mesd_advantage_reweighting",
    "build_mesd_answer_anchored_avsd_gap",
    "build_mesd_multiview_direction_token_loss",
    "build_mesd_privileged_views",
    "build_mesd_view_hint",
    "compute_mesd_answer_anchored_avsd_log_probs",
    "compute_mesd_common_support_log_probs",
    "compute_mesd_mixing_lambda",
    "inject_mesd_hindsight_hint",
    "mesd_teacher_enabled",
    "mesd_view_available",
]
