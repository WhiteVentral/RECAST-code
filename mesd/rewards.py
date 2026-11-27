"""RLSD batch-reward adapter for the seven released Open-o3/MeSD rewards."""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .schema import SUPPORTED_TASKS


REWARD_COMPONENTS = (
    ("ans_acc_reward", "ans_acc_reward", True),
    ("ans_tiou_reward", "ans_tiou_reward", True),
    ("ans_viou_reward", "ans_viou_reward", True),
    ("thk_temporal_point_reward", "thk_temporal_point_reward", False),
    ("thk_temporal_segment_reward", "thk_temporal_segment_reward", False),
    ("thk_spatial_reward", "thk_spatial_reward", False),
    ("format_reward", "format_reward", False),
)

OFFICIAL_SPATIAL_OPTIONS = {
    "spatial_iou_mode": "avg",
    "identity_match_mode": "soft",
    "spatial_norm_mode": "matched",
    "correct_tempgate": True,
}

_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_TIME_PATTERN = re.compile(
    r"<t>\s*(\d+(?:\.\d+)?)\s*</t>s\s+to\s+<t>\s*(\d+(?:\.\d+)?)\s*</t>s",
    re.IGNORECASE,
)
_CHOICE_PATTERN = re.compile(r"^\s*[\[(]?\s*([A-Ha-h])\s*[\])\.]?\s*$")
_DIRECTION_TASKS = frozenset(
    {"General video QA MCQ", "temporal QA", "temporal QA (MCQ)"}
)


class RewardMetadataError(ValueError):
    """Raised instead of silently assigning an invalid sample a zero reward."""


def _unknown_outcome(*, row_valid: bool, task_eligible: bool = False) -> dict[str, float]:
    return {
        "outcome_success": 0.0,
        "outcome_failure": 0.0,
        "outcome_unknown": 1.0,
        "outcome_row_valid": float(row_valid),
        "outcome_rkl_eligible": 0.0,
        "outcome_task_eligible": float(task_eligible),
        "outcome_option_valid": 0.0,
        "outcome_time_valid": 0.0,
    }


def _extract_answer_text(response: Any) -> str | None:
    matches = list(_ANSWER_PATTERN.finditer(str(response or "")))
    if len(matches) != 1:
        return None
    text = matches[0].group(1).strip()
    return text or None


def _parse_choice(value: Any) -> str | None:
    text = str(value or "").strip()
    match = _CHOICE_PATTERN.fullmatch(text)
    return match.group(1).upper() if match else None


def _parse_temporal_answer(value: Any) -> tuple[float, float] | None:
    matches = list(_TIME_PATTERN.finditer(str(value or "")))
    if len(matches) != 1:
        return None
    start, end = float(matches[0].group(1)), float(matches[0].group(2))
    if not (math.isfinite(start) and math.isfinite(end) and end > start):
        return None
    return start, end


def _parse_ground_truth_window(task: str, answer: Any) -> tuple[float, float] | None:
    text = str(answer or "").strip()
    if task == "temporal QA (MCQ)":
        parts = text.splitlines()
        if len(parts) < 2:
            return None
        text = parts[1].strip()
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 2:
        return None
    try:
        start, end = float(parsed[0]), float(parsed[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(end) and end > start):
        return None
    return start, end


def _interval_iou(
    first: tuple[float, float], second: tuple[float, float]
) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0.0 else 0.0


def classify_mesd_outcome(
    *,
    task: Any,
    answer: Any,
    response: Any,
    success_iou_threshold: float = 0.5,
    failure_iou_threshold: float = 0.0,
    temporal_mcq_strict_rkl: bool = True,
) -> dict[str, float]:
    """Return conservative S/F/U routing signals independent of GRPO advantage.

    V1 only hard-verifies General MCQ, temporal localization, and temporal
    MCQ.  Malformed model answers are kept as valid-but-unknown trajectories;
    invalid ground truth is a row-validity failure.  The strict temporal-MCQ
    RKL gate requires both an explicitly wrong option and zero temporal IoU,
    even though either verified component is enough to mark the task outcome F.
    """

    task_name = str(task or "")
    task_eligible = task_name in _DIRECTION_TASKS
    if not task_eligible:
        return _unknown_outcome(row_valid=True, task_eligible=False)
    if not (
        math.isfinite(float(success_iou_threshold))
        and math.isfinite(float(failure_iou_threshold))
        and 0.0 <= float(failure_iou_threshold) < float(success_iou_threshold) <= 1.0
    ):
        raise ValueError(
            "Outcome IoU thresholds must satisfy 0 <= failure < success <= 1, "
            f"got failure={failure_iou_threshold}, success={success_iou_threshold}."
        )

    answer_text = _extract_answer_text(response)
    if task_name == "General video QA MCQ":
        gt_choice = _parse_choice(str(answer or "").splitlines()[0])
        if gt_choice is None:
            return _unknown_outcome(row_valid=False, task_eligible=True)
        predicted_choice = _parse_choice(answer_text)
        if predicted_choice is None:
            return _unknown_outcome(row_valid=True, task_eligible=True)
        success = predicted_choice == gt_choice
        return {
            "outcome_success": float(success),
            "outcome_failure": float(not success),
            "outcome_unknown": 0.0,
            "outcome_row_valid": 1.0,
            "outcome_rkl_eligible": float(not success),
            "outcome_task_eligible": 1.0,
            "outcome_option_valid": 1.0,
            "outcome_time_valid": 0.0,
        }

    gt_window = _parse_ground_truth_window(task_name, answer)
    if gt_window is None:
        return _unknown_outcome(row_valid=False, task_eligible=True)
    predicted_window = _parse_temporal_answer(answer_text)
    time_valid = predicted_window is not None
    tiou = _interval_iou(predicted_window, gt_window) if time_valid else 0.0
    time_success = time_valid and tiou >= float(success_iou_threshold)
    time_failure = time_valid and tiou <= float(failure_iou_threshold)

    if task_name == "temporal QA":
        if time_success:
            status = "success"
        elif time_failure:
            status = "failure"
        else:
            status = "unknown"
        return {
            "outcome_success": float(status == "success"),
            "outcome_failure": float(status == "failure"),
            "outcome_unknown": float(status == "unknown"),
            "outcome_row_valid": 1.0,
            "outcome_rkl_eligible": float(status == "failure"),
            "outcome_task_eligible": 1.0,
            "outcome_option_valid": 0.0,
            "outcome_time_valid": float(time_valid),
        }

    gt_choice = _parse_choice(str(answer or "").splitlines()[0])
    if gt_choice is None:
        return _unknown_outcome(row_valid=False, task_eligible=True)
    option_text = None
    if answer_text is not None:
        option_matches = re.findall(
            r"Correct\s+Option\s*:\s*([^\n\r]+)", answer_text, re.IGNORECASE
        )
        if len(option_matches) == 1:
            option_text = option_matches[0].strip()
    predicted_choice = _parse_choice(option_text)
    option_valid = predicted_choice is not None
    option_success = option_valid and predicted_choice == gt_choice
    option_failure = option_valid and predicted_choice != gt_choice

    success = option_success and time_success
    failure = option_failure or time_failure
    unknown = not success and not failure
    if temporal_mcq_strict_rkl:
        rkl_eligible = option_failure and time_failure
    else:
        rkl_eligible = failure
    return {
        "outcome_success": float(success),
        "outcome_failure": float(failure),
        "outcome_unknown": float(unknown),
        "outcome_row_valid": 1.0,
        "outcome_rkl_eligible": float(rkl_eligible),
        "outcome_task_eligible": 1.0,
        "outcome_option_valid": float(option_valid),
        "outcome_time_valid": float(time_valid),
    }


def _decode_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(decoded, Mapping):
            return dict(decoded)
    return {}


def _flatten_metadata(value: Any) -> dict[str, Any]:
    source = _decode_mapping(value)
    result: dict[str, Any] = {}
    for key in ("extra_info", "metadata", "reward_metadata"):
        if key in source:
            result.update(_flatten_metadata(source[key]))
    result.update({key: item for key, item in source.items() if key not in {"extra_info", "metadata", "reward_metadata"}})
    return result


def _drop_none_values(value: Any) -> Any:
    """Undo Arrow's union-struct materialization for dynamic STGR key_items."""

    if isinstance(value, Mapping):
        return {
            key: _drop_none_values(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_drop_none_values(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return tuple(_drop_none_values(item) for item in value if item is not None)
    return value


def extract_reward_metadata(reward_input: Mapping[str, Any]) -> dict[str, Any]:
    """Accept direct columns or a JSON/dict ground-truth payload.

    The stock RLSD reward manager originally forwards only ``ground_truth``.
    Encoding ``PreparedSTGRSample.reward_metadata`` into that field therefore
    remains supported, while an enhanced manager can forward the same keys as
    direct columns without changing this adapter.
    """

    metadata: dict[str, Any] = {}
    ground_truth = reward_input.get("ground_truth")
    decoded_ground_truth = _flatten_metadata(ground_truth)
    if decoded_ground_truth:
        metadata.update(decoded_ground_truth)
    elif ground_truth is not None:
        metadata["answer"] = ground_truth

    for key in ("extra_info", "metadata", "reward_metadata"):
        metadata.update(_flatten_metadata(reward_input.get(key)))

    direct_keys = {
        "id",
        "source",
        "task",
        "answer",
        "image_size",
        "image_size_refine",
        "annotation_image_size",
        "processed_image_size",
        "key_frames",
        "key_items",
        "step_percent",
    }
    for key in direct_keys:
        if key in reward_input and reward_input[key] is not None:
            metadata[key] = reward_input[key]
    return metadata


def serialize_reward_metadata(value: Any) -> str:
    """Serialize a prepared sample or metadata mapping for RLSD ground_truth."""

    metadata = getattr(value, "reward_metadata", value)
    if not isinstance(metadata, Mapping):
        raise TypeError("Expected PreparedSTGRSample or a reward metadata mapping")
    return json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True)


def _is_size(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2


def _validate_metadata(metadata: Mapping[str, Any], *, step_percent: float | None) -> None:
    task = metadata.get("task")
    if task not in SUPPORTED_TASKS:
        raise RewardMetadataError(f"Missing or unsupported STGR reward task: {task!r}")
    if "answer" not in metadata or metadata.get("answer") is None:
        raise RewardMetadataError("STGR reward metadata has no answer")

    if task == "visual QA":
        if not _is_size(metadata.get("image_size")):
            raise RewardMetadataError("visual QA reward requires annotation image_size=[W,H]")
        if not _is_size(metadata.get("image_size_refine")):
            raise RewardMetadataError("visual QA reward requires processed image_size_refine=[W,H]")
    elif task == "temporal-spatial free-form QA":
        if not _is_size(metadata.get("image_size")):
            raise RewardMetadataError("temporal-spatial reward requires sampled-frame image_size=[W,H]")
        if not isinstance(metadata.get("key_frames"), (list, tuple)) or not metadata.get("key_frames"):
            raise RewardMetadataError("temporal-spatial reward requires non-empty key_frames")
        if not isinstance(metadata.get("key_items"), Mapping):
            raise RewardMetadataError("temporal-spatial reward requires key_items")
        if step_percent is None:
            raise RewardMetadataError(
                "temporal-spatial reward requires the current trainer step_percent for MeSD's temporal curriculum"
            )


def _single_score(output: Any, component_name: str) -> float:
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes)) or len(output) != 1:
        raise RuntimeError(f"{component_name} returned an invalid one-sample result: {output!r}")
    value = output[0]
    if isinstance(value, Mapping):
        value = value.get("score", value.get("overall"))
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{component_name} returned a non-numeric score: {value!r}") from exc
    if not math.isfinite(score):
        raise RuntimeError(f"{component_name} returned a non-finite score: {score!r}")
    return score


def _resolve_weights(reward_weights: Mapping[str, float] | Sequence[float] | None) -> dict[str, float]:
    names = [name for name, _, _ in REWARD_COMPONENTS]
    if reward_weights is None:
        return {name: 1.0 for name in names}
    if isinstance(reward_weights, Mapping):
        unknown = set(reward_weights) - set(names)
        if unknown:
            raise ValueError(f"Unknown reward weight names: {sorted(unknown)}")
        return {name: float(reward_weights.get(name, 1.0)) for name in names}
    if len(reward_weights) != len(names):
        raise ValueError(f"Expected {len(names)} reward weights, got {len(reward_weights)}")
    return {name: float(weight) for name, weight in zip(names, reward_weights)}


def score_mesd_batch(
    reward_inputs: list[Mapping[str, Any]],
    *,
    step_percent: float | None = None,
    strict_metadata: bool = True,
    spatial_iou_mode: str = OFFICIAL_SPATIAL_OPTIONS["spatial_iou_mode"],
    identity_match_mode: str = OFFICIAL_SPATIAL_OPTIONS["identity_match_mode"],
    spatial_norm_mode: str = OFFICIAL_SPATIAL_OPTIONS["spatial_norm_mode"],
    correct_tempgate: bool = OFFICIAL_SPATIAL_OPTIONS["correct_tempgate"],
    outcome_success_iou_threshold: float = 0.5,
    outcome_failure_iou_threshold: float = 0.0,
    outcome_temporal_mcq_strict_rkl: bool = True,
    reward_weights: Mapping[str, float] | Sequence[float] | None = None,
    reward_module: Any | None = None,
) -> list[dict[str, float]]:
    """Compute MeSD's equal-weight sum using RLSD's batch reward API.

    Each row is evaluated independently because the released functions branch
    on ``kwargs['task'][0]``.  Per-row invocation preserves correctness for a
    heterogeneous STGR batch.
    """

    if reward_module is None:
        from . import reward_func as reward_module

    weights = _resolve_weights(reward_weights)
    results: list[dict[str, float]] = []
    for index, reward_input in enumerate(reward_inputs):
        if bool(reward_input.get("vision_read_failed", False)):
            row_scores = {name: 0.0 for name, _, _ in REWARD_COMPONENTS}
            row_scores["overall"] = 0.0
            row_scores["vision_skip"] = 1.0
            row_scores.update(_unknown_outcome(row_valid=False))
            results.append(row_scores)
            continue
        metadata = extract_reward_metadata(reward_input)
        row_step = metadata.get("step_percent", step_percent)
        if row_step is not None:
            try:
                row_step = float(row_step)
            except (TypeError, ValueError) as exc:
                raise RewardMetadataError(f"Row {index} has invalid step_percent: {row_step!r}") from exc
        if strict_metadata:
            try:
                _validate_metadata(metadata, step_percent=row_step)
            except RewardMetadataError as exc:
                raise RewardMetadataError(f"Invalid reward metadata at row {index}: {exc}") from exc

        task = metadata.get("task", reward_input.get("problem_type", ""))
        answer = metadata.get("answer", reward_input.get("ground_truth", ""))
        completion = [[{"role": "assistant", "content": str(reward_input.get("response", ""))}]]
        reward_kwargs = {
            "task": [task],
            "answer": [answer],
            "image_size": [metadata.get("image_size")],
            "image_size_refine": [metadata.get("image_size_refine")],
            "key_frames": [metadata.get("key_frames") or []],
            "key_items": [_drop_none_values(metadata.get("key_items") or {})],
            # The function reads this before task dispatch, so all tasks need a
            # value even though only temporal-spatial QA consumes it.
            "step_percent": [0.0 if row_step is None else row_step],
            "spatial_iou_mode": spatial_iou_mode,
            "identity_match_mode": identity_match_mode,
            "spatial_norm_mode": spatial_norm_mode,
            "correct_tempgate": bool(correct_tempgate),
        }

        row_scores: dict[str, float] = {}
        for metric_name, function_name, positional_answer in REWARD_COMPONENTS:
            function = getattr(reward_module, function_name)
            if positional_answer:
                kwargs_without_answer = {key: value for key, value in reward_kwargs.items() if key != "answer"}
                output = function(completion, [answer], **kwargs_without_answer)
            else:
                output = function(completion, **reward_kwargs)
            row_scores[metric_name] = _single_score(output, metric_name)
        row_scores["overall"] = sum(row_scores[name] * weights[name] for name in weights)
        row_scores["vision_skip"] = 0.0
        row_scores.update(
            classify_mesd_outcome(
                task=task,
                answer=answer,
                response=reward_input.get("response", ""),
                success_iou_threshold=outcome_success_iou_threshold,
                failure_iou_threshold=outcome_failure_iou_threshold,
                temporal_mcq_strict_rkl=outcome_temporal_mcq_strict_rkl,
            )
        )
        results.append(row_scores)
    return results


__all__ = [
    "OFFICIAL_SPATIAL_OPTIONS",
    "REWARD_COMPONENTS",
    "RewardMetadataError",
    "classify_mesd_outcome",
    "extract_reward_metadata",
    "score_mesd_batch",
    "serialize_reward_metadata",
]
