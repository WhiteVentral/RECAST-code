"""Conservative field verification of semantic chunks (method sections 4, 7, 9 and 15).

Field verification decides *support*, *contradiction* or *unknown* for fields
whose identity and evidence correspondence are clear.  It is a separate
object from task reward and from teacher trustworthiness: in this
implementation it is recorded for diagnostics and provenance only and never
switches the distillation objective.

Evidence rules preserved from the dataset analysis (section 4):

* object boxes are joined to frame times only through an explicit
  ``str(frame.idx)`` match (delegated to ``build_mesd_privileged_views``);
  no time is guessed for key-item indices without a key frame;
* video boxes are normalized and image boxes are pixel boxes; both are
  converted to the processed-image pixel space seen by the model before any
  comparison;
* temporal-QA supervision lives in ``answer``;
* unknown fields stay unknown -- they are neither errors nor verified
  tokens, and they never enter the denominator of a verified error rate.

Decision rules (deliberately conservative; IoU tolerances instead of exact
string matching).  A field is only decided when its correspondence with an
annotated event or object is established; otherwise it stays unknown:

* final answer -- option letters are compared with the annotated option;
  intervals use temporal IoU (>= ``supported_iou`` supported, no overlap
  contradicted, otherwise unknown); visual-QA object/box answers use the
  same rule on name-matched boxes.  The answer field corresponds to the
  question by construction (it is the field the task reward parses);
* reasoning interval -- unknown.  A free-form sentence that mentions the
  annotated interval may describe another event ("the dog barks from 2 s to
  4 s") or deny the event ("the door does not open from 2 s to 4 s"); the
  interval value alone does not establish that it asserts the annotated
  event, so it is never labeled from the answer's interval status;
* reasoning object-box-time claim -- decidable only when a key frame lies
  within ``time_tolerance`` seconds and an annotated object of that frame
  has the same identity (same normalized name, or in ``soft`` mode the same
  head noun; sharing only a modifier or an article is not an identity
  match); then box IoU decides as above;
* a chunk with a negation or contrast cue (``has_negation``: "not", "no",
  "without", "instead of", ...) leaves its interval and box fields unknown;
  structured option letters are unaffected;
* everything else (free-form text, single times, unnamed boxes) -- unknown.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from mesd.rewards import _interval_iou, _parse_choice, _parse_temporal_answer

from ..mesd_algos import build_mesd_privileged_views
from .chunks import ROLE_ANSWER, ChunkedResponse, FieldClaim, SemanticChunk


__all__ = [
    "STATUS_CONTRADICTED",
    "STATUS_SUPPORTED",
    "STATUS_UNKNOWN",
    "RowEvidence",
    "VerificationConfig",
    "build_row_evidence",
    "verify_response",
]

STATUS_SUPPORTED = "supported"
STATUS_CONTRADICTED = "contradicted"
STATUS_UNKNOWN = "unknown"

_MCQ_TASKS = frozenset({"General video QA MCQ", "temporal QA (MCQ)"})
_WINDOW_TASKS = frozenset({"temporal QA", "temporal QA (MCQ)"})
_OPTION_RE = re.compile(r"Correct\s+Option\s*:\s*([^\n\r]+)", re.IGNORECASE)


@dataclass(frozen=True)
class VerificationConfig:
    time_tolerance: float = 1.0
    """Seconds between a claimed time and an annotated key frame (reward's gate)."""
    supported_iou: float = 0.5
    """Temporal/spatial IoU at or above which a field is supported."""
    name_match: str = "soft"
    """Object identity rule: ``soft`` (reward's rule) or ``strict``."""

    def __post_init__(self) -> None:
        if not (math.isfinite(float(self.time_tolerance)) and float(self.time_tolerance) >= 0.0):
            raise ValueError("time_tolerance must be finite and non-negative.")
        if not (0.0 < float(self.supported_iou) <= 1.0):
            raise ValueError("supported_iou must be in (0, 1].")
        if self.name_match not in ("soft", "strict"):
            raise ValueError("name_match must be 'soft' or 'strict'.")


@dataclass
class RowEvidence:
    task: str
    option: Optional[str] = None
    window: Optional[tuple[float, float]] = None
    frame_objects: list[dict] = field(default_factory=list)
    image_objects: list[dict] = field(default_factory=list)

    @property
    def has_any(self) -> bool:
        return bool(self.option or self.window or self.frame_objects or self.image_objects)


def _as_window(value: Any) -> Optional[tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(end) and end > start):
        return None
    return start, end


def build_row_evidence(row: Mapping[str, Any]) -> RowEvidence:
    """Collect the original evidence of one training row in model coordinates."""

    task = str(row.get("task") or "")
    answer = row.get("ground_truth", row.get("answer"))
    views = build_mesd_privileged_views(row)
    evidence = RowEvidence(task=task)
    if task in _MCQ_TASKS and answer is not None:
        lines = str(answer).splitlines()
        evidence.option = _parse_choice(lines[0]) if lines else None
    if task in _WINDOW_TASKS:
        evidence.window = _as_window(views.get("temporal_window"))
    if task == "temporal-spatial free-form QA":
        for frame in views.get("spatiotemporal_evidence") or []:
            if frame.get("coordinate_space") != "processed_pixel_xyxy":
                continue
            evidence.frame_objects.append(
                {
                    "timestamp": float(frame["timestamp"]),
                    "annotation_idx": frame.get("annotation_idx"),
                    "objects": list(frame.get("objects") or []),
                }
            )
    if task == "visual QA":
        for frame in views.get("spatial_evidence") or []:
            if frame.get("anchor") == "input_image" and frame.get("coordinate_space") == "processed_pixel_xyxy":
                evidence.image_objects.extend(frame.get("objects") or [])
    return evidence


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0.0 else 0.0


_NAME_STOPWORDS = frozenset(
    {"a", "an", "the", "this", "that", "these", "those", "his", "her", "its", "their", "our", "my", "your", "some"}
)
_NAME_WORD_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")


def _singular(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _name_words(name: str) -> list[str]:
    return [_singular(word) for word in _NAME_WORD_RE.findall(name.lower()) if word not in _NAME_STOPWORDS]


def _names_match(predicted: Optional[str], annotated: Optional[str], mode: str) -> bool:
    """Object identity: same normalized name, or (``soft``) the same head noun.

    Unlike the task reward's lenient overlap rule, sharing only a modifier
    ("red cup" / "red car") or an article ("the man" / "the cup") is not an
    identity match here: verification needs correspondence, not tolerance.
    """

    if not predicted or not annotated:
        return False
    first, second = _name_words(predicted), _name_words(annotated)
    if not first or not second:
        return False
    if first == second:
        return True
    if mode == "strict":
        return False
    return first[-1] == second[-1]


def _spatial_status(
    claim: FieldClaim, objects: Sequence[Mapping[str, Any]], cfg: VerificationConfig
) -> Optional[str]:
    """IoU decision on name-matched annotated objects; ``None`` when undecidable."""

    if not claim.boxes:
        return None
    matched = [obj for obj in objects if _names_match(claim.object_name, obj.get("name"), cfg.name_match)]
    if not matched:
        return None
    best = 0.0
    for obj in matched:
        for annotated in obj.get("bboxes") or []:
            for predicted in claim.boxes:
                best = max(best, _box_iou(predicted, annotated))
    if best >= float(cfg.supported_iou):
        return STATUS_SUPPORTED
    if best <= 0.0:
        return STATUS_CONTRADICTED
    return None


def _temporal_status(predicted: Optional[tuple[float, float]], window: Optional[tuple[float, float]], cfg) -> Optional[str]:
    if predicted is None or window is None:
        return None
    if not (math.isfinite(predicted[0]) and math.isfinite(predicted[1]) and predicted[1] > predicted[0]):
        return None
    iou = _interval_iou(predicted, window)
    if iou >= float(cfg.supported_iou):
        return STATUS_SUPPORTED
    if iou <= 0.0:
        return STATUS_CONTRADICTED
    return None


def _combine(statuses: Sequence[Optional[str]]) -> str:
    decided = [status for status in statuses if status is not None]
    if STATUS_CONTRADICTED in decided:
        return STATUS_CONTRADICTED
    if STATUS_SUPPORTED in decided:
        return STATUS_SUPPORTED
    return STATUS_UNKNOWN


def _answer_fields(answer_text: Optional[str], task: str) -> dict[str, Any]:
    fields: dict[str, Any] = {"option": None, "interval": None}
    if not answer_text:
        return fields
    if task == "General video QA MCQ":
        fields["option"] = _parse_choice(answer_text)
    elif task == "temporal QA (MCQ)":
        options = _OPTION_RE.findall(answer_text)
        fields["option"] = _parse_choice(options[0].strip()) if len(options) == 1 else None
    if task in _WINDOW_TASKS:
        fields["interval"] = _parse_temporal_answer(answer_text)
    return fields


def _verify_answer_chunk(chunk: SemanticChunk, answer_text, evidence: RowEvidence, cfg) -> tuple[str, list[str]]:
    statuses: list[Optional[str]] = []
    refs: list[str] = []
    fields = _answer_fields(answer_text, evidence.task)
    if evidence.option is not None and fields["option"] is not None:
        status = STATUS_SUPPORTED if fields["option"] == evidence.option else STATUS_CONTRADICTED
        statuses.append(status)
        refs.append(f"answer.option:{status}")
    if chunk.has_negation:
        if evidence.window is not None or (evidence.task == "visual QA" and evidence.image_objects):
            refs.append(f"answer.fields:{STATUS_UNKNOWN}(negation cue)")
        return _combine(statuses), refs
    if evidence.window is not None:
        status = _temporal_status(fields["interval"], evidence.window, cfg)
        if status is not None:
            statuses.append(status)
            refs.append(f"answer.temporal_window:{status}")
    if evidence.task == "visual QA" and evidence.image_objects:
        for claim in chunk.claims:
            if claim.kind in ("object_box", "object_box_time"):
                status = _spatial_status(claim, evidence.image_objects, cfg)
                if status is not None:
                    statuses.append(status)
                    refs.append(f"answer.image_object[{claim.object_name}]:{status}")
    return _combine(statuses), refs


def _verify_reasoning_chunk(chunk: SemanticChunk, evidence: RowEvidence, cfg) -> tuple[str, list[str]]:
    statuses: list[Optional[str]] = []
    refs: list[str] = []
    if chunk.has_negation and chunk.claims:
        return STATUS_UNKNOWN, [f"reasoning.fields:{STATUS_UNKNOWN}(negation cue)"]
    for claim in chunk.claims:
        if claim.kind == "interval":
            # The interval value alone does not tie a free-form sentence to the annotated event.
            refs.append(f"reasoning.interval:{STATUS_UNKNOWN}(event correspondence not established)")
        elif claim.kind == "object_box_time" and claim.time is not None and evidence.frame_objects:
            frames = [
                frame
                for frame in evidence.frame_objects
                if abs(float(frame["timestamp"]) - float(claim.time)) <= float(cfg.time_tolerance)
            ]
            if not frames:
                continue
            frame = min(frames, key=lambda item: abs(float(item["timestamp"]) - float(claim.time)))
            status = _spatial_status(claim, frame["objects"], cfg)
            if status is not None:
                statuses.append(status)
                refs.append(f"key_items[idx={frame.get('annotation_idx')}][{claim.object_name}]:{status}")
        elif claim.kind == "object_box" and evidence.task == "visual QA" and evidence.image_objects:
            status = _spatial_status(claim, evidence.image_objects, cfg)
            if status is not None:
                statuses.append(status)
                refs.append(f"image_object[{claim.object_name}]:{status}")
    return _combine(statuses), refs


def verify_response(chunked: ChunkedResponse, evidence: RowEvidence, cfg: Optional[VerificationConfig] = None) -> None:
    """Set ``status`` and ``evidence_refs`` of every chunk in place."""

    cfg = cfg or VerificationConfig()
    for chunk in chunked.chunks:
        if not chunk.grounded:
            chunk.status, chunk.evidence_refs = STATUS_UNKNOWN, []
        elif chunk.role == ROLE_ANSWER:
            chunk.status, chunk.evidence_refs = _verify_answer_chunk(chunk, chunked.answer_text, evidence, cfg)
        else:
            chunk.status, chunk.evidence_refs = _verify_reasoning_chunk(chunk, evidence, cfg)
