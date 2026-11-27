"""STGR schema validation and source-aware media path resolution."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SUPPORTED_TASKS = frozenset(
    {
        "visual QA",
        "temporal-spatial free-form QA",
        "temporal QA",
        "temporal QA (MCQ)",
        "General video QA MCQ",
        "General video QA Free-form",
    }
)


def load_stgr_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load released STGR JSON without materializing a huge Arrow union struct.

    ``datasets.load_dataset("json")`` expands every sparse object name inside
    ``key_items`` into one shared nested Arrow schema.  The released RL split
    has thousands of such dynamic fields, so the expansion wastes RAM before
    training even starts.  STGR is small enough to retain as ordinary sparse
    Python dictionaries in the central dataset process.
    """

    source = Path(path).expanduser()
    if source.suffix.lower() in {".jsonl", ".ndjson"}:
        rows: Any = []
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                value = json.loads(text)
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"STGR JSONL row {line_number} must be an object, "
                        f"got {type(value).__name__}."
                    )
                rows.append(dict(value))
    else:
        with source.open(encoding="utf-8") as handle:
            rows = json.load(handle)

    if not isinstance(rows, list):
        raise ValueError(
            f"STGR annotation root must be a JSON list, got {type(rows).__name__}."
        )
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"STGR row {index} must be an object, got {type(row).__name__}."
            )
    return [dict(row) for row in rows]


def _drop_none_values(value: Any) -> Any:
    """Prune Arrow union-struct nulls before rows enter Ray/n-way repeat."""

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


@dataclass(frozen=True)
class STGRRecord:
    """Stable in-memory view of one STGR SFT or RL row.

    ``reasoning_process`` is present for SFT rows and absent for RL rows.  The
    remaining fields match the released Open-o3/MeSD JSON annotations.
    """

    id: str
    source: str
    task: str
    question: str
    answer: str
    video_path: str = ""
    image_path: str = ""
    image_size: tuple[int | float, ...] = ()
    key_frames: tuple[dict[str, Any], ...] = ()
    key_items: dict[str, Any] | None = None
    reasoning_process: str | None = None

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "STGRRecord":
        missing = [key for key in ("id", "source", "task", "question", "answer") if key not in row]
        if missing:
            raise ValueError(f"STGR row is missing required fields: {', '.join(missing)}")

        task = str(row["task"])
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"Unsupported STGR task: {task!r}")

        video_path = str(row.get("video_path") or "")
        image_path = str(row.get("image_path") or "")
        if task == "visual QA":
            if not image_path:
                raise ValueError(f"Visual STGR row {row['id']!r} has no image_path")
        elif not video_path:
            raise ValueError(f"Video STGR row {row['id']!r} has no video_path")

        raw_size = row.get("image_size") or []
        if not isinstance(raw_size, (list, tuple)):
            raise ValueError(f"image_size must be a list/tuple, got {type(raw_size).__name__}")

        raw_frames = row.get("key_frames") or []
        if not isinstance(raw_frames, (list, tuple)):
            raise ValueError(f"key_frames must be a list/tuple, got {type(raw_frames).__name__}")
        if not all(isinstance(frame, Mapping) for frame in raw_frames):
            raise ValueError("Every key_frames entry must be a mapping")

        raw_items = row.get("key_items") or {}
        if not isinstance(raw_items, Mapping):
            raise ValueError(f"key_items must be a mapping, got {type(raw_items).__name__}")

        reasoning = row.get("reasoning_process")
        return cls(
            id=str(row["id"]),
            source=str(row["source"]),
            task=task,
            question=str(row["question"]),
            answer=str(row["answer"]),
            video_path=video_path,
            image_path=image_path,
            image_size=tuple(raw_size),
            key_frames=tuple(
                deepcopy(_drop_none_values(dict(frame))) for frame in raw_frames
            ),
            key_items=deepcopy(_drop_none_values(dict(raw_items))),
            reasoning_process=None if reasoning is None else str(reasoning),
        )

    def to_reward_metadata(self, *, processed_image_size: tuple[int, int] | None = None) -> dict[str, Any]:
        """Return a fixed-key reward payload safe to JSON-serialize.

        Dynamic object names inside ``key_items`` should be serialized as one
        JSON column when converting STGR to Arrow/Parquet, rather than expanded
        into thousands of dynamic struct columns.
        """

        annotation_size = list(self.image_size)
        processed_size = list(processed_image_size) if processed_image_size else None
        # These two names intentionally mirror Open-o3's reward kwargs.  For
        # GQA, image_size is the annotation coordinate space and
        # image_size_refine is the processor output size.  Video boxes are
        # normalized, so image_size is the sampled-frame size instead.
        reward_image_size = annotation_size if self.task == "visual QA" else processed_size
        reward_image_size_refine = processed_size if self.task == "visual QA" else None
        return {
            "id": self.id,
            "source": self.source,
            "task": self.task,
            "answer": self.answer,
            "image_size": reward_image_size,
            "image_size_refine": reward_image_size_refine,
            "annotation_image_size": annotation_size,
            "processed_image_size": processed_size,
            "key_frames": deepcopy(list(self.key_frames)),
            "key_items": deepcopy(self.key_items or {}),
        }


def coerce_record(row: STGRRecord | Mapping[str, Any]) -> STGRRecord:
    return row if isinstance(row, STGRRecord) else STGRRecord.from_mapping(row)


def is_stgr_record(row: Any) -> bool:
    """Return whether ``row`` satisfies the supported STGR task contract."""

    if not isinstance(row, (STGRRecord, Mapping)):
        return False
    try:
        coerce_record(row)
    except (TypeError, ValueError):
        return False
    return True


def _video_subdir(source: str) -> Path:
    if source == "videoespresso_train_video":
        return Path("videoespresso/videos")
    if source == "timerft":
        return Path("timerft")
    if source.startswith("STR_plm"):
        return Path("stgr/plm/videos")
    if source.startswith("STR"):
        return Path("stgr/temporal_grounding/videos")
    if source.startswith("TVG"):
        return Path("tvg_r1")
    if source.startswith("videor1"):
        return Path("videor1")
    raise ValueError(f"Unsupported STGR video source: {source!r}")


def resolve_media_path(row: STGRRecord | Mapping[str, Any], data_root: str | Path) -> Path:
    """Resolve an STGR media path using the released Open-o3 directory layout."""

    record = coerce_record(row)
    videos_root = Path(data_root).expanduser() / "videos"
    if record.task == "visual QA":
        if record.source == "gqa":
            return videos_root / "gqa" / record.image_path
        if record.source == "TreeVGR":
            return videos_root / "treevgr" / record.image_path
        raise ValueError(f"Unsupported STGR image source: {record.source!r}")
    return videos_root / _video_subdir(record.source) / record.video_path


def resolve_keyframe_root(row: STGRRecord | Mapping[str, Any], data_root: str | Path) -> Path:
    record = coerce_record(row)
    videos_root = Path(data_root).expanduser() / "videos"
    if record.source == "videoespresso_train_video":
        return videos_root / "videoespresso/kfs"
    if record.source.startswith("STR_plm"):
        return videos_root / "stgr/plm/kfs"
    if record.source.startswith("STR"):
        return videos_root / "stgr/temporal_grounding/kfs"
    raise ValueError(f"Source {record.source!r} has no released STGR keyframe root")


def resolve_keyframe_path(
    row: STGRRecord | Mapping[str, Any],
    key_frame: Mapping[str, Any],
    data_root: str | Path,
) -> Path:
    path = key_frame.get("path")
    if not path:
        raise ValueError("key frame has no path")
    return resolve_keyframe_root(row, data_root) / str(path)
