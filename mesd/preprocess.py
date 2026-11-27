"""Open-o3 multi-image video preprocessing for vLLM rollouts.

The released training path samples a video first, then exposes the sampled
frames as timestamped *images*.  Temporal-spatial rows additionally splice in
the annotated keyframes.  Keeping this representation is necessary for prompt
and coordinate parity; using RLSD's native-video path directly is not
equivalent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from PIL import Image

from .prompts import build_student_messages
from .schema import STGRRecord, coerce_record, resolve_keyframe_path, resolve_media_path


VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


class STGRVisionError(RuntimeError):
    pass


class STGRMediaMissingError(STGRVisionError):
    pass


class STGRFrameCacheError(RuntimeError):
    """An unexpected cache-only miss/corruption that must stop training."""


@dataclass(frozen=True)
class KeyframeImage:
    timestamp: int | float
    image: Any


@dataclass(frozen=True)
class PreparedSTGRSample:
    record: STGRRecord
    messages: tuple[dict[str, Any], ...]
    media_path: Path
    images: list[Image.Image]
    frame_times: tuple[float, ...]
    source_frame_count: int
    sample_fps: float | None
    image_size: tuple[int | float, ...]
    image_size_refine: tuple[int, int] | None
    frame_prompt: str
    reward_metadata: dict[str, Any]

    @property
    def processed_image_size(self) -> tuple[int, int]:
        """Processor output size, retained as a compatibility alias."""

        size = self.image_size_refine or self.image_size
        if len(size) != 2:
            raise STGRVisionError(f"Expected a 2-D image size, got {size!r}")
        return int(size[0]), int(size[1])

    def render_prompt(self, processor) -> str:
        """Apply the checkpoint chat template and insert timestamped image slots."""

        prompt = processor.apply_chat_template(
            list(self.messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        if self.record.task != "visual QA":
            if VIDEO_PLACEHOLDER not in prompt:
                raise ValueError("Chat template did not emit the expected Qwen video placeholder")
            prompt = prompt.replace(VIDEO_PLACEHOLDER, self.frame_prompt, 1)
        return prompt

    def as_vllm_input(self, processor) -> dict[str, Any]:
        """Return the vLLM prompt/multimodal payload used by student rollout."""

        return {
            "prompt": self.render_prompt(processor),
            "multi_modal_data": {"image": list(self.images)},
        }


@dataclass(frozen=True)
class SkippedSTGRSample:
    """A MeSD-style recoverable preprocessing failure.

    Training code should synchronize this skip across workers and assign zero
    loss.  It must not optimize against a dummy black video.
    """

    sample_id: str
    source: str
    task: str
    error_type: str
    error_message: str


def _as_frame_list(frames: Sequence[Any] | Any) -> list[Any]:
    try:
        count = len(frames)
    except TypeError as exc:
        raise STGRVisionError("Decoded video frames must be an indexable sequence") from exc
    if count <= 0:
        raise STGRVisionError("Decoded video contains no sampled frames")
    return [frames[index] for index in range(count)]


def _image_size(image: Any) -> tuple[int, int]:
    if isinstance(image, Image.Image):
        return int(image.size[0]), int(image.size[1])
    shape = getattr(image, "shape", None)
    if shape is None or len(shape) < 2:
        raise STGRVisionError(f"Cannot infer image size from {type(image).__name__}")
    return int(shape[-1]), int(shape[-2])


def _to_pil_image(image: Any) -> Image.Image:
    """Normalize Qwen/PyTorch/numpy frames to independent RGB PIL images."""

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - needed only for tensor/array loaders
        raise RuntimeError("Converting tensor video frames to PIL requires numpy") from exc

    value = image
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = array.transpose(1, 2, 0)
    if array.ndim not in (2, 3):
        raise STGRVisionError(f"Cannot convert frame shape {array.shape!r} to PIL")
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
    if array.dtype != np.uint8:
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def build_frame_prompt(frame_times: Sequence[float], *, duration_seconds: int | float) -> str:
    lines = [
        f"Frame {index + 1} at {round(float(timestamp), 1)}s: {IMAGE_PLACEHOLDER}\n"
        for index, timestamp in enumerate(frame_times)
    ]
    lines.append(f"The video is in total {int(duration_seconds)} seconds.\n")
    return "".join(lines)


def assemble_timestamped_frames(
    sampled_frames: Sequence[Any] | Any,
    *,
    sample_fps: float,
    keyframes: Sequence[KeyframeImage] = (),
) -> tuple[tuple[Any, ...], tuple[float, ...], str]:
    """Reproduce Open-o3's sampled-frame/keyframe merge loop.

    Keyframe timestamps are rounded before insertion in the released RL code.
    The input order is preserved; STGR annotations are already chronological.
    """

    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be positive, got {sample_fps}")
    source_frames = _as_frame_list(sampled_frames)
    normalized_keyframes = [KeyframeImage(round(float(item.timestamp)), item.image) for item in keyframes]

    merged: list[Any] = []
    frame_times: list[float] = []
    source_index = 0
    keyframe_index = 0
    while source_index < len(source_frames):
        integer_time = int(source_index / sample_fps)
        if (
            keyframe_index < len(normalized_keyframes)
            and integer_time >= normalized_keyframes[keyframe_index].timestamp
        ):
            keyframe = normalized_keyframes[keyframe_index]
            merged.append(keyframe.image)
            frame_times.append(round(float(keyframe.timestamp), 1))
            keyframe_index += 1
        else:
            merged.append(source_frames[source_index])
            frame_times.append(round(source_index / sample_fps, 1))
            source_index += 1

    duration = int(len(source_frames) / sample_fps)
    prompt = build_frame_prompt(frame_times, duration_seconds=duration)
    return tuple(merged), tuple(frame_times), prompt


def _default_video_loader(path: Path) -> tuple[Any, float]:
    from .vision_process import fetch_video

    # STGR training must fail closed: torchvision reads the complete video
    # before sampling and can turn one damaged input into an unbounded stall.
    # A Decord exception is wrapped as STGRVisionError below, allowing the
    # dataset's existing synchronized dummy/zero-reward path to handle it.
    return fetch_video(
        {"video": str(path)},
        return_video_sample_fps=True,
        video_reader_backend="decord",
        allow_torchvision_fallback=False,
    )


def _default_image_loader(path: Path) -> Image.Image:
    from .vision_process import fetch_image

    return fetch_image({"image": str(path)})


def _default_keyframe_loader(path: Path, target_size: tuple[int, int]) -> Image.Image:
    """Match Open-o3's RGB resize while retaining PIL for the vLLM payload."""

    with Image.open(path) as image:
        return image.convert("RGB").resize(target_size)


def prepare_stgr_sample(
    row: STGRRecord | Mapping[str, Any],
    data_root: str | Path,
    *,
    video_loader: Callable[[Path], tuple[Any, float]] | None = None,
    image_loader: Callable[[Path], Any] | None = None,
    keyframe_loader: Callable[[Path, tuple[int, int]], Any] | None = None,
) -> PreparedSTGRSample:
    """Resolve, decode, and timestamp one STGR sample.

    Loader injection keeps the behavior unit-testable on CPU and lets a future
    dataset layer substitute a local preprocessing cache without changing task
    semantics.
    """

    record = coerce_record(row)
    media_path = resolve_media_path(record, data_root)
    if not media_path.exists():
        raise STGRMediaMissingError(f"Missing STGR media for {record.id}: {media_path}")

    messages = tuple(build_student_messages(record))
    if record.task == "visual QA":
        loader = image_loader or _default_image_loader
        try:
            image = _to_pil_image(loader(media_path))
        except Exception as exc:
            raise STGRVisionError(f"Failed to read image {media_path}: {exc}") from exc
        size = _image_size(image)
        metadata = record.to_reward_metadata(processed_image_size=size)
        return PreparedSTGRSample(
            record=record,
            messages=messages,
            media_path=media_path,
            images=[image],
            frame_times=(),
            source_frame_count=0,
            sample_fps=None,
            image_size=record.image_size,
            image_size_refine=size,
            frame_prompt="",
            reward_metadata=metadata,
        )

    loader = video_loader or _default_video_loader
    try:
        sampled_frames, sample_fps = loader(media_path)
        source_frames = [_to_pil_image(frame) for frame in _as_frame_list(sampled_frames)]
        sample_fps = float(sample_fps)
        size = _image_size(source_frames[0])
    except Exception as exc:
        if isinstance(exc, STGRVisionError):
            raise
        raise STGRVisionError(f"Failed to decode video {media_path}: {exc}") from exc

    keyframes: list[KeyframeImage] = []
    if record.task == "temporal-spatial free-form QA":
        load_keyframe = keyframe_loader or _default_keyframe_loader
        for key_frame in record.key_frames:
            path = resolve_keyframe_path(record, key_frame, data_root)
            if not path.exists():
                raise STGRMediaMissingError(f"Missing STGR keyframe for {record.id}: {path}")
            try:
                image = _to_pil_image(load_keyframe(path, size))
            except Exception as exc:
                raise STGRVisionError(f"Failed to read keyframe {path}: {exc}") from exc
            keyframes.append(KeyframeImage(timestamp=float(key_frame["time"]), image=image))

    images, times, frame_prompt = assemble_timestamped_frames(
        source_frames,
        sample_fps=sample_fps,
        keyframes=keyframes,
    )
    metadata = record.to_reward_metadata(processed_image_size=size)
    return PreparedSTGRSample(
        record=record,
        messages=messages,
        media_path=media_path,
        images=list(images),
        frame_times=times,
        source_frame_count=len(source_frames),
        sample_fps=sample_fps,
        image_size=size,
        image_size_refine=None,
        frame_prompt=frame_prompt,
        reward_metadata=metadata,
    )


STGR_FRAME_CACHE_VERSION = 3
_FRAME_CACHE_VERSION = STGR_FRAME_CACHE_VERSION


def _cache_fingerprint(record: STGRRecord, data_root: str | Path) -> str:
    # Required-cache training must not touch the original video filesystem at
    # all: even a metadata stat on a wedged network mount can block a loader.
    # The cache is an explicit immutable snapshot, tied to the exact dataset
    # SHA/config in the builder manifest rather than a live mount path/mtime.
    # All relative media/keyframe identities are already present in `record`.
    payload = {
        "version": _FRAME_CACHE_VERSION,
        "record": {
            "id": record.id,
            "source": record.source,
            "task": record.task,
            "video_path": record.video_path,
            "image_path": record.image_path,
            "key_frames": list(record.key_frames),
        },
        "vision_config": {
            # Hash effective values, not raw env spelling: unset and an
            # explicitly exported default must address the same cache entry.
            "video_pixels_factor": int(os.environ.get("VIDEO_PIXELS_FACTOR") or "128"),
            "video_total_pixels": int(
                float(os.environ.get("VIDEO_MAX_PIXELS") or str(128000 * 28 * 28 * 0.9))
            ),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_cached_sample(
    cache_entry: Path,
    *,
    record: STGRRecord,
    data_root: str | Path,
) -> PreparedSTGRSample | None:
    metadata_path = cache_entry / "metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a JSON object")
        if metadata.get("version") != _FRAME_CACHE_VERSION:
            return None
        if metadata.get("fingerprint") != cache_entry.name:
            raise ValueError("metadata fingerprint does not match its cache directory")
        if str(metadata.get("record_id")) != record.id:
            raise ValueError("metadata record_id does not match the current sample")
        filenames = metadata["images"]
        if not isinstance(filenames, list) or not filenames:
            raise ValueError("images must be a non-empty list")
        if any(Path(str(filename)).name != filename for filename in filenames):
            raise ValueError("image filenames must be cache-local basenames")
        images: list[Image.Image] = []
        for filename in filenames:
            with Image.open(cache_entry / filename) as image:
                images.append(image.convert("RGB").copy())
        if not images:
            return None
        frame_times = tuple(float(value) for value in metadata["frame_times"])
        source_frame_count = int(metadata["source_frame_count"])
        sample_fps = None if metadata["sample_fps"] is None else float(metadata["sample_fps"])
        frame_prompt = str(metadata["frame_prompt"])
        if record.task == "visual QA":
            if len(images) != 1 or frame_times or sample_fps is not None or frame_prompt:
                raise ValueError("visual cache metadata is inconsistent")
        else:
            if len(images) != len(frame_times):
                raise ValueError("video image/frame-time counts differ")
            if any(not math.isfinite(value) or value < 0 for value in frame_times):
                raise ValueError("video frame times must be finite and non-negative")
            if frame_prompt.count(IMAGE_PLACEHOLDER) != len(images):
                raise ValueError("video frame prompt does not match cached images")
            if source_frame_count <= 0:
                raise ValueError("video source_frame_count must be positive")
            if sample_fps is None or not math.isfinite(sample_fps) or sample_fps <= 0:
                raise ValueError("video sample_fps must be finite and positive")
        image_size_refine = metadata.get("image_size_refine")
        processed_size = tuple(image_size_refine or metadata["image_size"])
        numeric_size = tuple(float(value) for value in processed_size)
        if len(numeric_size) != 2 or any(
            not math.isfinite(value) or value <= 0 for value in numeric_size
        ):
            raise ValueError("processed image size must contain two positive values")
        return PreparedSTGRSample(
            record=record,
            messages=tuple(build_student_messages(record)),
            media_path=resolve_media_path(record, data_root),
            images=images,
            frame_times=frame_times,
            source_frame_count=source_frame_count,
            sample_fps=sample_fps,
            image_size=tuple(metadata["image_size"]),
            image_size_refine=None if image_size_refine is None else tuple(image_size_refine),
            frame_prompt=frame_prompt,
            # Answers and privileged annotations are deliberately rebuilt from
            # the current row.  The frame cache contains vision data only and
            # must not pin training to stale reward metadata.
            reward_metadata=record.to_reward_metadata(
                processed_image_size=(int(processed_size[0]), int(processed_size[1]))
            ),
        )
    except (IndexError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _store_cached_sample(cache_entry: Path, sample: PreparedSTGRSample) -> None:
    cache_entry.mkdir(parents=True, exist_ok=True)
    nonce = f"{os.getpid()}-{uuid4().hex}"
    filenames = [f"frame_{index:04d}.png" for index in range(len(sample.images))]
    temporary_paths: list[Path] = []
    try:
        for image, filename in zip(sample.images, filenames):
            target = cache_entry / filename
            temporary = cache_entry / f".{filename}.{nonce}.png"
            temporary_paths.append(temporary)
            image.save(temporary, format="PNG")
            os.replace(temporary, target)
        metadata = {
            "version": _FRAME_CACHE_VERSION,
            "fingerprint": cache_entry.name,
            "record_id": sample.record.id,
            "images": filenames,
            "frame_times": list(sample.frame_times),
            "source_frame_count": sample.source_frame_count,
            "sample_fps": sample.sample_fps,
            "image_size": list(sample.image_size),
            "image_size_refine": (
                None if sample.image_size_refine is None else list(sample.image_size_refine)
            ),
            "frame_prompt": sample.frame_prompt,
        }
        metadata_target = cache_entry / "metadata.json"
        metadata_temporary = cache_entry / f".metadata.{nonce}.tmp"
        temporary_paths.append(metadata_temporary)
        metadata_temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(metadata_temporary, metadata_target)
        try:
            (cache_entry / "failure.json").unlink(missing_ok=True)
        except OSError:
            pass
    finally:
        for temporary in temporary_paths:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def stgr_frame_cache_entry(
    record: STGRRecord | Mapping[str, Any],
    media_root: str | Path,
    frame_cache_dir: str | Path,
) -> Path:
    """Return the stable cache entry used by offline builders and training."""

    normalized = coerce_record(record)
    return Path(frame_cache_dir).expanduser() / _cache_fingerprint(normalized, media_root)


def store_stgr_frame_cache_entry(
    cache_entry: str | Path,
    sample: PreparedSTGRSample,
) -> None:
    """Atomically publish a successfully decoded cache entry."""

    _store_cached_sample(Path(cache_entry), sample)


def store_stgr_frame_cache_failure(
    cache_entry: str | Path,
    *,
    record_id: str,
    error_type: str,
    error_message: str,
    **extra: Any,
) -> None:
    """Atomically mark a decoder failure that training may safely zero-mask."""

    entry = Path(cache_entry)
    entry.mkdir(parents=True, exist_ok=True)
    payload = {
        **extra,
        "version": _FRAME_CACHE_VERSION,
        "fingerprint": entry.name,
        "record_id": str(record_id),
        "error_type": str(error_type),
        "error_message": str(error_message),
    }
    nonce = f"{os.getpid()}-{uuid4().hex}"
    temporary = entry / f".failure.{nonce}.tmp"
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, entry / "failure.json")
        try:
            (entry / "metadata.json").unlink(missing_ok=True)
        except OSError:
            pass
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _load_cached_failure(
    cache_entry: Path,
    *,
    record: STGRRecord,
    required: bool,
) -> None:
    failure_path = cache_entry / "failure.json"
    if not failure_path.is_file():
        return
    try:
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        if not isinstance(failure, dict):
            raise TypeError("failure marker must be a JSON object")
        if failure.get("version") != _FRAME_CACHE_VERSION:
            raise ValueError(f"unsupported failure version {failure.get('version')!r}")
        if failure.get("fingerprint") != cache_entry.name:
            raise ValueError("failure fingerprint does not match its cache directory")
        if str(failure.get("record_id")) != record.id:
            raise ValueError("failure record_id does not match the current sample")
        error_type_value = failure["error_type"]
        error_message_value = failure["error_message"]
        if not isinstance(error_type_value, str) or not error_type_value.strip():
            raise ValueError("failure error_type must be a non-empty string")
        if not isinstance(error_message_value, str) or not error_message_value.strip():
            raise ValueError("failure error_message must be a non-empty string")
        error_type = error_type_value
        error_message = error_message_value
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        if required:
            raise STGRFrameCacheError(
                f"Corrupt STGR frame-cache failure marker {failure_path}: {exc}"
            ) from exc
        return
    raise STGRVisionError(
        f"Offline frame cache marked this sample failed ({error_type}): {error_message}"
    )


def load_stgr_frame_cache_entry(
    record: STGRRecord | Mapping[str, Any],
    media_root: str | Path,
    frame_cache_dir: str | Path,
    *,
    required: bool = False,
) -> PreparedSTGRSample | None:
    """Load one cache entry, enforcing fail-fast cache-only semantics if requested."""

    normalized = coerce_record(record)
    cache_entry = stgr_frame_cache_entry(normalized, media_root, frame_cache_dir)
    metadata_exists = (cache_entry / "metadata.json").is_file()
    cached = _load_cached_sample(cache_entry, record=normalized, data_root=media_root)
    if cached is not None:
        return cached
    if metadata_exists and required:
        raise STGRFrameCacheError(
            f"Corrupt STGR frame-cache entry for id={normalized.id}: {cache_entry}"
        )
    _load_cached_failure(cache_entry, record=normalized, required=required)
    if required:
        raise STGRFrameCacheError(
            f"Missing required STGR frame-cache entry for id={normalized.id}: {cache_entry}"
        )
    return None


def prepare_stgr_example(
    record: STGRRecord | Mapping[str, Any],
    media_root: str | Path,
    frame_cache_dir: str | Path | None = None,
    frame_cache_required: bool = False,
    *,
    video_loader: Callable[[Path], tuple[Any, float]] | None = None,
    image_loader: Callable[[Path], Any] | None = None,
    keyframe_loader: Callable[[Path, tuple[int, int]], Any] | None = None,
) -> PreparedSTGRSample:
    """Stable dataset-facing adapter with an optional decoded-frame cache.

    ``media_root`` is the STGR data root that contains the ``videos/``
    directory.  Cached entries are content/config keyed and store RGB PNGs,
    making them safe for reuse by independent vLLM rollout workers.
    """

    normalized = coerce_record(record)
    if frame_cache_required and frame_cache_dir is None:
        raise STGRFrameCacheError(
            "frame_cache_required=True requires a configured STGR frame-cache directory"
        )
    if frame_cache_dir is None:
        return prepare_stgr_sample(
            normalized,
            media_root,
            video_loader=video_loader,
            image_loader=image_loader,
            keyframe_loader=keyframe_loader,
        )

    cache_entry = stgr_frame_cache_entry(normalized, media_root, frame_cache_dir)
    cached = load_stgr_frame_cache_entry(
        normalized,
        media_root,
        frame_cache_dir,
        required=frame_cache_required,
    )
    if cached is not None:
        return cached
    sample = prepare_stgr_sample(
        normalized,
        media_root,
        video_loader=video_loader,
        image_loader=image_loader,
        keyframe_loader=keyframe_loader,
    )
    try:
        store_stgr_frame_cache_entry(cache_entry, sample)
    except OSError:
        # A cache must never make a valid training example unusable (read-only
        # shared datasets and concurrent workers are both common).
        pass
    return sample


def prepare_stgr_sample_or_skip(
    row: STGRRecord | Mapping[str, Any],
    data_root: str | Path,
    **kwargs: Any,
) -> PreparedSTGRSample | SkippedSTGRSample:
    record = coerce_record(row)
    try:
        return prepare_stgr_sample(record, data_root, **kwargs)
    except STGRVisionError as exc:
        return SkippedSTGRSample(
            sample_id=record.id,
            source=record.source,
            task=record.task,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )


__all__ = [
    "IMAGE_PLACEHOLDER",
    "VIDEO_PLACEHOLDER",
    "KeyframeImage",
    "PreparedSTGRSample",
    "SkippedSTGRSample",
    "STGRMediaMissingError",
    "STGRFrameCacheError",
    "STGR_FRAME_CACHE_VERSION",
    "STGRVisionError",
    "assemble_timestamped_frames",
    "build_frame_prompt",
    "load_stgr_frame_cache_entry",
    "prepare_stgr_example",
    "prepare_stgr_sample",
    "prepare_stgr_sample_or_skip",
    "stgr_frame_cache_entry",
    "store_stgr_frame_cache_entry",
    "store_stgr_frame_cache_failure",
]
