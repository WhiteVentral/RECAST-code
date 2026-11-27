"""MeSD task adapters, preprocessing, and reward utilities.

This package intentionally stays independent from ``verl``.  It contains the
task semantics that must remain stable when the rollout/training backend is
changed to vLLM.
"""

from ._version import __version__
from .preprocess import PreparedSTGRSample, prepare_stgr_example
from .prompts import SYSTEM_PROMPTS, build_student_messages
from .rewards import score_mesd_batch, serialize_reward_metadata
from .schema import STGRRecord, is_stgr_record, resolve_keyframe_path, resolve_media_path


__all__ = [
    "SYSTEM_PROMPTS",
    "PreparedSTGRSample",
    "STGRRecord",
    "__version__",
    "build_student_messages",
    "is_stgr_record",
    "prepare_stgr_example",
    "resolve_keyframe_path",
    "resolve_media_path",
    "score_mesd_batch",
    "serialize_reward_metadata",
]
