"""MeSD reward entry point for STGR training."""

from __future__ import annotations

from typing import Any

from mesd.rewards import score_mesd_batch


REWARD_NAME = "mesd"
REWARD_TYPE = "batch"


def compute_mesd_reward(reward_inputs: list[dict[str, Any]], **kwargs: Any) -> list[dict[str, float]]:
    return score_mesd_batch(reward_inputs, **kwargs)
