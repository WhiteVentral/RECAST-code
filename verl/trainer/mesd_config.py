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

"""Configuration for STGR Direct RKL with multiview EMA replay."""

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class MeSDConfig:
    mesd_teacher_aggregation: str = "anchored_avsd"
    """Privileged-teacher evidence aggregation mode.

    The supported ``anchored_avsd`` mode combines A, A+T and A+S views
    while keeping the answer-only view as its anchor.
    """
    mesd_avsd_evidence_alpha: float = 1.0
    """Anchored-AVSD evidence interpolation strength in [0, 1]."""
    mesd_avsd_epsilon: float = 1.0e-6
    """Strictly positive numerical-stability constant for anchored AVSD."""
    mesd_avsd_use_token_roles: bool = False
    """Route anchored-AVSD evidence by response-token role when enabled."""
    mesd_teacher_max_prompt_length: int = 18432
    """Maximum tokenized length of the privileged teacher prompt."""
    mesd_teacher_update_rate: float = 0.01
    """EMA coefficient: teacher <- (1-rate) * teacher + rate * student."""
    mesd_teacher_update_interval: int = 1
    """Apply the EMA update every N completed optimizer steps."""
    mesd_reweighting_lambda: float = 0.5
    """Initial interpolation weight for MeSD token credit."""
    mesd_reweighting_anneal_steps: int = 1000
    """Number of 1-indexed steps over which lambda anneals to pure GRPO."""
    mesd_teacher_disable_after_step: int = 1000
    """Use the teacher through this step; later steps skip its forward pass."""
    mesd_reweighting_topk: int = 16
    """Teacher-centred local support size."""
    mesd_reweighting_topk_gamma: float = 0.0
    """Interpolation between local-support and sampled-token gaps (released: 0)."""
    mesd_reweighting_weight_clip: Optional[float] = 0.2
    """Symmetric clip around one for the direction-magnitude evidence weight."""
    mesd_include_sampled_token_in_topk: bool = True
    """Union the realized completion token with teacher top-k support."""
    mesd_policy_clip_low: float = 0.2
    mesd_policy_clip_high: float = 0.2
    """Two-sided token-GRPO policy clip; MeSD does not use PPO dual clipping."""
    mesd_failure_direction_mode: str = "direct_rkl"
    """Confirmed-failure objective; this release supports direct_rkl only."""
    mesd_failure_direction_gamma: float = 1.0
    """Sequence-level coefficient for the failure direction loss."""
    mesd_failure_direction_warmup_steps: int = 100
    """Linearly warm the failure direction coefficient over this many steps."""
    mesd_failure_direction_topk: int = 32
    """Per-distribution top-K used to build the common K+tail alphabet."""
    mesd_failure_target_epsilon: float = 1.0e-8
    """Target-only probability floor on the compressed K+tail alphabet."""

    def post_init(self):
        if self.mesd_teacher_aggregation != "anchored_avsd":
            raise ValueError("Only anchored_avsd teacher aggregation is supported.")
        if self.mesd_failure_direction_mode != "direct_rkl":
            raise ValueError("Only direct_rkl failure direction is supported.")
        if not (math.isfinite(float(self.mesd_avsd_evidence_alpha)) and 0.0 <= self.mesd_avsd_evidence_alpha <= 1.0):
            raise ValueError(f"mesd.mesd_avsd_evidence_alpha must be in [0, 1], got {self.mesd_avsd_evidence_alpha}.")
        if not (math.isfinite(float(self.mesd_avsd_epsilon)) and self.mesd_avsd_epsilon > 0.0):
            raise ValueError(f"mesd.mesd_avsd_epsilon must be positive, got {self.mesd_avsd_epsilon}.")
        if not isinstance(self.mesd_avsd_use_token_roles, bool):
            raise ValueError("mesd.mesd_avsd_use_token_roles must be a boolean.")
        if not (0.0 <= self.mesd_teacher_update_rate <= 1.0):
            raise ValueError(f"mesd.mesd_teacher_update_rate must be in [0, 1], got {self.mesd_teacher_update_rate}.")
        if self.mesd_teacher_update_interval <= 0:
            raise ValueError("mesd.mesd_teacher_update_interval must be positive.")
        if not (0.0 <= self.mesd_reweighting_lambda <= 1.0):
            raise ValueError(f"mesd.mesd_reweighting_lambda must be in [0, 1], got {self.mesd_reweighting_lambda}.")
        if self.mesd_reweighting_anneal_steps < 0:
            raise ValueError("mesd.mesd_reweighting_anneal_steps must be non-negative.")
        if self.mesd_teacher_disable_after_step < 0:
            raise ValueError("mesd.mesd_teacher_disable_after_step must be non-negative.")
        if self.mesd_reweighting_topk <= 0:
            raise ValueError("mesd.mesd_reweighting_topk must be positive.")
        if not (0.0 <= self.mesd_reweighting_topk_gamma <= 1.0):
            raise ValueError(
                f"mesd.mesd_reweighting_topk_gamma must be in [0, 1], got {self.mesd_reweighting_topk_gamma}."
            )
        if self.mesd_reweighting_weight_clip is not None and not (
            math.isfinite(float(self.mesd_reweighting_weight_clip)) and self.mesd_reweighting_weight_clip >= 0
        ):
            raise ValueError("mesd.mesd_reweighting_weight_clip must be finite and non-negative when set.")
        if self.mesd_teacher_max_prompt_length <= 0:
            raise ValueError("mesd.mesd_teacher_max_prompt_length must be positive.")
        if not (0.0 <= self.mesd_policy_clip_low < 1.0):
            raise ValueError("mesd.mesd_policy_clip_low must be in [0, 1).")
        if self.mesd_policy_clip_high < 0.0:
            raise ValueError("mesd.mesd_policy_clip_high must be non-negative.")
        if self.mesd_failure_direction_mode != "none" and self.mesd_teacher_aggregation != "anchored_avsd":
            raise ValueError(
                "Failure direction control currently requires mesd.mesd_teacher_aggregation=anchored_avsd."
            )
        if not (math.isfinite(float(self.mesd_failure_direction_gamma)) and self.mesd_failure_direction_gamma >= 0.0):
            raise ValueError("mesd.mesd_failure_direction_gamma must be finite and non-negative.")
        if self.mesd_failure_direction_warmup_steps < 0:
            raise ValueError("mesd.mesd_failure_direction_warmup_steps must be non-negative.")
        if self.mesd_failure_direction_topk <= 0:
            raise ValueError("mesd.mesd_failure_direction_topk must be positive.")
        if not (math.isfinite(float(self.mesd_failure_target_epsilon)) and self.mesd_failure_target_epsilon > 0.0):
            raise ValueError("mesd.mesd_failure_target_epsilon must be finite and positive.")
