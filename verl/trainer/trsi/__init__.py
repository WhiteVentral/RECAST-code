"""Video-t-RSI: token-level recursive self-improvement of self-distillation teachers.

This sub-package holds the algorithmic pieces of the method described in
``Method_and_Motivation_EN.md`` (package root).  Everything here is free of
Ray, vLLM and Transformers so that the contracts can be unit-tested on CPU:

* :mod:`.teacher`  -- full-vocabulary teacher distributions (shared interface,
  Route A training-trajectory extrapolation, numerical constraints, view
  mixture).
* :mod:`.chunks`   -- semantic chunking of generated responses with exact
  character-to-token alignment.
* :mod:`.verify`   -- conservative field verification (supported /
  contradicted / unknown) against the original STGR annotations.
* :mod:`.credit`   -- chunk means, finite-horizon normalized chunk
  return-to-go, equal-chunk token weights, and the reward-anchored factors
  that reweight the GRPO advantages with the chunk credit (package v5).
* :mod:`.losses`   -- PPO-style task and chunk-balanced distillation losses.
* :mod:`.schedule` -- alternating / joint stage schedules (loss coefficients
  per ``sd_mode``) and the persisted teacher-version state.
* :mod:`.health`   -- rollout health, the collapse guard and the teacher health
  gate; :mod:`.rollout_correction` -- vLLM/FSDP mismatch metrics and truncated
  importance weights (package v4).
* :mod:`.targets`  -- driver-side assembly of distillation targets for one
  rollout batch.
* :mod:`.route_c`  -- the Route C inner loop at the least absorbed chunks
  (package v6): candidate gap, branch selection, continuation requests,
  continuation values, the reward-tilted targets and their loss.
* :mod:`.torch_utils` -- small model-side helpers (response-position logits in
  float32, exact snapshot copies).
"""

TRSI_SCHEMA_VERSION = 1

__all__ = ["TRSI_SCHEMA_VERSION"]
