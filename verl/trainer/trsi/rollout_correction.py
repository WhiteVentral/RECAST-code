"""Sampler/trainer mismatch: vLLM's sampled-token log-probabilities against the FSDP policy.

The rollout samples with vLLM (bf16 kernels, its own attention and sampling
code); ``pi_old`` and the PPO ratio are computed by the FSDP actor.  The two
disagree slightly even with identical weights, and the disagreement can grow
during training (Yao et al., 2025, "Your efficient RL framework secretly brings
you off-policy RL training"; Liu, Li et al., 2025, "When speed kills
stability").  With ``worker.rollout.calculate_log_probs=true`` the rollout
returns ``rollout_log_probs`` = log mu(y_t) of the sampled tokens, and the
driver records per step (``mismatch/*``, over response tokens)::

    kl_k1            mean(log mu - log pi_old)                 KL(mu || pi_old), unbiased under mu
    kl_k3            mean(r - 1 - log r), r = pi_old / mu      KL(mu || pi_old), low variance, >= 0
    abs_mean / abs_max of log mu - log pi_old
    prob_diff_mean   mean |mu - pi_old|                        (verl: training/rollout_probs_diff_mean)
    log_ppl_abs_diff mean over responses of |mean_t (log pi_old - log mu)|
    seq_abs_max      max over responses of |sum_t (log pi_old - log mu)|
    extreme_fraction tokens with r > 2 or r < 0.5
    ess              (sum r)^2 / (N sum r^2), the normalized effective sample size of the token weights
    is_weight_mean / is_weight_max / is_capped_fraction of r and of min(r, C)

Reference values (dense bf16 models; R3, arXiv 2510.11370, and vLLM's train-inference consistency notes):
kl_k3 about 5e-4 to 2e-3 and prob_diff_mean below 0.005 are normal; a sustained upward trend of kl_k3,
prob_diff_mean above 0.01 or a growing extreme_fraction indicate a numerical problem worth correcting.

``rsi.rollout_correction=tis`` multiplies every surrogate term by the detached
``min(r, C)`` (truncated importance sampling, ``C = rsi.rollout_correction_cap``).

vLLM's V1 sampler returns, by default, the log-probabilities of the raw model
distribution: before temperature, top-k and top-p.  With temperature 1 (the
shipped configs) this is the untruncated distribution that FSDP computes, so
the metrics measure numerical disagreement only; the top-k/top-p truncation of
the sampler is not corrected here.
"""

from __future__ import annotations

import torch


__all__ = ["rollout_mismatch_metrics", "truncated_is_weights"]


def _log_ratio(old_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tuple(old_log_probs.shape) != tuple(rollout_log_probs.shape) or tuple(mask.shape) != tuple(old_log_probs.shape):
        raise ValueError(
            "old_log_probs, rollout_log_probs and response_mask must have the same shape, got "
            f"{tuple(old_log_probs.shape)}, {tuple(rollout_log_probs.shape)}, {tuple(mask.shape)}."
        )
    return (old_log_probs.float() - rollout_log_probs.float()) * mask


def truncated_is_weights(
    old_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor, response_mask: torch.Tensor, cap: float
) -> torch.Tensor:
    """Per-token ``min(pi_old / mu, cap)`` on response tokens (0 elsewhere), detached."""

    if not float(cap) >= 1.0:
        raise ValueError(f"The importance-weight cap must be >= 1, got {cap}.")
    mask = response_mask.to(torch.float32)
    log_ratio = _log_ratio(old_log_probs.detach(), rollout_log_probs.detach(), mask)
    # Clamp before exp: a token vLLM considered nearly impossible would overflow, and min(r, cap) is all we keep.
    return torch.exp(log_ratio.clamp(max=float(torch.log(torch.tensor(float(cap)))))) * mask


def rollout_mismatch_metrics(
    old_log_probs: torch.Tensor, rollout_log_probs: torch.Tensor, response_mask: torch.Tensor, cap: float
) -> dict[str, float]:
    mask = response_mask.to(torch.float32)
    tokens = float(mask.sum().item())
    if tokens <= 0:
        return {}
    log_ratio = _log_ratio(old_log_probs.detach(), rollout_log_probs.detach(), mask)  # log pi_old - log mu
    valid = mask.bool()
    values = log_ratio[valid]
    ratio = torch.exp(values.clamp(max=50.0))
    capped = ratio.clamp(max=float(cap))
    old_p = torch.exp(old_log_probs.float()[valid])
    mu = torch.exp(rollout_log_probs.float()[valid])
    sequence = log_ratio.sum(dim=-1)
    lengths = mask.sum(dim=-1)
    rows = lengths > 0
    per_token_mean = sequence[rows] / lengths[rows]
    return {
        "mismatch/kl_k1": float((-values).mean().item()),
        "mismatch/kl_k3": float((ratio - 1.0 - values).mean().item()),
        "mismatch/abs_mean": float(values.abs().mean().item()),
        "mismatch/abs_max": float(values.abs().max().item()),
        "mismatch/prob_diff_mean": float((old_p - mu).abs().mean().item()),
        "mismatch/seq_abs_max": float(sequence.abs().max().item()),
        "mismatch/log_ppl_abs_diff": float(per_token_mean.abs().mean().item()),
        "mismatch/extreme_fraction": float(((ratio > 2.0) | (ratio < 0.5)).float().mean().item()),
        "mismatch/ess": float((ratio.sum() ** 2 / (ratio.numel() * (ratio**2).sum())).item()),
        "mismatch/is_weight_mean": float(ratio.mean().item()),
        "mismatch/is_weight_max": float(ratio.max().item()),
        "mismatch/is_capped_mean": float(capped.mean().item()),
        "mismatch/is_capped_fraction": float((ratio > float(cap)).float().mean().item()),
    }
