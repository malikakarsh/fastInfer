"""Readable PyTorch reference for Qwen's root mean square normalization."""

import math

import torch


def reference_rmsnorm(x: torch.Tensor, weight: torch.Tensor, *, eps=1e-6):
    """Normalize each token along its final dimension, then apply learned weights.

    Match Qwen's arithmetic order: calculate normalization in FP32, cast the
    normalized vector back to the input dtype, then multiply by the weight.
    Input can be [hidden], [tokens, hidden], or [batch, tokens, hidden].
    """
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("input requires a nonempty final dimension")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("weight must contain one value per input feature")
    if x.device != weight.device or x.dtype != weight.dtype:
        raise ValueError("input and weight must share a device and dtype")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("FP16, BF16 or FP32 tensors required")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")

    input_fp32 = x.float()
    squared_values = input_fp32.square()
    mean_square = squared_values.mean(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(mean_square + eps)
    normalized_fp32 = input_fp32 * inverse_rms
    normalized = normalized_fp32.to(x.dtype)
    return normalized * weight
