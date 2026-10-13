"""One Triton program per token for Qwen-style RMSNorm."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(X, W, Y, HIDDEN: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    features = tl.arange(0, BLOCK)
    valid = features < HIDDEN

    x = tl.load(X + token * HIDDEN + features, mask=valid, other=0)
    x_fp32 = x.to(tl.float32)
    mean_square = tl.sum(x_fp32 * x_fp32, axis=0) / HIDDEN
    inverse_rms = tl.rsqrt(mean_square + EPS)

    # Qwen rounds the normalized vector to its input dtype BEFORE weight scaling.
    normalized = (x_fp32 * inverse_rms).to(x.dtype)
    weight = tl.load(W + features, mask=valid, other=0).to(tl.float32)
    output = normalized.to(tl.float32) * weight
    tl.store(Y + token * HIDDEN + features, output, mask=valid)


def rmsnorm(x, weight, *, eps=1e-6):
    """Forward-only contiguous CUDA RMSNorm; preserve the input shape/dtype."""
    if x.ndim < 1 or x.numel() == 0 or not 0 < x.shape[-1] <= 16384:
        raise ValueError("nonempty input with hidden width in 1..16384 required")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError("one weight per input feature required")
    if not x.is_cuda or x.device != weight.device:
        raise ValueError("input and weight must share a CUDA device")
    if x.dtype != weight.dtype or x.dtype not in (torch.float16, torch.float32):
        raise TypeError("matching FP16 or FP32 input and weight required")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("contiguous input and weight required")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise RuntimeError("forward-only kernel; disable gradients for training tensors")

    hidden = x.shape[-1]
    tokens = x.numel() // hidden
    output = torch.empty_like(x)
    _rmsnorm[(tokens,)](
        x, weight, output, hidden, eps, triton.next_power_of_2(hidden),
        num_warps=4, enable_fp_fusion=False,
    )
    return output
