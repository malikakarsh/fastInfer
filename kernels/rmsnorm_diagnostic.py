"""Instrument RMSNorm arithmetic for investigation, never for timing or dispatch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _stages(X, W, SQUARES, SCALARS, NORMALIZED, ROUNDED, Y,
            HIDDEN: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    features = tl.arange(0, BLOCK)
    valid = features < HIDDEN
    x = tl.load(X + features, mask=valid, other=0)
    x_fp32 = x.to(tl.float32)
    squares = x_fp32 * x_fp32
    mean_square = tl.sum(squares, axis=0) / HIDDEN
    variance_with_eps = mean_square + EPS
    inverse_rms = tl.rsqrt(variance_with_eps)
    normalized_fp32 = x_fp32 * inverse_rms
    normalized = normalized_fp32.to(x.dtype)
    weight = tl.load(W + features, mask=valid, other=0).to(tl.float32)
    output = normalized.to(tl.float32) * weight

    tl.store(SQUARES + features, squares, mask=valid)
    tl.store(SCALARS, mean_square)
    tl.store(SCALARS + 1, variance_with_eps)
    tl.store(SCALARS + 2, inverse_rms)
    tl.store(NORMALIZED + features, normalized_fp32, mask=valid)
    tl.store(ROUNDED + features, normalized, mask=valid)
    tl.store(Y + features, output, mask=valid)


def triton_stages(x, weight, *, eps):
    """Capture one FP16 token; extra stores can affect compiler decisions."""
    if (not x.is_cuda or x.dtype != torch.float16 or not x.is_contiguous()
            or x.numel() != x.shape[-1] or not 0 < x.numel() <= 16384
            or weight.shape != (x.numel(),) or weight.dtype != x.dtype
            or weight.device != x.device or not weight.is_contiguous()):
        raise ValueError("one contiguous CUDA FP16 token and matching weight required")
    squares = torch.empty_like(x, dtype=torch.float32)
    scalars = torch.empty(3, device=x.device, dtype=torch.float32)
    normalized = torch.empty_like(squares)
    rounded = torch.empty_like(x)
    output = torch.empty_like(x)
    _stages[(1,)](
        x, weight, squares, scalars, normalized, rounded, output,
        x.numel(), eps, triton.next_power_of_2(x.numel()),
        num_warps=4, enable_fp_fusion=False,
    )
    return {
        "squared_values": squares,
        "mean_square": scalars[0:1],
        "mean_square_plus_eps": scalars[1:2],
        "inverse_rms": scalars[2:3],
        "normalized_fp32": normalized,
        "normalized_fp16": rounded,
        "weighted_output_fp16": output,
    }
