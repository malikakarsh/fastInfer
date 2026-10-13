"""Experimental one-token RMSNorm following a PyTorch 2.5.1 reduction grouping.

For a contiguous FP32 mean with one output and width 1536, Reduce.cuh selects
256 logical lanes, four values per vector, and no inter-block reduction. This
kernel models that source path; GPU checks must establish actual agreement.
It is deliberately separate from the production RMSNorm kernel.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _add_halves(values, SIZE: tl.constexpr):
    halves = tl.reshape(values, (2, SIZE // 2))
    left, right = tl.split(tl.trans(halves))
    return left + right


@triton.jit
def _add_adjacent(values, SIZE: tl.constexpr):
    left, right = tl.split(tl.reshape(values, (SIZE // 2, 2)))
    return left + right


@triton.jit
def _candidate(X, W, Y, MEAN, EPS: tl.constexpr, SAVE_MEAN: tl.constexpr):
    features = tl.arange(0, 2048)
    valid = features < 1536
    x = tl.load(X + features, mask=valid, other=0)
    x_fp32 = x.to(tl.float32)
    squares = x_fp32 * x_fp32

    # Model 256 logical lanes, each accumulating four separate vector columns.
    # Lanes 0..127 also have a second vector at feature offset 1024.
    vectors = tl.reshape(squares, (2, 256, 4))
    first_vector, second_vector = tl.split(tl.permute(vectors, (1, 2, 0)))
    columns = first_vector + second_vector
    even, odd = tl.split(tl.reshape(columns, (256, 2, 2)))
    a0, a2 = tl.split(even)
    a1, a3 = tl.split(odd)
    partial = ((a0 + a1) + a2) + a3

    # Preserve lane zero's arithmetic from block offsets 128,64,32.
    partial = _add_halves(partial, 256)
    partial = _add_halves(partial, 128)
    partial = _add_halves(partial, 64)
    # Adjacent pairing reproduces lane zero's tree for shuffle offsets 1,2,4,8,16.
    # Reshape/transpose/split are available in the pinned Triton 3.1 release.
    partial = _add_adjacent(partial, 32)
    partial = _add_adjacent(partial, 16)
    partial = _add_adjacent(partial, 8)
    partial = _add_adjacent(partial, 4)
    partial = _add_adjacent(partial, 2)
    total = tl.sum(partial, axis=0)
    factor = tl.full((), 1.0 / 1536, tl.float32)
    mean_square = total * factor

    inverse_rms = tl.rsqrt(mean_square + EPS)
    normalized = (x_fp32 * inverse_rms).to(x.dtype)
    weight = tl.load(W + features, mask=valid, other=0).to(tl.float32)
    output = normalized.to(tl.float32) * weight
    tl.store(Y + features, output, mask=valid)
    if SAVE_MEAN:
        tl.store(MEAN, mean_square)


def rmsnorm_candidate(x, weight, *, eps=1e-6, return_mean=False):
    """Test candidate only: exactly one contiguous CUDA FP16 token, hidden=1536."""
    if (x.ndim < 1 or x.shape[-1] != 1536 or x.numel() != 1536
            or weight.shape != (1536,) or not x.is_cuda or weight.device != x.device
            or x.dtype != torch.float16 or weight.dtype != x.dtype
            or not x.is_contiguous() or not weight.is_contiguous()):
        raise ValueError("one contiguous CUDA FP16 token of width 1536 and matching weight required")
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative")
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        raise RuntimeError("experimental forward-only kernel; disable gradients")
    output = torch.empty_like(x)
    mean = torch.empty(1, device=x.device, dtype=torch.float32) if return_mean else output
    _candidate[(1,)](x, weight, output, mean, eps, return_mean,
                     num_warps=4, enable_fp_fusion=False)
    return (output, mean) if return_mean else output
