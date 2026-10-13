"""Scoped, experimental replacement of one Qwen decode projection."""

from contextlib import contextmanager

import torch


def eligible_decode_input(linear, x):
    """Only dispatch the exact FP16 single-token shape validated on T4."""
    return (
        not torch.is_grad_enabled()
        and not linear.training
        and linear.bias is None
        and tuple(linear.weight.shape) == (8960, 1536)
        and tuple(x.shape) == (1, 1, 1536)
        and x.dtype == linear.weight.dtype == torch.float16
        and x.is_cuda
        and x.device == linear.weight.device
        and x.is_contiguous()
        and linear.weight.is_contiguous()
    )


@contextmanager
def use_decode_gemv(linear, *, kernel=None):
    """Temporarily dispatch eligible calls; use only around cached decode.

    Other calls fall back to the original forward. This mutates one module and
    is intended for serial experiments, not concurrent serving or training.
    """
    if kernel is None:
        from fastinfer.kernels.triton_gemv import gemv
        kernel = gemv

    original = linear.forward
    had_override = "forward" in linear.__dict__
    override = linear.__dict__.get("forward")
    counts = {"gemv_calls": 0, "fallback_calls": 0}

    def forward(input):
        if eligible_decode_input(linear, input):
            counts["gemv_calls"] += 1
            result = kernel(input.reshape(1, 1536), linear.weight,
                            block_rows=1, num_warps=4)
            return result.reshape(1, 1, 8960)
        counts["fallback_calls"] += 1
        return original(input)

    linear.forward = forward
    try:
        yield counts
    finally:
        if had_override:
            linear.forward = override
        else:
            del linear.forward
