"""Accuracy-only SwiGLU arithmetic experiments; production kernel is unchanged."""

import torch
import triton
import triton.language as tl
from triton.language import math as tl_math
from triton.language.extra.cuda import libdevice

from .swiglu import validate_pair
from .triton_swiglu import default_swiglu_config

VARIANTS = {
    "fast_exp_reciprocal": (False, 0),
    "libdevice_exp_reciprocal": (True, 0),
    "fast_exp_direct": (False, 1),
    "libdevice_exp_direct": (True, 1),
    "libdevice_exp_div_rn": (True, 2),
}


@triton.jit
def _arithmetic(GATE, UP, SILU32, SILU16, PRODUCT16, ELEMENTS: tl.constexpr,
                BLOCK: tl.constexpr, LIBDEVICE_EXP: tl.constexpr, DIVISION: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < ELEMENTS
    gate = tl.load(GATE + offsets, mask=valid, other=0)
    up = tl.load(UP + offsets, mask=valid, other=0)
    x = gate.to(tl.float32)
    if LIBDEVICE_EXP:
        exponential = libdevice.exp(-x)
    else:
        exponential = tl.exp(-x)
    denominator = 1.0 + exponential
    if DIVISION == 0:
        silu32 = x * (1.0 / denominator)
    elif DIVISION == 1:
        silu32 = x / denominator
    else:
        silu32 = tl_math.div_rn(x, denominator)
    silu16 = silu32.to(gate.dtype)
    product = silu16.to(tl.float32) * up.to(tl.float32)
    tl.store(SILU32 + offsets, silu32, mask=valid)
    tl.store(SILU16 + offsets, silu16, mask=valid)
    tl.store(PRODUCT16 + offsets, product, mask=valid)


def arithmetic_stages(gate, up, *, variant):
    """Instrument one variant with unchanged geometry and FP16 boundaries."""
    validate_pair(gate, up)
    if (not gate.is_cuda or gate.dtype != torch.float16
            or not gate.is_contiguous() or not up.is_contiguous()):
        raise ValueError("matching contiguous CUDA FP16 branches required")
    if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
        raise RuntimeError("forward-only experiment; disable gradients")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    stages = {name: torch.empty_like(gate, dtype=dtype) for name, dtype in (
        ("silu_fp32", torch.float32), ("silu_fp16", torch.float16),
        ("product_fp16", torch.float16))}
    block, warps = default_swiglu_config(gate)
    use_libdevice, division = VARIANTS[variant]
    handle = _arithmetic[(triton.cdiv(gate.numel(), block),)](
        gate, up, stages["silu_fp32"], stages["silu_fp16"], stages["product_fp16"],
        gate.numel(), block, use_libdevice, division,
        num_warps=warps, enable_fp_fusion=False,
    )
    return stages, handle
