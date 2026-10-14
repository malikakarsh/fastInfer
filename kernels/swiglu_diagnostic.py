"""Instrument the current SwiGLU arithmetic; never use this kernel for timing."""

import torch
import triton
import triton.language as tl

from .swiglu import validate_pair
from .triton_swiglu import default_swiglu_config


@triton.jit
def _stages(GATE, UP, SIGMOID, SILU32, SILU16, PRODUCT32, PRODUCT16,
            ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < ELEMENTS
    gate = tl.load(GATE + offsets, mask=valid, other=0)
    up = tl.load(UP + offsets, mask=valid, other=0)
    gate_fp32 = gate.to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate_fp32))
    silu_fp32 = gate_fp32 * sigmoid
    silu_fp16 = silu_fp32.to(gate.dtype)
    product_fp32 = silu_fp16.to(tl.float32) * up.to(tl.float32)
    tl.store(SIGMOID + offsets, sigmoid, mask=valid)
    tl.store(SILU32 + offsets, silu_fp32, mask=valid)
    tl.store(SILU16 + offsets, silu_fp16, mask=valid)
    tl.store(PRODUCT32 + offsets, product_fp32, mask=valid)
    tl.store(PRODUCT16 + offsets, product_fp32, mask=valid)


def triton_swiglu_stages(gate, up):
    """Retain FP32 stages and FP16 rounding boundaries; extra stores may affect codegen."""
    validate_pair(gate, up)
    if (not gate.is_cuda or gate.dtype != torch.float16
            or not gate.is_contiguous() or not up.is_contiguous()):
        raise ValueError("matching contiguous CUDA FP16 branches required")
    if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
        raise RuntimeError("diagnostic is forward-only; disable gradients")
    stages = {name: torch.empty_like(gate, dtype=dtype) for name, dtype in (
        ("sigmoid_fp32", torch.float32), ("silu_fp32", torch.float32),
        ("silu_fp16", torch.float16), ("product_fp32", torch.float32),
        ("product_fp16", torch.float16))}
    block, warps = default_swiglu_config(gate)
    _stages[(triton.cdiv(gate.numel(), block),)](
        gate, up, *(stages[name] for name in (
            "sigmoid_fp32", "silu_fp32", "silu_fp16", "product_fp32", "product_fp16")),
        gate.numel(), block, num_warps=warps, enable_fp_fusion=False,
    )
    return stages
