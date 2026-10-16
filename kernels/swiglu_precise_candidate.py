"""Experimental FP16 SwiGLU using library exp and round-to-nearest division.

Returns only the product: diagnostic intermediate stores are deliberately absent.
Uses the existing launch policy; agreement must be measured, not assumed.
"""

import torch
import triton
import triton.language as tl
from triton.language import math as tl_math
from triton.language.extra.cuda import libdevice

from .swiglu import validate_pair
from .triton_swiglu import default_swiglu_config


@triton.jit
def _swiglu_precise(GATE, UP, OUTPUT, ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < ELEMENTS
    gate = tl.load(GATE + offsets, mask=valid, other=0)
    up = tl.load(UP + offsets, mask=valid, other=0).to(tl.float32)
    x = gate.to(tl.float32)
    activated = tl_math.div_rn(x, 1.0 + libdevice.exp(-x)).to(gate.dtype)
    product = activated.to(tl.float32) * up
    tl.store(OUTPUT + offsets, product, mask=valid)


def swiglu_precise(gate, up):
    """Forward-only candidate for matching contiguous CUDA FP16 branches.

Finite inputs required. Keeps eager's FP16 rounding after SiLU and multiply.
Does not promise exact agreement on inputs or devices not yet tested.
"""
    validate_pair(gate, up)
    if (not gate.is_cuda or gate.dtype != torch.float16
            or not gate.is_contiguous() or not up.is_contiguous()):
        raise ValueError("matching contiguous CUDA FP16 branches required")
    if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
        raise RuntimeError("forward-only candidate; disable gradients")
    output = torch.empty_like(gate)
    block, warps = default_swiglu_config(gate)
    _swiglu_precise[(triton.cdiv(gate.numel(), block),)](
        gate, up, output, gate.numel(), block,
        num_warps=warps, enable_fp_fusion=False,
    )
    return output
