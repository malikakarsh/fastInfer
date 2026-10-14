"""Forward-only fused SiLU and elementwise multiplication."""

import torch
import triton
import triton.language as tl

from .swiglu import validate_pair

BLOCK_SIZE = 256
NUM_WARPS = 4


@triton.jit
def _swiglu(GATE, UP, OUTPUT, ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < ELEMENTS

    gate = tl.load(GATE + offsets, mask=valid, other=0)
    up = tl.load(UP + offsets, mask=valid, other=0)
    gate_fp32 = gate.to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-gate_fp32))
    activated_fp32 = gate_fp32 * sigmoid

    # Match eager PyTorch: SiLU produces an input-dtype tensor before multiply.
    activated = activated_fp32.to(gate.dtype)
    result = activated.to(tl.float32) * up.to(tl.float32)
    tl.store(OUTPUT + offsets, result, mask=valid)


def swiglu(gate, up):
    """Process equal contiguous CUDA FP16/FP32 tensors, with no backward pass.

    Fixed 256-element chunks and four warps are initial settings, not tuned
    choices. Finite inputs are required; no bitwise-agreement promise.
    """
    validate_pair(gate, up)
    if not gate.is_cuda:
        raise ValueError("CUDA inputs required")
    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("contiguous inputs required")
    if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
        raise RuntimeError("forward-only kernel; disable gradients for training tensors")

    output = torch.empty_like(gate)
    elements = gate.numel()
    _swiglu[(triton.cdiv(elements, BLOCK_SIZE),)](
        gate, up, output, elements, BLOCK_SIZE,
        num_warps=NUM_WARPS, enable_fp_fusion=False,
    )
    return output
