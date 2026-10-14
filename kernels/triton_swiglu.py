"""Forward-only fused SiLU and elementwise multiplication."""

import torch
import triton
import triton.language as tl

from .swiglu import validate_pair
from .swiglu_config import choose_swiglu_config

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


def default_swiglu_config(gate):
    """Return the shape/dtype/hardware/software-scoped automatic setting."""
    return choose_swiglu_config(
        gate.shape, dtype=str(gate.dtype).removeprefix("torch."),
        gpu_name=torch.cuda.get_device_name(gate.device),
        torch_version=torch.__version__, triton_version=triton.__version__,
    )


def swiglu(gate, up, *, block_size=None, num_warps=None):
    """Process equal contiguous CUDA FP16/FP32 tensors, with no backward pass.

    Omit both settings for automatic selection:512/4 for the confirmed T4
    FP16 shape16x8960, otherwise256/4. Supplying either setting disables
    automatic selection and uses the original default for the other setting.
    Finite inputs are required; no bitwise-agreement promise.
    """
    validate_pair(gate, up)
    if not gate.is_cuda:
        raise ValueError("CUDA inputs required")
    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("contiguous inputs required")
    if torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad):
        raise RuntimeError("forward-only kernel; disable gradients for training tensors")

    if block_size is None and num_warps is None:
        block_size, num_warps = default_swiglu_config(gate)
    else:
        block_size = BLOCK_SIZE if block_size is None else block_size
        num_warps = NUM_WARPS if num_warps is None else num_warps
    if block_size not in (128, 256, 512, 1024):
        raise ValueError("block_size must be 128, 256, 512 or 1024")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be 1, 2, 4 or 8")

    output = torch.empty_like(gate)
    elements = gate.numel()
    _swiglu[(triton.cdiv(elements, block_size),)](
        gate, up, output, elements, block_size,
        num_warps=num_warps, enable_fp_fusion=False,
    )
    return output
