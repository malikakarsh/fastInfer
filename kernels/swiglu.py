"""PyTorch reference for the elementwise stage of Qwen's SwiGLU MLP."""

import torch
import torch.nn.functional as F


def validate_pair(gate, up):
    """Require equal, nonempty floating tensors; do not broadcast branches."""
    if gate.ndim < 1 or gate.numel() == 0 or gate.shape != up.shape:
        raise ValueError("gate and up must have the same nonempty shape")
    if gate.device != up.device or gate.dtype != up.dtype:
        raise ValueError("gate and up must share a device and dtype")
    if gate.dtype not in (torch.float16, torch.float32):
        raise TypeError("FP16 or FP32 tensors required")


def reference_swiglu(gate, up):
    """Return SiLU(gate) * up, preserving the intermediate input-dtype rounding.

    Finite input tensors are expected. This is only the activation/multiply
    stage; the learned gate/up/down projections are separate operations.
    """
    validate_pair(gate, up)
    activated_gate = F.silu(gate)
    return activated_gate * up
