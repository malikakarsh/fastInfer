"""Initial fused softmax: one Triton program per contiguous matrix row."""

import torch
import triton
import triton.language as tl

from .softmax_config import choose_softmax_warps


@triton.jit
def _softmax(X, Y, COLUMNS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    valid = columns < COLUMNS

    scores = tl.load(X + row * COLUMNS + columns, mask=valid, other=-float("inf"))
    scores_fp32 = scores.to(tl.float32)
    shifted_scores = scores_fp32 - tl.max(scores_fp32, axis=0)
    exponentials = tl.exp(shifted_scores)
    row_total = tl.sum(exponentials, axis=0)
    probabilities = exponentials / row_total
    tl.store(Y + row * COLUMNS + columns, probabilities, mask=valid)


def default_softmax_warps(x):
    """Select a confirmed configuration; explicit settings bypass selection."""
    return choose_softmax_warps(
        x.shape, dtype=str(x.dtype).removeprefix("torch."),
        gpu_name=torch.cuda.get_device_name(x.device),
        torch_version=torch.__version__, triton_version=triton.__version__,
    )


def softmax(x, *, num_warps=None):
    """Forward-only softmax for finite contiguous CUDA FP16/FP32 matrices.

    None selects confirmed T4 FP16 shapes; other cases use four warps.
    Explicit num_warps=1/2/4/8 retains manual control.
    Normalize each row in FP32, then store in the input dtype. This first
    version supports row widths 1..8192; it does not promise bitwise agreement.
    """
    if x.ndim != 2 or x.numel() == 0 or not 0 < x.shape[1] <= 8192:
        raise ValueError("a nonempty matrix with row width in 1..8192 is required")
    if not x.is_cuda:
        raise ValueError("CUDA input required")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError("FP16 or FP32 scores required")
    if not x.is_contiguous():
        raise ValueError("contiguous input required")
    if torch.is_grad_enabled() and x.requires_grad:
        raise RuntimeError("forward-only kernel; disable gradients for training tensors")
    if num_warps is None:
        num_warps = default_softmax_warps(x)
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be 1, 2, 4 or 8")

    rows, columns = x.shape
    output = torch.empty_like(x)
    _softmax[(rows,)](
        x, output, columns, triton.next_power_of_2(columns),
        num_warps=num_warps, enable_fp_fusion=False,
    )
    return output
