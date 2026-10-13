"""Readable stable softmax reference; normalize along the final dimension."""

import torch


def reference_softmax(x: torch.Tensor):
    """Softmax for nonempty finite FP16/FP32 scores, preserving shape and dtype.

    Calculate in FP32 and round the final probabilities to the input dtype.
    This is an inspectable reference, not an optimized GPU implementation.
    """
    if x.ndim < 1 or x.numel() == 0:
        raise ValueError("a nonempty input with at least one dimension is required")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError("FP16 or FP32 scores required")

    scores = x.float()
    row_maximum = scores.max(dim=-1, keepdim=True).values
    shifted_scores = scores - row_maximum
    exponentials = torch.exp(shifted_scores)
    row_total = exponentials.sum(dim=-1, keepdim=True)
    probabilities = exponentials / row_total
    return probabilities.to(x.dtype)
