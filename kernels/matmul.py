import torch


def reference_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply a 2D [M, K] matrix by a 2D [K, N] matrix using PyTorch."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("reference_matmul expects two 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("inner dimensions must match")
    return a @ b
