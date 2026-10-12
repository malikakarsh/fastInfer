"""Single-row FP16 linear projection with contiguous weights W[N, K]."""

import torch
import triton
import triton.language as tl


GEMV_CONFIGS = {
    f"gemv_rows{rows}": {"block_rows": rows, "num_warps": 4}
    for rows in (1, 4, 8)
}


@triton.jit
def _gemv(A, W, C, N: tl.constexpr, K: tl.constexpr,
          ROWS: tl.constexpr, PAD_K: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    inner = tl.arange(0, PAD_K)
    x = tl.load(A + inner, inner < K, other=0).to(tl.float32)
    weights = tl.load(W + rows[:, None] * K + inner[None, :],
                      (rows[:, None] < N) & (inner[None, :] < K),
                      other=0).to(tl.float32)
    # Parallel sum along each contiguous weight row; no padded output rows.
    totals = tl.sum(weights * x[None, :], axis=1)
    tl.store(C + rows, totals, rows < N)


def gemv(a, weight, *, block_rows=4, num_warps=4):
    """Return a @ weight.T. Requires contiguous FP16 a[1,K], weight[N,K]."""
    if a.ndim != 2 or weight.ndim != 2 or a.shape[0] != 1:
        raise ValueError("expected a[1,K] and weight[N,K]")
    n, k = weight.shape
    if a.shape[1] != k or n <= 0 or not 0 < k <= 16384:
        raise ValueError("matching K in 1..16384 and positive N required")
    if not a.is_cuda or not weight.is_cuda or a.device != weight.device:
        raise ValueError("inputs must share a CUDA device")
    if a.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("FP16 inputs required")
    if not a.is_contiguous() or not weight.is_contiguous():
        raise ValueError("contiguous input and stored weights required")
    if block_rows not in (1, 4, 8) or num_warps != 4:
        raise ValueError("supported rows: 1,4,8; warps: 4")
    output = torch.empty((1, n), device=a.device, dtype=a.dtype)
    _gemv[(triton.cdiv(n, block_rows),)](
        a, weight, output, n, k, block_rows, triton.next_power_of_2(k),
        num_warps=num_warps)
    return output
