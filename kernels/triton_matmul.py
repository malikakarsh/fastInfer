import torch
import triton
import triton.language as tl


@triton.jit
def _single_k_block_matmul(
    a_ptr,
    b_ptr,
    c_ptr,
    m,
    n,
    k,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program owns one BLOCK_M by BLOCK_N rectangle of the output.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)

    # These blocks apply a[row, inner] and b[inner, col] across the tile.
    a_locations = a_ptr + rows[:, None] * stride_am + inner[None, :] * stride_ak
    b_locations = b_ptr + inner[:, None] * stride_bk + cols[None, :] * stride_bn

    """
    4x4
    (0,0) -> rows (0,1), cols (0,1)
    (1,0) -> rows (2,3), cols (0,1)
    (0,1) -> rows (0,1), cols (2,3)
    (1,1) -> rows (2,3), cols (2,3)

    A -> [4,2]
    B -> [2,4]

    stride_am = 2
    stride_ak = 1
    stride_bk = 4
    stride_bn = 1
    BLOCK_M = 2
    BLOCK_N = 2
    BLOCK_K = 2

    inner = [0,1]

    rows = [2,3]
    cols = [0,1]

    a_locations = [[2],[3]] * 2 + inner[[0,1]]*1
                = [[4],[6]] + [[0,1]]
                = [[4,5]
                   [6,7]]
    b_location = [[0],[1]] * 4 + inner[[0,1]]*1
               = [[0],[4]] + [[0,1]]
               = [[0,1]
                  [4,5]]
    """

    a_block = tl.load(
        a_locations,
        mask=(rows[:, None] < m) & (inner[None, :] < k),
        other=0.0,
    )
    b_block = tl.load(
        b_locations,
        mask=(inner[:, None] < k) & (cols[None, :] < n),
        other=0.0,
    )

    totals = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    totals = tl.dot(a_block, b_block, totals)

    c_locations = c_ptr + rows[:, None] * n + cols[None, :]
    tl.store(
        c_locations,
        totals,
        mask=(rows[:, None] < m) & (cols[None, :] < n),
    )


def triton_matmul_one_k_block(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply float16 CUDA matrices with K at most 32."""

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("expected two 2D matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError("inner dimensions must match")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("both inputs must be on the same CUDA device")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("this first-stage kernel requires float16 inputs")

    m, k = a.shape
    _, n = b.shape
    if m == 0 or n == 0 or not 1 <= k <= 32:
        raise ValueError("M and N must be positive, and K must be between 1 and 32")

    output = torch.empty((m, n), device=a.device, dtype=torch.float16)
    block_m, block_n, block_k = 16, 32, 32
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _single_k_block_matmul[grid](
        a,
        b,
        output,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
    )
    return output
