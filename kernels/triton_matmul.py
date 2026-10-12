import torch
import triton
import triton.language as tl


# Explicit candidates: compare output-tile reuse and threads per program.
# These are experimental configurations, not automatic dispatch rules.
MATMUL_SWEEP_CONFIGS = {
    f"triton_m{bm}_n{bn}_k{bk}_w{warps}": {
        "block_m": bm, "block_n": bn, "block_k": bk, "num_warps": warps,
    }
    for bm, bn, bk, warps in (
        (16, 32, 32, 4),  # Original baseline.
        (32, 32, 32, 4),  # Best measured M=128, K=N=1024 configuration so far.
        (64, 32, 32, 4),
        (32, 64, 32, 4),
        (64, 64, 32, 4),
        (64, 64, 64, 4),
        (64, 64, 32, 2),
        (64, 64, 32, 8),
        (64, 128, 32, 4),
    )
}


@triton.jit
def _tiled_matmul(
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
    totals = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_block in range(tl.cdiv(k, BLOCK_K)):
        inner = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load the next pieces along K and add their product to the output tile.
        a_locations = a_ptr + rows[:, None] * stride_am + inner[None, :] * stride_ak
        b_locations = b_ptr + inner[:, None] * stride_bk + cols[None, :] * stride_bn
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
        totals = tl.dot(a_block, b_block, totals)

    c_locations = c_ptr + rows[:, None] * n + cols[None, :]
    tl.store(
        c_locations,
        totals,
        mask=(rows[:, None] < m) & (cols[None, :] < n),
    )


def _launch_matmul(
    a: torch.Tensor, b: torch.Tensor, *,
    block_m: int = 16, block_n: int = 32, block_k: int = 32,
    num_warps: int = 4,
):
    """Validate, launch, and return the output plus compiled kernel metadata."""

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("expected two 2D matrices")
    if a.shape[1] != b.shape[0]:
        raise ValueError("inner dimensions must match")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise ValueError("both inputs must be on the same CUDA device")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("this kernel requires float16 inputs")
    if block_m not in (16, 32, 64):
        raise ValueError("block_m must be 16, 32, or 64")
    if block_k not in (32, 64):
        raise ValueError("block_k must be 32 or 64")
    if block_n not in (32, 64, 128):
        raise ValueError("block_n must be 32, 64, or 128")
    if num_warps not in (2, 4, 8):
        raise ValueError("num_warps must be 2, 4, or 8")

    m, k = a.shape
    _, n = b.shape
    if m == 0 or n == 0 or k == 0:
        raise ValueError("M, N, and K must be positive")

    output = torch.empty((m, n), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    compiled = _tiled_matmul[grid](
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
        num_warps=num_warps,
    )
    return output, compiled


def triton_matmul(
    a: torch.Tensor, b: torch.Tensor, *,
    block_m: int = 16, block_n: int = 32, block_k: int = 32,
    num_warps: int = 4,
) -> torch.Tensor:
    """Multiply nonempty float16 CUDA matrices with tiled K accumulation."""
    output, _ = _launch_matmul(
        a, b, block_m=block_m, block_n=block_n,
        block_k=block_k, num_warps=num_warps,
    )
    return output


def matmul_kernel_resources(
    a: torch.Tensor, b: torch.Tensor, *,
    block_m: int = 16, block_n: int = 32, block_k: int = 32,
    num_warps: int = 4,
) -> dict:
    """Launch once and report static resource counts; do not call while timing."""
    output, compiled = _launch_matmul(
        a, b, block_m=block_m, block_n=block_n,
        block_k=block_k, num_warps=num_warps,
    )
    grid = (triton.cdiv(a.shape[0], block_m), triton.cdiv(b.shape[1], block_n))
    return {
        "grid": list(grid),
        "programs": grid[0] * grid[1],
        "registers_per_thread": compiled.n_regs,
        "spills": compiled.n_spills,
        "shared_memory_bytes_per_program": compiled.metadata.shared,
        "warps_per_program": compiled.metadata.num_warps,
        "num_stages": compiled.metadata.num_stages,
    }
