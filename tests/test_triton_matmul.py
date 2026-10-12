"""GPU correctness checks for the first-stage Triton matmul."""

import unittest

import torch

try:
    import triton  # noqa: F401
except ImportError:
    triton = None


@unittest.skipUnless(triton is not None and torch.cuda.is_available(), "CUDA Triton required")
class TritonMatmulTest(unittest.TestCase):
    def test_small_known_result(self) -> None:
        from fastinfer.kernels.triton_matmul import triton_matmul_one_k_block

        a = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda", dtype=torch.float16)
        b = torch.tensor([[10, 20], [30, 40], [50, 60]], device="cuda", dtype=torch.float16)
        expected = torch.tensor([[220, 280], [490, 640]], device="cuda", dtype=torch.float16)

        torch.testing.assert_close(triton_matmul_one_k_block(a, b), expected, rtol=0, atol=0)

    def test_masked_edges_against_pytorch(self) -> None:
        from fastinfer.kernels.triton_matmul import triton_matmul_one_k_block

        torch.manual_seed(0)
        a = torch.randn((17, 19), device="cuda", dtype=torch.float16)
        b = torch.randn((19, 35), device="cuda", dtype=torch.float16)

        torch.testing.assert_close(
            triton_matmul_one_k_block(a, b), a @ b, rtol=1e-2, atol=1e-2
        )


if __name__ == "__main__":
    unittest.main()
