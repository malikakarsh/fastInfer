"""GPU correctness checks for tiled Triton matmul."""

import unittest

import torch

try:
    import triton  # noqa: F401
except ImportError:
    triton = None


@unittest.skipUnless(triton is not None and torch.cuda.is_available(), "CUDA Triton required")
class TritonMatmulTest(unittest.TestCase):
    def test_small_known_result(self) -> None:
        from fastinfer.kernels.triton_matmul import triton_matmul

        a = torch.tensor([[1, 2, 3], [4, 5, 6]], device="cuda", dtype=torch.float16)
        b = torch.tensor([[10, 20], [30, 40], [50, 60]], device="cuda", dtype=torch.float16)
        expected = torch.tensor([[220, 280], [490, 640]], device="cuda", dtype=torch.float16)

        torch.testing.assert_close(triton_matmul(a, b), expected, rtol=0, atol=0)

    def test_masked_edges_against_pytorch(self) -> None:
        from fastinfer.kernels.triton_matmul import triton_matmul

        torch.manual_seed(0)
        a = torch.randn((17, 19), device="cuda", dtype=torch.float16)
        b = torch.randn((19, 35), device="cuda", dtype=torch.float16)

        expected = a @ b
        for block_m, block_n in ((16, 32), (16, 64), (32, 32)):
            with self.subTest(block_m=block_m, block_n=block_n):
                torch.testing.assert_close(
                    triton_matmul(
                        a, b, block_m=block_m, block_n=block_n, block_k=64
                    ), expected,
                    rtol=1e-2, atol=1e-2,
                )

    def test_multiple_k_blocks_against_pytorch(self) -> None:
        from fastinfer.kernels.triton_matmul import triton_matmul

        torch.manual_seed(1)
        for m, k, n in ((17, 65, 35), (1, 1024, 256)):
            a = torch.randn((m, k), device="cuda", dtype=torch.float16)
            b = torch.randn((k, n), device="cuda", dtype=torch.float16)
            expected = a @ b
            for block_k in (32, 64):
                with self.subTest(shape=(m, k, n), block_k=block_k):
                    torch.testing.assert_close(
                        triton_matmul(a, b, block_k=block_k), expected,
                        rtol=1e-2, atol=1e-2,
                    )

    def test_large_k_against_float32_reference(self) -> None:
        from fastinfer.kernels.triton_matmul import triton_matmul

        torch.manual_seed(0)
        a = torch.randn((128, 1024), device="cuda", dtype=torch.float16)
        b = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
        expected = a.float() @ b.float()

        for block_m, block_k in ((16, 32), (16, 64), (32, 32)):
            with self.subTest(block_m=block_m, block_k=block_k):
                torch.testing.assert_close(
                    triton_matmul(a, b, block_m=block_m, block_k=block_k).float(),
                    expected,
                    rtol=1e-2, atol=1e-2,
                )

    def test_configuration_sweep_against_float32_reference(self) -> None:
        from fastinfer.kernels.triton_matmul import MATMUL_SWEEP_CONFIGS, triton_matmul

        previous_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            torch.manual_seed(2)
            for m, k, n in ((65, 67, 131), (128, 1024, 1024)):
                # Transposed inputs exercise strides and every partial tile edge.
                if k == 67:
                    a = torch.randn((k, m), device="cuda", dtype=torch.float16).t()
                    b = torch.randn((n, k), device="cuda", dtype=torch.float16).t()
                else:
                    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
                    b = torch.randn((k, n), device="cuda", dtype=torch.float16)
                expected = a.float() @ b.float()
                for name, config in MATMUL_SWEEP_CONFIGS.items():
                    with self.subTest(shape=(m, k, n), configuration=name):
                        torch.testing.assert_close(
                            triton_matmul(a, b, **config).float(), expected,
                            rtol=1e-2, atol=1e-2,
                        )
        finally:
            torch.set_float32_matmul_precision(previous_precision)


if __name__ == "__main__":
    unittest.main()
