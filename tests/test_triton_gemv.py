"""Check real and masked dimensions for the FP32-reduction GEMV kernel."""

import unittest
import torch

try:
    import triton
except ImportError:
    triton = None


@unittest.skipUnless(triton is not None and torch.cuda.is_available(), "CUDA Triton required")
class GemvTest(unittest.TestCase):
    def test_reference_and_edges(self):
        from fastinfer.kernels.triton_gemv import gemv, GEMV_CONFIGS
        previous = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")
        try:
            for seed in (0, 1):
                torch.manual_seed(seed)
                for n, k in ((1, 1), (13, 65), (256, 1536), (8960, 1536), (17, 8960)):
                    a = torch.randn((1, k), device="cuda", dtype=torch.float16)
                    w = torch.randn((n, k), device="cuda", dtype=torch.float16)
                    expected = a.float() @ w.float().t()
                    for name, config in GEMV_CONFIGS.items():
                        with self.subTest(seed=seed, n=n, k=k, config=name):
                            torch.testing.assert_close(gemv(a, w, **config).float(),
                                                       expected, rtol=.01, atol=.01)
        finally:
            torch.set_float32_matmul_precision(previous)

    def test_cancellation(self):
        from fastinfer.kernels.triton_gemv import gemv, GEMV_CONFIGS
        a = torch.ones((1, 4), device="cuda", dtype=torch.float16)
        w = torch.tensor([[1000, .25, -1000, .25]], device="cuda", dtype=torch.float16)
        expected = torch.tensor([[.5]], device="cuda", dtype=torch.float16)
        for config in GEMV_CONFIGS.values():
            torch.testing.assert_close(gemv(a, w, **config), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
