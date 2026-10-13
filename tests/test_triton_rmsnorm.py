"""GPU RMSNorm checks for shape edges, numeric range and Qwen casting order."""

import unittest
import torch

try:
    import triton
except ImportError:
    triton = None


@unittest.skipUnless(triton is not None and torch.cuda.is_available(), "CUDA Triton required")
class TritonRmsNormTest(unittest.TestCase):
    def test_shapes_and_dtypes_against_reference(self):
        from fastinfer.kernels.rmsnorm import reference_rmsnorm
        from fastinfer.kernels.triton_rmsnorm import rmsnorm
        for dtype in (torch.float16, torch.float32):
            tolerance = 2e-3 if dtype == torch.float16 else 1e-5
            for seed in (0, 1):
                torch.manual_seed(seed)
                for shape in ((1,), (3, 65), (1, 1, 1536), (16, 1536), (128, 1536),
                              (1, 1024), (1, 4096), (1, 8192)):
                    with self.subTest(dtype=dtype, seed=seed, shape=shape):
                        x = torch.randn(shape, dtype=dtype, device="cuda")
                        weight = torch.randn(shape[-1], dtype=dtype, device="cuda")
                        expected = reference_rmsnorm(x, weight)
                        actual = rmsnorm(x, weight)
                        self.assertEqual(actual.shape, x.shape)
                        self.assertEqual(actual.dtype, x.dtype)
                        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)

    def test_zero_and_large_values(self):
        from fastinfer.kernels.rmsnorm import reference_rmsnorm
        from fastinfer.kernels.triton_rmsnorm import rmsnorm
        x = torch.tensor([[0., 0., 0.], [60000., -60000., 30000.]],
                         dtype=torch.float16, device="cuda")
        weight = torch.tensor([1., 2., -1.], dtype=x.dtype, device=x.device)
        output = rmsnorm(x, weight)
        self.assertTrue(torch.isfinite(output).all().item())
        torch.testing.assert_close(output, reference_rmsnorm(x, weight), rtol=2e-3, atol=2e-3)

    def test_rejects_noncontiguous_and_training_inputs(self):
        from fastinfer.kernels.triton_rmsnorm import rmsnorm
        x = torch.randn(2, 6, device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            rmsnorm(x[:, ::2], torch.ones(3, device="cuda", dtype=x.dtype))
        x.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "forward-only"):
            rmsnorm(x, torch.ones(6, device="cuda", dtype=x.dtype))


if __name__ == "__main__":
    unittest.main()
