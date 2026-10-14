"""GPU checks for row independence, masked padding, and numerical range."""

import unittest

import torch

try:
    import triton
except ImportError:
    triton = None


@unittest.skipUnless(triton is not None and torch.cuda.is_available(), "CUDA Triton required")
class TritonSoftmaxTest(unittest.TestCase):
    def check_output(self, scores, *, num_warps=4):
        from fastinfer.kernels.softmax import reference_softmax
        from fastinfer.kernels.triton_softmax import softmax

        actual = softmax(scores, num_warps=num_warps)
        native = torch.softmax(scores, dim=-1)
        oracle = torch.softmax(scores.double(), dim=-1).float()
        rtol, atol = (2e-3, 2e-6) if scores.dtype == torch.float16 else (1e-5, 1e-7)
        self.assertEqual(actual.shape, scores.shape)
        self.assertEqual(actual.dtype, scores.dtype)
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertTrue((actual >= 0).all().item())
        torch.testing.assert_close(actual, native, rtol=rtol, atol=atol)
        torch.testing.assert_close(actual, reference_softmax(scores), rtol=rtol, atol=atol)
        torch.testing.assert_close(actual.float(), oracle, rtol=rtol, atol=atol)
        torch.testing.assert_close(actual.float().sum(dim=-1), torch.ones(scores.shape[0], device="cuda"),
                                   rtol=0, atol=5e-4 if scores.dtype == torch.float16 else 1e-6)
        return actual

    def test_shapes_and_dtypes(self):
        for dtype in (torch.float16, torch.float32):
            for seed in (0, 1):
                torch.manual_seed(seed)
                for shape in ((1, 1), (5, 3), (3, 65), (16, 128), (5, 1536), (1, 8192)):
                    with self.subTest(dtype=dtype, seed=seed, shape=shape):
                        self.check_output(torch.randn(shape, device="cuda", dtype=dtype))

    def test_negative_padding_equal_rows_and_large_score_gaps(self):
        for dtype in (torch.float16, torch.float32):
            scores = torch.tensor([[-1002., -1001., -1000.],
                                   [-1000., -1000., -1000.],
                                   [1., 1000., -1000.],
                                   [60000., 60000., 60000.]], device="cuda", dtype=dtype)
            actual = self.check_output(scores)
            # Three columns pad to four: padding must not turn these rows into NaNs.
            torch.testing.assert_close(actual[1], torch.full((3,), 1 / 3, device="cuda", dtype=dtype))
            torch.testing.assert_close(actual[2], torch.tensor([0., 1., 0.], device="cuda", dtype=dtype))

    def test_warp_variants_on_padding_and_wide_rows(self):
        torch.manual_seed(3)
        for dtype in (torch.float16, torch.float32):
            inputs = (torch.tensor([[-1002., -1001., -1000.]], device="cuda", dtype=dtype),
                      torch.randn(1, 8192, device="cuda", dtype=dtype))
            for scores in inputs:
                for warps in (1, 2, 4, 8):
                    with self.subTest(dtype=dtype, width=scores.shape[1], warps=warps):
                        self.check_output(scores, num_warps=warps)

    def test_rejects_unsupported_inputs_and_training(self):
        from fastinfer.kernels.triton_softmax import softmax

        for scores in (torch.ones(3, device="cuda"), torch.empty(0, 3, device="cuda"),
                       torch.ones(1, 8193, device="cuda"), torch.ones(2, 3)):
            with self.assertRaises(ValueError):
                softmax(scores)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            softmax(torch.ones(2, 6, device="cuda")[:, ::2])
        with self.assertRaises(TypeError):
            softmax(torch.ones(2, 3, device="cuda", dtype=torch.int64))
        with torch.enable_grad(), self.assertRaisesRegex(RuntimeError, "forward-only"):
            softmax(torch.ones(2, 3, device="cuda", requires_grad=True))
        with self.assertRaisesRegex(ValueError, "num_warps"):
            softmax(torch.ones(2, 3, device="cuda"), num_warps=3)


if __name__ == "__main__":
    unittest.main()
