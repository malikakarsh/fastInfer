"""Compare only rotation: both paths receive the same Q/K and tables."""

import unittest

import torch

try:
    import triton
except ImportError:
    triton = None

from fastinfer.kernels.rope import reference_rope, rope_cos_sin


@unittest.skipUnless(triton is not None and torch.cuda.is_available(), "CUDA Triton required")
class TritonRopeTest(unittest.TestCase):
    def check_output(self, q, k, cos, sin):
        from fastinfer.kernels.triton_rope import triton_rope
        before = [x.clone() for x in (q, k, cos, sin)]
        with torch.inference_mode():
            expected = reference_rope(q, k, cos, sin)
            actual = triton_rope(q, k, cos, sin)
        for result, wanted in zip(actual, expected):
            torch.testing.assert_close(result, wanted, rtol=0, atol=0)
            self.assertTrue(torch.isfinite(result).all().item())
            self.assertEqual(result.shape, wanted.shape)
            self.assertEqual(result.dtype, wanted.dtype)
        for x, old in zip((q, k, cos, sin), before):
            torch.testing.assert_close(x, old, rtol=0, atol=0)
        return actual

    def test_known_quarter_turn_and_identity(self):
        q = torch.tensor([[[[5., 2., 1., 3.]]]], device="cuda")
        cos, sin = torch.zeros(1, 1, 4, device="cuda"), torch.ones(1, 1, 4, device="cuda")
        result, _ = self.check_output(q, 2 * q, cos, sin)
        torch.testing.assert_close(result, torch.tensor([[[[-1., -3., 5., 2.]]]], device="cuda"),
                                   rtol=0, atol=0)
        result, _ = self.check_output(q, q, sin, cos)
        torch.testing.assert_close(result, q, rtol=0, atol=0)

    def test_shapes_dtypes_heads_offsets_and_masked_tails(self):
        for dtype in (torch.float16, torch.float32):
            for seed in (41, 42):
                torch.manual_seed(seed)
                for batch, hq, hk, tokens, width in (
                    (1, 1, 1, 1, 2), (2, 3, 1, 5, 6), (1, 4, 2, 16, 64),
                    (2, 4, 1, 3, 66), (1, 12, 2, 1, 128), (1, 2, 1, 3, 256),
                ):
                    with self.subTest(dtype=dtype, seed=seed, shape=(batch, hq, hk, tokens, width)):
                        q = torch.randn(batch, hq, tokens, width, device="cuda", dtype=dtype)
                        k = torch.randn(batch, hk, tokens, width, device="cuda", dtype=dtype)
                        positions = torch.arange(batch * tokens, device="cuda").reshape(batch, tokens) + 17
                        cos, sin = rope_cos_sin(positions, width, theta=10000, dtype=dtype)
                        self.check_output(q, k, cos, sin)

    def test_rounding_and_per_feature_coefficients(self):
        # Deliberately asymmetric tables catch incorrect coefficient indexing.
        for dtype in (torch.float16, torch.float32):
            q = torch.tensor([[[[.33325, -.125, 1000., -2., .001, 3.]]]], device="cuda", dtype=dtype)
            k = -q
            cos = torch.tensor([[[.7, -.2, .1, -.3, .25, .9]]], device="cuda", dtype=dtype)
            sin = torch.tensor([[[.3, .6, -.4, .8, -.5, .1]]], device="cuda", dtype=dtype)
            self.check_output(q, k, cos, sin)

    def test_rejects_unsupported_inputs_and_gradients(self):
        from fastinfer.kernels.triton_rope import triton_rope
        q = torch.ones(1, 2, 3, 4, device="cuda")
        cos, sin = rope_cos_sin(torch.tensor([[0, 1, 2]], device="cuda"), 4)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            triton_rope(q.cpu(), q.cpu(), cos.cpu(), sin.cpu())
        # Same shape but noncontiguous memory.
        strided = torch.ones(1, 2, 3, 8, device="cuda")[..., ::2]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            triton_rope(strided, q, cos, sin)
        wide = torch.ones(1, 1, 1, 258, device="cuda")
        c, s = rope_cos_sin(torch.tensor([[0]], device="cuda"), 258)
        with self.assertRaisesRegex(ValueError, "256"):
            triton_rope(wide, wide, c, s)
        for index in range(4):
            inputs = [q, q, cos, sin]
            inputs[index] = inputs[index].clone().requires_grad_(True)
            with torch.enable_grad(), self.assertRaisesRegex(RuntimeError, "forward-only"):
                triton_rope(*inputs)


if __name__ == "__main__":
    unittest.main()
