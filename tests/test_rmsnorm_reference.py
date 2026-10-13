"""Numerical examples and dtype checks for the readable RMSNorm reference."""

import unittest

import torch
from fastinfer.kernels.rmsnorm import reference_rmsnorm


class RmsNormReferenceTest(unittest.TestCase):
    def test_known_vector_with_learned_weights(self):
        x = torch.tensor([3., 4.])
        weight = torch.tensor([1., 2.])
        expected = torch.tensor([3 / (12.5 ** .5), 8 / (12.5 ** .5)])
        torch.testing.assert_close(reference_rmsnorm(x, weight, eps=0), expected)

    def test_each_token_has_its_own_scale(self):
        x = torch.tensor([[[3., 4.], [6., 8.]]])
        weight = torch.ones(2)
        result = reference_rmsnorm(x, weight, eps=0)
        torch.testing.assert_close(result[:, 0], result[:, 1])
        self.assertEqual(result.shape, x.shape)
        torch.testing.assert_close(result.square().mean(dim=-1), torch.ones(1, 2))

    def test_zero_vector_and_large_half_values_stay_finite(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                x = torch.tensor([[0., 0.], [60000., -60000.]], dtype=dtype)
                result = reference_rmsnorm(x, torch.ones(2, dtype=dtype))
                self.assertEqual(result.dtype, dtype)
                self.assertTrue(torch.isfinite(result).all().item())
                torch.testing.assert_close(result[0], torch.zeros(2, dtype=dtype))
                torch.testing.assert_close(result[1], torch.tensor([1., -1.], dtype=dtype))

    def test_rejects_wrong_weight_length(self):
        with self.assertRaisesRegex(ValueError, "one value per"):
            reference_rmsnorm(torch.ones(2, 3), torch.ones(2))


if __name__ == "__main__":
    unittest.main()
