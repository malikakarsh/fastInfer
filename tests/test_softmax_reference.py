"""Small numerical checks before implementing GPU softmax."""

import math
import unittest

import torch
from fastinfer.kernels.softmax import reference_softmax


class SoftmaxReferenceTest(unittest.TestCase):
    def test_known_probabilities(self):
        scores = torch.tensor([0., math.log(2), math.log(3)])
        expected = torch.tensor([1 / 6, 2 / 6, 3 / 6])
        torch.testing.assert_close(reference_softmax(scores), expected)

    def test_rows_are_independent_and_constant_shifts_preserve_probabilities(self):
        scores = torch.tensor([[1., 2., 3.], [2., 2., 2.]])
        result = reference_softmax(scores)
        shifted = scores + torch.tensor([[1000.], [-1000.]])
        torch.testing.assert_close(reference_softmax(shifted), result, rtol=0, atol=0)
        torch.testing.assert_close(result[1], torch.full((3,), 1 / 3))
        torch.testing.assert_close(result.sum(dim=-1), torch.ones(2))

    def test_extreme_finite_scores_and_native_pytorch_agreement(self):
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                scores = torch.tensor([[[1., 1000., -1000.],
                                        [60000., 60000., 60000.]]], dtype=dtype)
                result = reference_softmax(scores)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.shape, scores.shape)
                self.assertTrue(torch.isfinite(result).all().item())
                self.assertTrue((result >= 0).all().item())
                torch.testing.assert_close(result[0, 0], torch.tensor([0., 1., 0.], dtype=dtype))
                torch.testing.assert_close(result, torch.softmax(scores, dim=-1))
                torch.testing.assert_close(result.float().sum(dim=-1), torch.ones(1, 2),
                                           rtol=0, atol=5e-4)

    def test_rejects_empty_scalar_and_integer_inputs(self):
        for scores in (torch.empty(0, 3), torch.empty(2, 0), torch.tensor(1.)):
            with self.assertRaises(ValueError):
                reference_softmax(scores)
        with self.assertRaises(TypeError):
            reference_softmax(torch.tensor([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
