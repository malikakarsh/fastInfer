"""Small CPU checks for the PyTorch matmul baseline."""

import unittest

import torch

from fastinfer.kernels.matmul import reference_matmul


class ReferenceMatmulTest(unittest.TestCase):
    def test_known_nonsquare_result(self) -> None:
        a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        expected = torch.tensor([[4.0, 5.0], [10.0, 11.0]])

        torch.testing.assert_close(reference_matmul(a, b), expected, rtol=0, atol=0)

    def test_rejects_mismatched_inner_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            reference_matmul(torch.empty(2, 3), torch.empty(4, 2))


if __name__ == "__main__":
    unittest.main()
