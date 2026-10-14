"""Element pairing, signs and intermediate rounding in the PyTorch reference."""

import math
import unittest

import torch
from fastinfer.kernels.swiglu import reference_swiglu


class SwigluReferenceTest(unittest.TestCase):
    def test_known_values_and_element_pairing(self):
        gate = torch.tensor([2., 0., -2.])
        up = torch.tensor([3., 100., 4.])
        expected = torch.tensor([6 / (1 + math.exp(-2)), 0., -8 / (1 + math.exp(2))])
        torch.testing.assert_close(reference_swiglu(gate, up), expected)

    def test_shapes_dtypes_and_staged_double_oracle(self):
        torch.manual_seed(0)
        for dtype in (torch.float16, torch.float32):
            for shape in ((7,), (3, 65), (1, 1, 8960)):
                gate = torch.randn(shape, dtype=dtype)
                up = torch.randn(shape, dtype=dtype)
                activated = (gate.double() * torch.sigmoid(gate.double())).to(dtype)
                expected = (activated.double() * up.double()).to(dtype)
                actual = reference_swiglu(gate, up)
                self.assertEqual(actual.shape, gate.shape)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, expected,
                                           rtol=.002 if dtype == torch.float16 else 1e-5,
                                           atol=2e-4 if dtype == torch.float16 else 2e-6)

    def test_large_gaps_zero_and_negative_up(self):
        gate = torch.tensor([-1000., 0., 1000., 2.])
        up = torch.tensor([2., 5., -2., 0.])
        torch.testing.assert_close(reference_swiglu(gate, up),
                                   torch.tensor([0., 0., -2000., 0.]))

    def test_rejects_broadcasting_empty_and_mismatched_dtype(self):
        for gate, up in ((torch.ones(2, 3), torch.ones(3)),
                         (torch.empty(0), torch.empty(0)),
                         (torch.tensor(1.), torch.tensor(2.)),
                         (torch.ones(3), torch.ones(3, dtype=torch.float16))):
            with self.assertRaises(ValueError):
                reference_swiglu(gate, up)
        with self.assertRaises(TypeError):
            reference_swiglu(torch.ones(3, dtype=torch.int64), torch.ones(3, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
