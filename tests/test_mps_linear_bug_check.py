"""Exact-reference/bound bookkeeping only; no models, MPS, or linear evaluation."""

from fractions import Fraction
import importlib.util
from pathlib import Path
import unittest

import torch


spec = importlib.util.spec_from_file_location(
    "mps_linear_bug_check", Path(__file__).resolve().parents[1] / "benchmarks/mps_linear_bug_check.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class ArithmeticReferenceTest(unittest.TestCase):
    def test_exact_dot_and_nearest_even_midpoints(self):
        self.assertEqual(suite.exact_dot([1., 2.], [3., 4.], 5.), Fraction(16))
        midpoint = Fraction(2049, 2048)
        self.assertEqual(suite.round_fraction(midpoint), 1.)
        self.assertEqual(suite.round_fraction(midpoint + Fraction(1, 2**40)), 1.0009765625)
        self.assertEqual(suite.round_fraction(Fraction(2051, 2048)), 1.001953125)
        self.assertEqual(suite.round_fraction(-midpoint), -1.)
        self.assertEqual(suite.round_fraction(Fraction(1, 2**25)), 0.)
        self.assertEqual(suite.round_fraction(Fraction(1, 2**25) + Fraction(1, 2**50)), 2**-24)

    def test_integer_reference_has_exact_small_partial_sums(self):
        x, w, bias, reference = suite.integer_case(4, 8, 5)
        expected = [[sum(int(x[row, k]) * int(w[column, k]) for k in range(8)) + int(bias[column])
                     for column in range(5)] for row in range(4)]
        self.assertTrue(torch.equal(reference, torch.tensor(expected, dtype=torch.float16)))
        self.assertLessEqual(int(x.abs().sum(-1).max()), 32)

    def test_conditional_bound_pass_and_failure(self):
        reference = torch.tensor([[11.]], dtype=torch.float64)
        bound = suite.fp32_model_bound(reference, reference, reference.abs(), torch.float16, False, 2)
        self.assertEqual(suite.check_bound(reference, reference, bound)["violating_values"], 0)
        self.assertEqual(suite.check_bound(torch.tensor([[12.]]), reference, bound)["violating_values"], 1)
        with self.assertRaises(ValueError):
            suite.gamma_n(4096, 2**-11)

    def test_exact_failure_does_not_hide_nonfinite_values(self):
        result = suite.exact_result(torch.tensor([float("nan")]), torch.zeros(1))
        self.assertFalse(result["passed"])
        self.assertEqual(result["examples"][0]["actual"], "nan")


if __name__ == "__main__":
    unittest.main()
