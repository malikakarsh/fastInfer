"""Paired timing interpretation must not be inferred from separate medians."""

from pathlib import Path
import sys
import unittest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
try:
    from softmax_compare import native_triton_comparison
finally:
    sys.path.pop(0)


class SoftmaxBenchmarkTest(unittest.TestCase):
    def test_separate_medians_do_not_replace_paired_differences(self):
        result = native_triton_comparison([1., 100., 101.], [2., 3., 102.])
        self.assertGreater(result["ratio_of_separate_medians"], 1)
        self.assertEqual(result["paired_median_saving_us"], -1)
        self.assertEqual(result["triton_faster_rounds"], 1)

    def test_ties_and_invalid_samples(self):
        result = native_triton_comparison([2., 3.], [2., 3.])
        self.assertEqual(result["tied_rounds"], 2)
        self.assertEqual(result["triton_faster_rounds"], 0)
        for native, triton in (([], []), ([1.], [1., 2.]), ([0.], [1.])):
            with self.assertRaises(ValueError):
                native_triton_comparison(native, triton)


if __name__ == "__main__":
    unittest.main()
