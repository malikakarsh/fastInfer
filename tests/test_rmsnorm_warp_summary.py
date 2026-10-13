"""Check paired savings report winner direction and require complete pairs."""

import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "benchmarks/rmsnorm_warp_check.py"
spec = importlib.util.spec_from_file_location("rmsnorm_warp_check", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WarpSummaryTest(unittest.TestCase):
    def test_positive_means_eight_faster_and_ties_are_separate(self):
        result = module.paired_four_eight({"warps4": [4., 5., 6., 7.],
                                           "warps8": [3., 6., 6., 5.]})
        self.assertEqual(result["saving_us_samples"], [1., -1., 0., 2.])
        self.assertEqual(result["median_saving_us"], .5)
        self.assertEqual(result["eight_faster_rounds"], 2)
        self.assertEqual(result["tied_rounds"], 1)
        self.assertEqual(result["rounds"], 4)

    def test_incomplete_or_empty_pairs_rejected(self):
        for sample in ({"warps4": [], "warps8": []},
                       {"warps4": [1., 2.], "warps8": [1.]}):
            with self.assertRaises(ValueError):
                module.paired_four_eight(sample)


if __name__ == "__main__":
    unittest.main()
