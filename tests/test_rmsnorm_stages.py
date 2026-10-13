"""Validate stage arithmetic, reporting, and the GPU diagnostic cross-check."""

from pathlib import Path
import sys
import unittest

import torch

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
try:
    from qwen_rmsnorm_stages import pytorch_stages, difference_report, compare_stages
finally:
    sys.path.pop(0)


class StageTest(unittest.TestCase):
    def test_known_arithmetic_and_round_before_weight(self):
        x = torch.tensor([[[3., 4.]]], dtype=torch.float16)
        weight = torch.tensor([1., 2.], dtype=torch.float16)
        stages = pytorch_stages(x, weight, eps=0)
        torch.testing.assert_close(stages["squared_values"], torch.tensor([[[9., 16.]]]))
        self.assertEqual(stages["mean_square"].item(), 12.5)
        expected = (x.float() / torch.sqrt(torch.tensor(12.5))).half() * weight
        torch.testing.assert_close(stages["weighted_output_fp16"], expected, rtol=0, atol=0)
        self.assertEqual(stages["normalized_fp32"].dtype, torch.float32)
        self.assertEqual(stages["normalized_fp16"].dtype, torch.float16)

    def test_reports_first_difference_and_example(self):
        reference = {"squares": torch.tensor([1., 4.]), "mean": torch.tensor([2.5]),
                     "output": torch.tensor([2., 3.])}
        actual = {"squares": reference["squares"], "mean": torch.tensor([2.6]),
                  "output": torch.tensor([2., 3.5])}
        result = compare_stages(actual, reference)
        self.assertEqual(result["first_different_stage"], "mean")
        row = result["stages"]["output"]
        self.assertEqual(row["different_elements"], 1)
        self.assertEqual(row["max_abs_difference"], .5)
        self.assertEqual(row["largest_difference_example"]["flat_index"], 1)
        self.assertIsNone(compare_stages(reference, reference)["first_different_stage"])

    def test_invalid_comparisons_rejected(self):
        with self.assertRaises(ValueError):
            difference_report(torch.ones(2), torch.ones(3))
        with self.assertRaises(RuntimeError):
            difference_report(torch.tensor([float("nan")]), torch.ones(1))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_diagnostic_matches_existing_kernel_and_saved_stages(self):
        from fastinfer.kernels.rmsnorm_diagnostic import triton_stages
        from fastinfer.kernels.triton_rmsnorm import rmsnorm

        with torch.inference_mode():
            for seed, width in ((7, 65), (8, 1536)):
                torch.manual_seed(seed)
                x = torch.randn(1, 1, width, device="cuda", dtype=torch.float16)
                weight = torch.randn(width, device="cuda", dtype=torch.float16)
                stages = triton_stages(x, weight, eps=1e-6)
                torch.testing.assert_close(stages["weighted_output_fp16"],
                                           rmsnorm(x, weight), rtol=0, atol=0)
                torch.testing.assert_close(stages["squared_values"], x.float().square(),
                                           rtol=0, atol=0)
                torch.testing.assert_close(stages["normalized_fp16"],
                                           stages["normalized_fp32"].half(), rtol=0, atol=0)
                torch.testing.assert_close(stages["weighted_output_fp16"],
                                           stages["normalized_fp16"] * weight, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
