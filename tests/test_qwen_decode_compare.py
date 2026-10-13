"""Ensure model-validation gates detect divergent tokens and invalid logits."""

from pathlib import Path
import sys
import unittest

import torch

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
try:
    import qwen_decode_compare as benchmark
finally:
    sys.path.pop(0)


class DecodeComparisonTest(unittest.TestCase):
    def test_logits_and_tokens_both_required(self):
        baseline = {"logits": torch.tensor([[[1., 2.]]]), "tokens": [[1, 2]]}
        candidate = {"logits": baseline["logits"].clone(), "tokens": [[1, 2]]}
        self.assertTrue(benchmark.compare_outputs(baseline, candidate)["passed"])
        candidate["tokens"] = [[1, 3]]
        self.assertFalse(benchmark.compare_outputs(baseline, candidate)["passed"])
        candidate["tokens"] = [[1, 2]]
        candidate["logits"] += 1
        self.assertFalse(benchmark.compare_outputs(baseline, candidate)["passed"])

    def test_diagnostics_locate_failing_step(self):
        baseline = {"logits": torch.zeros(2, 1, 2), "tokens": [[1, 2, 3]]}
        logits = torch.tensor([[[0., 0.]], [[.02, 0.]]])
        candidate = {"logits": logits, "tokens": [[1, 2, 3]]}
        report = benchmark.compare_outputs(baseline, candidate)
        self.assertEqual([row["mismatches"] for row in report["per_step"]], [0, 1])
        self.assertEqual(report["mismatch_fraction"], .25)
        self.assertAlmostEqual(report["per_step"][0]["probability_total_variation_max"], 0.)
        self.assertGreater(report["per_step"][1]["probability_total_variation_max"], 0.)

    def test_projection_diagnostics_use_identical_input(self):
        weight = torch.tensor([[1., 2.], [3., 4.]])
        x = torch.tensor([[[2., 3.]]])
        actual = torch.nn.functional.linear(x, weight)
        candidate = {"projections": [(x, actual)]}
        row = benchmark.projection_diagnostics(candidate, weight)[0]
        self.assertEqual(row["pytorch_gemv_different_elements"], 0)
        self.assertEqual(row["accuracy"]["gemv"]["fp32_mismatches"], 0)
        candidate["projections"] = [(x, actual + 2)]
        row = benchmark.projection_diagnostics(candidate, weight)[0]
        self.assertEqual(row["accuracy"]["gemv"]["fp32_mismatches"], 2)
        self.assertEqual(row["accuracy"]["pytorch"]["fp32_mismatches"], 0)

    def test_nonfinite_logits_rejected(self):
        baseline = {"logits": torch.ones(1, 1, 2), "tokens": [[1, 2]]}
        candidate = {"logits": torch.full((1, 1, 2), float("nan")), "tokens": [[1, 2]]}
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            benchmark.compare_outputs(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
