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

    def test_nonfinite_logits_rejected(self):
        baseline = {"logits": torch.ones(1, 1, 2), "tokens": [[1, 2]]}
        candidate = {"logits": torch.full((1, 1, 2), float("nan")), "tokens": [[1, 2]]}
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            benchmark.compare_outputs(baseline, candidate)


if __name__ == "__main__":
    unittest.main()
