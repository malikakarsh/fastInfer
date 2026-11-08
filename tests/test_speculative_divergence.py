"""Report reconstruction and logit summaries only; no model inference."""

import importlib.util
from pathlib import Path
import unittest

import torch


spec = importlib.util.spec_from_file_location(
    "speculative_divergence", Path(__file__).resolve().parents[1] / "benchmarks/speculative_divergence.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class Tokenizer:
    def decode(self, ids):
        return str(ids)


class DivergenceReportTest(unittest.TestCase):
    def case(self):
        return {"validation": {
            "baseline": {"generated": [2, 3, 4, 5]},
            "speculative": {"generated": [2, 3, 6, 7], "proposed": 3, "accepted": 3}}}

    def test_recover_first_block_without_inventing_rejected_proposals(self):
        index, baseline, candidate, block = diagnostic.recover_first_block(self.case(), 3)
        self.assertEqual(index, 2)
        self.assertEqual(baseline[:index], candidate[:index])
        self.assertEqual(block, [2, 3, 6, 7])
        case = self.case()
        case["validation"]["speculative"]["accepted"] = 2
        with self.assertRaises(ValueError):
            diagnostic.recover_first_block(case, 3)
        with self.assertRaises(ValueError):
            diagnostic.recover_first_block(self.case(), 1)

    def test_rank_flip_and_score_margin(self):
        baseline = torch.tensor([0., 3., 2.], dtype=torch.bfloat16)
        candidate = torch.tensor([0., 2., 3.], dtype=torch.bfloat16)
        description, values = diagnostic.describe_logits(candidate, Tokenizer(), (1, 2))
        self.assertEqual(description["argmax_id"], 2)
        self.assertEqual(description["top_two_gap"], 1.)
        self.assertEqual(description["baseline_minus_speculative_score"], -1.)
        self.assertFalse(diagnostic.compare_logits(values, baseline.float())["argmax_matches"])
        self.assertEqual(diagnostic.compare_logits(values, values)["different_logit_values"], 0)

    def test_winner_uses_argmax_tie_policy(self):
        values = torch.ones(32)
        description, _ = diagnostic.describe_logits(values, Tokenizer(), (0, 1))
        self.assertEqual(description["argmax_id"], 0)
        self.assertEqual(description["top_two_gap"], 0.)


if __name__ == "__main__":
    unittest.main()
