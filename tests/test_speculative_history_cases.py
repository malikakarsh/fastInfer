"""Case selection and replay controls with manual tensors/mocks; no inference."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


directory = Path(__file__).resolve().parents[1] / "benchmarks"
spec = importlib.util.spec_from_file_location("speculative_history_cases", directory / "speculative_history_cases.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def saved_case(identifier, length, mismatch=None, category="math"):
    a = [1] * length
    b = list(a)
    if mismatch is not None:
        b[mismatch] = 2
    return {"id": identifier, "category": category, "input_tokens": 10,
            "baseline_repeat_matches": True, "correctness_passed": mismatch is None,
            "validation": {"baseline": {"generated": a}, "speculative": {"generated": b}}}


class HistoryCasesTest(unittest.TestCase):
    def test_plan_spans_early_to_late_and_has_distinct_passing_controls(self):
        failures = [saved_case(i + 10, 30, mismatch=point) for i, point in enumerate([1, 3, 8, 14, 24])]
        passes = [saved_case(i + 100, 30) for i in range(4)]
        plan = diagnostic.make_plan({"cases": failures + passes}, 3, 3)
        entries = plan["entries"]
        self.assertEqual([e["decision_token_1_based"] for e in entries[:3]], [2, 9, 25])
        self.assertEqual(len(set(e["id"] for e in entries)), 6)
        self.assertTrue(all(e["group"] == "saved_pass_control" for e in entries[3:]))
        self.assertTrue(all(e["same_decision_position"] for e in entries[3:]))
        self.assertEqual([e["matched_failure_id"] for e in entries[3:]], [10, 12, 14])

    def test_length_only_and_prefill_failures_not_misclassified(self):
        length_only = saved_case(1, 4)
        length_only["correctness_passed"] = False
        length_only["validation"]["speculative"]["generated"] = [1, 1]
        prefill = saved_case(2, 4, mismatch=0)
        unstable = saved_case(3, 4, mismatch=1)
        unstable["baseline_repeat_matches"] = False
        plan = diagnostic.make_plan({"cases": [length_only, prefill, unstable, saved_case(4, 8, 3)]}, 3, 0)
        self.assertEqual([e["id"] for e in plan["entries"]], [4])
        self.assertEqual(len(plan["excluded"]), 3)

    def test_winner_patterns_do_not_force_mixed_effects_into_one_cause(self):
        def pattern(a, b, c, d):
            return diagnostic.winner_pattern(dict(zip(
                ["ordinary_serial", "ordinary_block", "speculative_serial", "speculative_block"], [a, b, c, d])))
        self.assertEqual(pattern(1, 2, 1, 2), "current_execution_winner_effect_only")
        self.assertEqual(pattern(1, 1, 2, 2), "cache_history_winner_effect_only")
        self.assertEqual(pattern(1, 1, 1, 2), "both_factors_or_interaction")
        self.assertEqual(pattern(1, 1, 1, 1), "no_winner_effect_observed")

    def test_passing_decision_can_be_captured_without_inventing_mismatch(self):
        r = diagnostic.DecisionRecorder([1, 2], [3, 4], decision_index=1)
        class Cache:
            def __init__(self, ids):
                value = torch.tensor(ids, dtype=torch.float32).reshape(1, 1, -1, 1)
                self.layers = [SimpleNamespace(keys=value, values=value.clone())]
            def get_seq_length(self):
                return self.layers[0].keys.shape[2]
        r("target_before", {"phase": "prefill", "cache": Cache([]),
                           "input_ids": torch.tensor([[1, 2]]), "history": torch.tensor([[1, 2]])})
        r("target_after", {"cache": Cache([1, 2])})
        r("emit", {"index": 0, "token": 3, "row": 0, "logits": torch.tensor([0., 0., 0., 2., 1.])})
        r("target_before", {"phase": "verification", "cache": Cache([1, 2]),
            "input_ids": torch.tensor([[3, 4]]), "history": torch.tensor([[1, 2, 3]]), "proposals": [4]})
        r("target_after", {"cache": Cache([1, 2, 3, 4])})
        with self.assertRaises(diagnostic.DivergenceFound):
            r("emit", {"index": 1, "token": 4, "row": 0, "logits": torch.tensor([0., 0., 0., 1., 2.])})
        self.assertEqual(r.mismatch["baseline_token"], r.mismatch["speculative_token"])
        self.assertEqual(r.mismatch["generated_token_1_based"], 2)

    def test_manual_cache_distance_is_separate_from_predictor_features(self):
        a = [{"keys": torch.tensor([1., 2.]), "values": torch.tensor([3., 4.])}]
        b = [{"keys": torch.tensor([1., 2.5]), "values": torch.tensor([3., 4.])}]
        result = diagnostic.cache_distance(a, b)[0]
        self.assertEqual(result["keys"]["different_values"], 1)
        self.assertEqual(result["keys"]["max_abs"], .5)
        self.assertEqual(result["values"]["different_values"], 0)

    def test_four_way_classifies_only_after_reference_reproduction(self):
        reference = torch.tensor([0., 0., 2., 1.])
        actual = torch.tensor([0., 0., 1., 2.])
        model = SimpleNamespace(config=None, parameters=lambda: iter([torch.tensor(0.)]))
        recorder = SimpleNamespace(prompt=[9, 8], generated=[5, 6, 3], snapshot={"origin": "speculative"},
            mismatch={"baseline_token": 2, "speculative_token": 3, "row": 1, "actual_vector": actual,
                "current_call": {"phase": "verification", "prefix_token_ids": [9, 8], "input_ids": [5, 6]}})
        def forward(model, tokens, cache, keep=0):
            vector = actual if cache.origin == "speculative" and tokens.shape[1] == 2 else reference
            return torch.stack([vector] * tokens.shape[1]).unsqueeze(0)
        tokenizer = SimpleNamespace(decode=lambda ids: str(ids))
        with patch.object(diagnostic, "DynamicCache", side_effect=lambda **kwargs: SimpleNamespace(origin="ordinary")), \
             patch.object(diagnostic, "_forward", side_effect=forward), \
             patch.object(diagnostic, "pack_cache", side_effect=lambda cache: {"origin": cache.origin}), \
             patch.object(diagnostic, "restore_cache", side_effect=lambda model, snap, device: SimpleNamespace(**snap)), \
             patch.object(diagnostic, "cache_equal", return_value=True), \
             patch.object(diagnostic, "cache_distance", return_value=[]), \
             patch.object(diagnostic, "structural_checks", return_value={"lineage_passed": True}), \
             patch("builtins.print"):
            result = diagnostic.four_way(model, tokenizer, recorder, reference, repeats=2)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["winner_pattern"], "both_factors_or_interaction")
            result = diagnostic.four_way(model, tokenizer, recorder, reference + 1, repeats=2)
            self.assertEqual(result["status"], "controls_failed")
            self.assertEqual(result["winner_pattern"], "unclassified_controls_failed")


if __name__ == "__main__":
    unittest.main()
