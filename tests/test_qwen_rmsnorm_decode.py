"""Per-step normalization diagnostics and explicit exploratory timing gates."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import torch
from fastinfer.kernels.rmsnorm import reference_rmsnorm

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
try:
    import qwen_rmsnorm_decode_compare as benchmark
    import qwen_decode_compare as shared
finally:
    sys.path.pop(0)


class ToyNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        return reference_rmsnorm(x, self.weight, eps=self.variance_epsilon)


class ToyModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.norm = ToyNorm()
        self.model = SimpleNamespace(layers=[SimpleNamespace(mlp=SimpleNamespace(up_proj=torch.nn.Identity()))])
        self.fail = fail

    def forward(self, input_ids, attention_mask, past_key_values, **kwargs):
        x = input_ids.float().unsqueeze(-1).expand(-1, -1, 2).contiguous()
        self.norm(x)
        if self.fail and input_ids.shape[1] == 1:
            raise RuntimeError("decode failure")
        logits = torch.zeros((*input_ids.shape, 4))
        logits[..., 3] = 1
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class NormDecodeDiagnosticsTest(unittest.TestCase):
    def run_trial(self, model):
        encoded = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}
        event = Mock()
        event.elapsed_time.return_value = 1.
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(DynamicCache=object)}), \
                patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "Event", return_value=event):
            return shared.decode_trial(model, encoded, 3, custom=False, validate=True,
                                       replacement="rmsnorm", norms={"norm": model.norm},
                                       capture_normalization=True)

    def test_capture_excludes_prefill_and_cleans_up(self):
        model = ToyModel()
        result = self.run_trial(model)
        self.assertEqual(len(result["normalization"]["norm"]), 3)
        self.assertTrue(all(tuple(x.shape) == (1, 1, 2)
                            for x, y in result["normalization"]["norm"]))
        self.assertEqual(len(model.norm._forward_hooks), 0)
        diagnostics = benchmark.normalization_diagnostics(result, {"norm": model.norm})
        self.assertEqual(diagnostics["calls_checked"], 3)
        self.assertTrue(diagnostics["all_passed"])

    def test_capture_hooks_removed_after_decode_failure(self):
        model = ToyModel(fail=True)
        with self.assertRaisesRegex(RuntimeError, "decode failure"):
            self.run_trial(model)
        self.assertEqual(len(model.norm._forward_hooks), 0)

    def test_local_checks_detect_wrong_normalization(self):
        norm = ToyNorm()
        x = torch.tensor([[[3., 4.]]])
        candidate = {"normalization": {"norm": [(x, norm(x) + .1)]}}
        diagnostics = benchmark.normalization_diagnostics(candidate, {"norm": norm})
        self.assertFalse(diagnostics["all_passed"])

    def test_explicit_exploratory_mode_retains_strict_failure(self):
        case = {"accuracy": {"passed": False, "identical_greedy_tokens": True},
                "baseline_repeat_accuracy": {"passed": True},
                "normalization_diagnostics": {"all_passed": True, "calls_checked": 912}}
        def decision(**kwargs):
            return benchmark.timing_decision(case, expected_calls=912, **kwargs)
        self.assertEqual(decision(diagnose=False, exploratory=False), (False, "strict_logits_failed"))
        self.assertEqual(decision(diagnose=False, exploratory=True), (True, "exploratory_strict_logits_failed"))
        self.assertFalse(case["accuracy"]["passed"])
        case["normalization_diagnostics"]["all_passed"] = False
        self.assertFalse(decision(diagnose=False, exploratory=True)[0])
        case["accuracy"]["passed"] = True
        self.assertFalse(decision(diagnose=False, exploratory=True)[0])
        self.assertFalse(decision(diagnose=True, exploratory=True)[0])
        case["accuracy"]["passed"] = False
        case["normalization_diagnostics"] = {"all_passed": True, "calls_checked": 911}
        self.assertFalse(decision(diagnose=False, exploratory=True)[0])
        case["normalization_diagnostics"]["calls_checked"] = 912
        case["accuracy"]["identical_greedy_tokens"] = False
        self.assertFalse(decision(diagnose=False, exploratory=True)[0])
        case["accuracy"]["identical_greedy_tokens"] = True
        case["baseline_repeat_accuracy"]["passed"] = False
        self.assertFalse(decision(diagnose=False, exploratory=True)[0])


if __name__ == "__main__":
    unittest.main()
