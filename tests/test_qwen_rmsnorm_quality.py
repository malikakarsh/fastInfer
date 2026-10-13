"""EOS stopping, shared-history comparison and normalization dispatch behavior."""

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import json

import torch
from fastinfer.decode_rmsnorm import use_decode_rmsnorm
from fastinfer.kernels.rmsnorm import reference_rmsnorm

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
try:
    import qwen_rmsnorm_quality as benchmark
finally:
    sys.path.pop(0)


class ToyNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.variance_epsilon = 1e-6

    def forward(self, x):
        return reference_rmsnorm(x, self.weight, eps=self.variance_epsilon)


class Cache:
    def __init__(self):
        self.prefilled = False
        self.decoded = []


class ToyModel(torch.nn.Module):
    def __init__(self, first=1, predict_after_one=2):
        super().__init__()
        self.norm = ToyNorm().eval()
        self.first = first
        self.predict_after_one = predict_after_one
        self.last_cache = None

    def forward(self, input_ids, attention_mask, past_key_values, **kwargs):
        cache = past_key_values
        self.last_cache = cache
        self.norm(input_ids.float().unsqueeze(-1).expand(-1, -1, 2).contiguous())
        if not cache.prefilled:
            cache.prefilled = True
            chosen = self.first
        else:
            token = input_ids.item()
            cache.decoded.append(token)
            chosen = self.predict_after_one if token == 1 else 3
        logits = torch.zeros((*input_ids.shape, 4))
        logits[..., chosen] = 1.
        return SimpleNamespace(logits=logits, past_key_values=cache)


class QualityCheckTest(unittest.TestCase):
    def rollout(self, model, *, limit=64, teacher=None):
        encoded = {"input_ids": torch.tensor([[1, 2]]),
                   "attention_mask": torch.ones(1, 2, dtype=torch.long)}
        def context(norms):
            return use_decode_rmsnorm(norms, kernel=reference_rmsnorm)
        with patch.dict(sys.modules, {"transformers": SimpleNamespace(DynamicCache=Cache)}), \
                patch.object(benchmark, "use_decode_rmsnorm", context), \
                patch("fastinfer.decode_rmsnorm.eligible_norm_input", return_value=True):
            return benchmark.rollout(model, encoded, {"norm": model.norm}, limit, {3},
                                     custom=True, teacher_tokens=teacher)

    def test_eos_stops_and_prefill_does_not_count_as_custom_decode(self):
        model = ToyModel()
        result = self.rollout(model)
        self.assertEqual(result["tokens"], [1, 2, 3])
        self.assertTrue(result["terminated_on_eos"])
        self.assertEqual(result["decode_calls"], 2)
        self.assertEqual(result["dispatch"]["rmsnorm_calls"], 2)
        self.assertEqual(model.last_cache.decoded, [1, 2])
        self.assertNotIn("forward", model.norm.__dict__)

    def test_eos_on_first_token_needs_no_custom_decode(self):
        result = self.rollout(ToyModel(first=3))
        self.assertEqual(result["tokens"], [3])
        self.assertEqual(result["dispatch"]["rmsnorm_calls"], 0)

    def test_token_cap_stops_without_an_extra_forward(self):
        model = ToyModel()
        result = self.rollout(model, limit=2)
        self.assertEqual(result["tokens"], [1, 2])
        self.assertFalse(result["terminated_on_eos"])
        self.assertEqual(result["decode_calls"], 1)

    def test_shared_history_feeds_baseline_tokens_despite_different_predictions(self):
        model = ToyModel(predict_after_one=0)
        result = self.rollout(model, teacher=[1, 2, 3])
        self.assertEqual(result["tokens"], [1, 2, 3])
        self.assertEqual(result["predictions"], [1, 0, 3])
        self.assertEqual(model.last_cache.decoded, [1, 2])

    def test_divergence_positions_and_different_lengths(self):
        self.assertIsNone(benchmark.first_divergence([1, 2], [1, 2]))
        self.assertEqual(benchmark.first_divergence([1, 2], [1, 3]), 2)
        self.assertEqual(benchmark.first_divergence([1, 2], [1, 2, 3]), 3)

    def test_probability_comparison_requires_shared_history(self):
        a = {"tokens": [0], "scores": torch.tensor([[2., 1., 0.]])}
        b = {"tokens": [1], "scores": a["scores"]}
        with self.assertRaises(ValueError):
            benchmark.probability_comparison(a, b)
        b["tokens"] = [0]
        b["scores"] = a["scores"] + 1
        metrics = benchmark.probability_comparison(a, b)
        self.assertFalse(metrics["strict_logits_passed"])
        self.assertAlmostEqual(metrics["worst_probability_total_variation"], 0., places=6)

    def test_archive_keeps_available_raw_files_and_reports_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "results/raw"
            raw.mkdir(parents=True)
            destination = raw / "quality.json"
            destination.write_text('{}')
            (raw / "qwen_rmsnorm_decode_20261013T065623423837Z.json").write_text('{}')
            with patch.object(benchmark, "ROOT", root):
                archive, missing = benchmark.archive_evidence(destination)
            self.assertEqual(len(missing), 1)
            with zipfile.ZipFile(archive) as bundle:
                self.assertIn("quality.json", bundle.namelist())
                self.assertIn("qwen_rmsnorm_decode_20261013T065623423837Z.json", bundle.namelist())
                self.assertEqual(len(json.loads(bundle.read("archive_manifest.json"))["missing_timing_files"]), 1)


if __name__ == "__main__":
    unittest.main()
