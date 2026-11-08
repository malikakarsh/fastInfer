"""Check trace invariants using manually supplied cache tensors; no inference."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch


directory = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location("speculative_failure_trace", directory / "speculative_failure_trace.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class ManualCache:
    def __init__(self, ids=()):
        self.set(ids)

    def set(self, ids):
        values = torch.tensor(ids, dtype=torch.float32).reshape(1, 1, -1, 1)
        self.layers = [SimpleNamespace(keys=values.clone(), values=values.clone())]

    def get_seq_length(self):
        return self.layers[0].keys.shape[2]


class LaterTraceTest(unittest.TestCase):
    def prepared(self):
        recorder = diagnostic.Recorder([1, 2], expected=[3, 4, 5])
        cache = ManualCache()
        recorder("target_before", {"phase": "prefill", "cache": cache,
                                  "input_ids": torch.tensor([[1, 2]]), "history": torch.tensor([[1, 2]])})
        cache.set([1, 2])
        recorder("target_after", {"cache": cache})
        recorder("emit", {"index": 0, "token": 3, "row": 0, "logits": torch.zeros(10)})
        return recorder, cache

    def test_captures_actual_block_and_causal_mismatch_row(self):
        recorder, cache = self.prepared()
        recorder("target_before", {"phase": "verification", "cache": cache,
            "input_ids": torch.tensor([[3, 4, 9]]), "history": torch.tensor([[1, 2, 3]]), "proposals": [4, 9]})
        cache.set([1, 2, 3, 4, 9])
        recorder("target_after", {"cache": cache})
        recorder("emit", {"index": 1, "token": 4, "row": 0, "logits": torch.zeros(10)})
        with self.assertRaises(diagnostic.DivergenceFound):
            recorder("emit", {"index": 2, "token": 9, "row": 1, "logits": torch.zeros(10)})
        self.assertEqual(recorder.mismatch["generated_token_1_based"], 3)
        self.assertEqual(recorder.mismatch["current_call"]["input_ids"], [3, 4, 9])
        self.assertEqual(recorder.snapshot[0]["keys"].flatten().tolist(), [1., 2.])

    def test_rejects_wrong_cached_history_and_wrong_logit_row(self):
        recorder, cache = self.prepared()
        with self.assertRaisesRegex(RuntimeError, "history/row"):
            recorder("target_before", {"phase": "verification", "cache": cache,
                "input_ids": torch.tensor([[3, 4]]), "history": torch.tensor([[1, 8, 3]]), "proposals": [4]})
        recorder, cache = self.prepared()
        recorder("target_before", {"phase": "verification", "cache": cache,
            "input_ids": torch.tensor([[3, 4]]), "history": torch.tensor([[1, 2, 3]]), "proposals": [4]})
        with self.assertRaisesRegex(RuntimeError, "history/row"):
            recorder("emit", {"index": 1, "token": 4, "row": 1, "logits": torch.zeros(10)})

    def test_crop_detects_mutation_of_retained_values(self):
        recorder = diagnostic.Recorder([1, 2])
        recorder.lineage = {"target": [1, 2, 3, 4, 9], "draft": [1, 2, 3, 4, 9]}
        target, draft = ManualCache([1, 2, 3, 4, 9]), ManualCache([1, 2, 3, 4, 9])
        payload = {"target_cache": target, "draft_cache": draft,
                   "history": torch.tensor([[1, 2, 3, 4, 5]]), "target_length": 4, "draft_length": 4}
        recorder("crop_before", payload)
        target.set([1, 2, 3, 4])
        draft.set([1, 2, 3, 4])
        recorder("crop_after", payload)
        self.assertTrue(recorder.records[-1]["prefix_values_preserved"]["target"])
        recorder("crop_before", payload)
        target.set([1, 2, 8, 4])
        with self.assertRaisesRegex(RuntimeError, "retained cache values"):
            recorder("crop_after", payload)


if __name__ == "__main__":
    unittest.main()
