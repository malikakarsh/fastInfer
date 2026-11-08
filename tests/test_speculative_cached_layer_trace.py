"""Replay validation and operation selection; manual tensors, no inference."""

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


directory = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location("speculative_cached_layer_trace", directory / "speculative_cached_layer_trace.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class CachedLayerHelpersTest(unittest.TestCase):
    def artifact(self):
        value = torch.zeros(1, 2, 3, 4, dtype=torch.float16)
        return {"verification_input_ids": [4, 5, 6, 7], "row": 3,
                "prefix_token_ids": [1, 2, 3], "cache": [{
                    "keys": value, "values": value.clone(),
                    "keys_stride": list(value.stride()), "values_stride": list(value.stride())}]}

    def test_accepts_valid_same_state_artifact(self):
        self.assertEqual(diagnostic.validate_artifact(self.artifact()), torch.float16)

    def test_rejects_wrong_row_or_cache_prefix(self):
        artifact = self.artifact()
        artifact["row"] = 4
        with self.assertRaises(ValueError):
            diagnostic.validate_artifact(artifact)
        artifact = self.artifact()
        artifact["prefix_token_ids"].append(9)
        with self.assertRaises(ValueError):
            diagnostic.validate_artifact(artifact)

    def test_selects_equal_input_independent_difference(self):
        seq, block = diagnostic.Trace(2), diagnostic.Trace(2)
        seq.active = block.active = True
        inputs = torch.zeros(1, 4, 3)
        outputs = inputs.clone()
        outputs[0, 1, 0] = .125
        for name in ("equal_input_op", "already_different_input_op"):
            for index in range(2):
                seq.record(name + ".input", inputs[:, index:index + 1])
                seq.record(name + ".output", inputs[:, index:index + 1])
            block.record(name + ".input", inputs if name == "equal_input_op" else outputs)
            block.record(name + ".output", outputs)
            block.inputs[name] = [inputs]
        self.assertEqual(diagnostic.candidate_modules(seq, block), ["equal_input_op"])


if __name__ == "__main__":
    unittest.main()
