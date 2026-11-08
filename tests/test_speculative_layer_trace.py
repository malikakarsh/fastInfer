"""Activation selection and metrics only; no model forward or GPU work."""

import importlib.util
from pathlib import Path
import sys
import unittest

import torch


directory = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(directory))
spec = importlib.util.spec_from_file_location("speculative_layer_trace", directory / "speculative_layer_trace.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class LayerTraceHelpersTest(unittest.TestCase):
    def test_fp64_conversion_moves_to_cpu_first(self):
        events = []

        class TensorStub:
            def detach(self):
                events.append("detach")
                return self

            def cpu(self):
                events.append("cpu")
                return self

            def double(self):
                if events != ["detach", "cpu"]:
                    raise RuntimeError("FP64 attempted before moving to CPU")
                events.append("double")
                return self

        tensor = TensorStub()
        self.assertIs(diagnostic.cpu_double(tensor), tensor)
        self.assertEqual(events, ["detach", "cpu", "double"])

    def test_align_two_single_token_calls_with_block_prefix(self):
        values = torch.arange(12).reshape(1, 4, 3).float()
        seq, block = diagnostic.Trace(2), diagnostic.Trace(2)
        seq.active = block.active = True
        seq.record("projection.output", values[:, :1])
        seq.record("projection.output", values[:, 1:2])
        block.record("projection.output", values)
        self.assertTrue(torch.equal(seq.joined("projection.output"), block.joined("projection.output")))
        self.assertEqual(len(seq.shapes["projection.output"]), 2)
        self.assertEqual(block.shapes["projection.output"][0]["shape"], [1, 4, 3])

    def test_inactive_trace_skips_prefill(self):
        trace = diagnostic.Trace(2)
        trace.record("projection.output", torch.ones(1, 32, 3))
        self.assertEqual(trace.values, {})

    def test_metrics_locate_token_difference_and_nonfinite(self):
        expected = torch.zeros(2, 3)
        actual = expected.clone()
        actual[1, 2] = .125
        comparison = diagnostic.difference(actual, expected)
        self.assertEqual(comparison["different_values_per_token"], [0, 1])
        self.assertEqual(comparison["max_abs"], .125)
        self.assertFalse(comparison["equal"])
        self.assertFalse(diagnostic.difference(torch.tensor([float("nan")]), torch.zeros(1))["finite"])


if __name__ == "__main__":
    unittest.main()
