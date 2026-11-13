"""Diagnostic bookkeeping only: no model, quantized operation or device timing."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch


spec = importlib.util.spec_from_file_location(
    "mps_quantized_projection", Path(__file__).resolve().parents[1] / "benchmarks/mps_quantized_projection.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class ProjectionDiagnosticTest(unittest.TestCase):
    def test_checkpoint_reader_handles_shards_without_reading_other_layers(self):
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file({"layer.weight": torch.tensor([[1.]]), "other.weight": torch.tensor([[9.]])},
                      str(root / "part1.safetensors"))
            save_file({"layer.bias": torch.tensor([2.])}, str(root / "part2.safetensors"))
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                "layer.weight": "part1.safetensors", "layer.bias": "part2.safetensors",
                "other.weight": "part1.safetensors"}}))
            result = diagnostic.read_projection(root, "layer")
            self.assertEqual(set(result), {"weight", "bias"})
            self.assertEqual(result["weight"].item(), 1.)
            with self.assertRaisesRegex(ValueError, "not found"):
                diagnostic.read_projection(root, "missing")

    def test_trace_observes_fallback_and_restores_wrappers(self):
        backend = SimpleNamespace(_get_kernel=lambda: None,
                                  _dequantize_4bit_impl=lambda: "decoded")
        original_get, original_decode = backend._get_kernel, backend._dequantize_4bit_impl
        def operation():
            backend._get_kernel()
            return diagnostic.F.linear("input", backend._dequantize_4bit_impl())
        with patch.object(diagnostic.F, "linear", return_value="output") as linear:
            report, output = diagnostic.trace_dispatch(operation, backend, lambda: None)
            self.assertEqual(output, "output")
            self.assertEqual(report["path"], "weight_reconstruction_then_linear_observed")
            self.assertEqual(report["counts"]["weight_reconstruction"], 1)
            self.assertIs(diagnostic.F.linear, linear)
        self.assertIs(backend._get_kernel, original_get)
        self.assertIs(backend._dequantize_4bit_impl, original_decode)

    def test_trace_observes_hub_and_restores_on_error(self):
        backend = SimpleNamespace(_get_kernel=lambda: SimpleNamespace(gemv_4bit=lambda: "result"),
                                  _dequantize_4bit_impl=lambda: "decoded")
        original_get = backend._get_kernel
        report, output = diagnostic.trace_dispatch(
            lambda: backend._get_kernel().gemv_4bit(), backend, lambda: None)
        self.assertEqual(report["path"], "specialized_hub_gemv_observed")
        self.assertEqual(output, "result")
        def fail():
            raise RuntimeError("probe failed")
        with self.assertRaisesRegex(RuntimeError, "probe failed"):
            diagnostic.trace_dispatch(fail, backend, lambda: None)
        self.assertIs(backend._get_kernel, original_get)
        self.assertEqual(diagnostic.classify_path({}), "unresolved")

    def test_timer_rotates_and_normalizes_completed_calls(self):
        calls, synchronization = [], []
        operations = {name: lambda name=name: calls.append(name) for name in ("a", "b", "c")}
        clock = iter(range(100))
        with patch.object(diagnostic.time, "perf_counter", side_effect=lambda: next(clock)):
            result = diagnostic.measure_operations(operations, lambda: synchronization.append(True),
                                                     warmup=1, samples=3, ops_per_sample=2)
        self.assertEqual(result["orders"], [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]])
        self.assertEqual(len(calls), 3 + 3 * 3 * 2)
        self.assertEqual(len(synchronization), 1 + 2 * 3 * 3)
        self.assertEqual(result["median_ms_per_call"], {"a": 500., "b": 500., "c": 500.})

    def test_nonfinite_output_fails_control_and_ratios_have_right_direction(self):
        self.assertFalse(diagnostic.difference(torch.tensor([float("nan")]), torch.tensor([1.]))["within_tolerance"])
        ratios = diagnostic.comparison_ratios({"normal_4bit": 8., "reconstruct_each_call": 6.,
                                               "predecoded_fp16": 2., "original_fp16": 1.,
                                               "reconstruction_only": 3.})
        self.assertEqual(ratios["normal_4bit_over_original_fp16"], 8.)
        self.assertEqual(ratios["reconstruction_only_over_reconstruct_each"], .5)


if __name__ == "__main__":
    unittest.main()
