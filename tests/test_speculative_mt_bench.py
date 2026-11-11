"""Verify category coverage and exclusion of incorrect outputs from timing summaries."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch


spec = importlib.util.spec_from_file_location(
    "speculative_mt_bench", Path(__file__).resolve().parents[1] / "benchmarks/speculative_mt_bench.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class MtBenchTest(unittest.TestCase):
    def test_first_turn_and_balanced_pilot(self):
        rows = [{"question_id": index, "category": category,
                 "prompt": [f"first-{index}", "followup"]}
                for index, category in enumerate(("writing", "writing", "math", "math"))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            selected = benchmark.load_prompts(path, limit=2)
            self.assertEqual([row["category"] for row in selected], ["writing", "math"])
            self.assertEqual([row["prompt"] for row in selected], ["first-0", "first-2"])
            path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]))
            with self.assertRaises(ValueError):
                benchmark.load_prompts(path)

    def test_quantized_checkpoint_requires_explicit_mode(self):
        config = SimpleNamespace(quantization_config={"quant_method": "bitsandbytes", "load_in_4bit": True})
        with patch.object(benchmark.AutoConfig, "from_pretrained", return_value=config), \
             patch.object(benchmark.AutoModelForCausalLM, "from_pretrained") as loader:
            with self.assertRaisesRegex(ValueError, "target-quantization"):
                benchmark.load_target("local", torch.device("mps"), torch.float16, "none")
            loader.assert_not_called()

    def test_quantized_loading_is_packed_explicit_and_serializable(self):
        # Loader mocks only: no model, device operation or inference is executed.
        raw = {"quant_method": "bitsandbytes", "load_in_4bit": True,
               "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_quant_type": "nf4"}
        config = SimpleNamespace(quantization_config=raw)
        class FakeLinear4bit:
            compute_dtype = torch.float16
            weight = SimpleNamespace(quant_state=object())
        layer = FakeLinear4bit()
        model = SimpleNamespace(is_loaded_in_4bit=True, modules=lambda: [layer],
                                parameters=lambda: [SimpleNamespace(device=torch.device("mps"))])
        model.eval = lambda: model
        bnb = SimpleNamespace(supported_torch_devices={"mps"}, nn=SimpleNamespace(Linear4bit=FakeLinear4bit))
        with patch.dict("sys.modules", {"bitsandbytes": bnb, "accelerate": SimpleNamespace()}), \
             patch.object(benchmark.AutoConfig, "from_pretrained", return_value=config), \
             patch.object(benchmark.AutoModelForCausalLM, "from_pretrained", return_value=model) as loader:
            loaded, metadata = benchmark.load_target("local", torch.device("mps"), torch.float16, "bnb-4bit")
            self.assertIs(loaded, model)
            self.assertEqual(loader.call_args.kwargs["device_map"], {"": "mps"})
            self.assertEqual(config.quantization_config["bnb_4bit_compute_dtype"], "float16")
            self.assertEqual(raw["bnb_4bit_compute_dtype"], "bfloat16")
            self.assertEqual(metadata["linear4bit_layers"], 1)
            json.dumps(metadata)
            bnb.supported_torch_devices = {"cuda"}
            loader.reset_mock()
            with self.assertRaisesRegex(ValueError, "does not advertise mps"):
                benchmark.load_target("local", torch.device("mps"), torch.float16, "bnb-4bit")
            loader.assert_not_called()

    def test_summary_during_partial_run_and_failed_case(self):
        good = {"id": 1, "correctness_passed": True, "samples": [
            {"baseline": {"total_s": 4.}, "speculative": {"total_s": 2.}}]}
        bad = {"id": 2, "correctness_passed": False, "samples": [
            {"baseline": {"total_s": 100.}, "speculative": {"total_s": .01}}]}
        summary = benchmark.summarize([good, bad])
        self.assertEqual(summary["aggregate_speedup"], 2.)
        self.assertEqual(summary["timed_cases"], 1)
        self.assertEqual(summary["failed_cases"], [2])


if __name__ == "__main__":
    unittest.main()
