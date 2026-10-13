"""Check profiling annotations preserve execution and clean up on failure."""

from pathlib import Path
import sys
import unittest

import torch

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
try:
    import qwen_decode_profile as benchmark
finally:
    sys.path.pop(0)


class ToyModel(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.lm_head = torch.nn.Linear(2, 3)
        self.fail = fail

    def forward(self, x):
        result = self.lm_head(x)
        if self.fail:
            raise RuntimeError("model failure")
        return result


class DecodeProfileTest(unittest.TestCase):
    def test_annotations_keep_output_and_count_calls(self):
        model = ToyModel()
        x = torch.ones(1, 2)
        expected = model(x)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            with benchmark.annotate_components(model):
                for _ in range(3):
                    actual = model(x)
                    torch.testing.assert_close(actual, expected)
        region = next(event for event in prof.key_averages()
                      if event.key == "component/vocabulary_projection")
        row = benchmark.event_summary(region)
        self.assertEqual(row["calls"], 3)
        self.assertGreater(row["total_cpu_us"], 0)
        self.assertEqual(row["total_device_us"], 0)
        self.assertEqual(len(model.lm_head._forward_hooks), 0)
        self.assertEqual(len(model.lm_head._forward_pre_hooks), 0)

    def test_hooks_removed_on_model_failure(self):
        model = ToyModel(fail=True)
        with self.assertRaisesRegex(RuntimeError, "model failure"):
            with benchmark.annotate_components(model):
                model(torch.ones(1, 2))
        self.assertEqual(len(model.lm_head._forward_hooks), 0)
        self.assertEqual(len(model.lm_head._forward_pre_hooks), 0)

    def test_region_closed_if_annotated_module_itself_fails(self):
        model = ToyModel()
        with self.assertRaises(RuntimeError):
            with benchmark.annotate_components(model):
                model(torch.ones(1, 7))
        self.assertEqual(len(model.lm_head._forward_hooks), 0)
        self.assertEqual(len(model.lm_head._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
