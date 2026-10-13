"""Check normalization captures and hook cleanup without a GPU or model download."""

from pathlib import Path
import sys
import unittest
import torch

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
try:
    import qwen_rmsnorm_check as benchmark
finally:
    sys.path.pop(0)


class NormCaptureTest(unittest.TestCase):
    def test_each_module_captured_once(self):
        a, b = torch.nn.LayerNorm(2), torch.nn.LayerNorm(2)
        x = torch.tensor([[[3., 4.]]])
        with torch.inference_mode():
            output, captures = benchmark.capture_norm_calls({"a": a, "b": b}, lambda: b(a(x)))
            self.assertEqual(set(captures), {"a", "b"})
            torch.testing.assert_close(captures["a"][0], x)
            torch.testing.assert_close(captures["b"][0], captures["a"][1])
            torch.testing.assert_close(output, captures["b"][1])
        self.assertEqual(len(a._forward_hooks), 0)
        self.assertEqual(len(b._forward_hooks), 0)

    def test_hooks_removed_on_decode_failure(self):
        a = torch.nn.LayerNorm(2)
        def fail():
            a(torch.ones(1, 1, 2))
            raise RuntimeError("decode failure")
        with self.assertRaisesRegex(RuntimeError, "decode failure"):
            benchmark.capture_norm_calls({"a": a}, fail)
        self.assertEqual(len(a._forward_hooks), 0)

    def test_repeated_module_call_rejected(self):
        a = torch.nn.LayerNorm(2)
        with self.assertRaisesRegex(RuntimeError, "one decode call"):
            benchmark.capture_norm_calls({"a": a}, lambda: a(a(torch.ones(1, 1, 2))))
        self.assertEqual(len(a._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
