"""Capture boundaries and cleanup without downloading Qwen."""

from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
try:
    from qwen_swiglu_check import capture_mlp_branches
finally:
    sys.path.pop(0)


class TinyMLP(torch.nn.Module):
    def __init__(self, fail=False):
        super().__init__()
        self.gate_proj = torch.nn.Linear(3, 4, bias=False)
        self.up_proj = torch.nn.Linear(3, 4, bias=False)
        self.down_proj = torch.nn.Linear(4, 3, bias=False)
        self.fail = fail

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        if self.fail:
            raise RuntimeError("intentional failure")
        return self.down_proj(F.silu(gate) * up)


class QwenSwigluCaptureTest(unittest.TestCase):
    def assert_clean(self, mlp):
        self.assertEqual(len(mlp.gate_proj._forward_hooks), 0)
        self.assertEqual(len(mlp.up_proj._forward_hooks), 0)
        self.assertEqual(len(mlp.down_proj._forward_pre_hooks), 0)

    def test_captures_exact_product_boundary_and_preserves_output(self):
        mlp = TinyMLP()
        x = torch.randn(1, 2, 3)
        with torch.inference_mode():
            expected = mlp(x)
            output, captures = capture_mlp_branches({"mlp": mlp}, lambda: mlp(x))
            tensors = captures["mlp"]
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(tensors["product"], F.silu(tensors["gate"]) * tensors["up"],
                                       rtol=0, atol=0)
            self.assertEqual(tensors["gate"].shape, (1, 2, 4))
        self.assert_clean(mlp)

    def test_removes_hooks_on_failure(self):
        mlp = TinyMLP(fail=True)
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            capture_mlp_branches({"mlp": mlp}, lambda: mlp(torch.ones(1, 3)))
        self.assert_clean(mlp)

    def test_multiple_calls_are_rejected_and_hooks_removed(self):
        mlp = TinyMLP()
        def execute():
            mlp(torch.ones(1, 3))
            return mlp(torch.ones(1, 3))
        with self.assertRaisesRegex(RuntimeError, "Expected one"):
            capture_mlp_branches({"mlp": mlp}, execute)
        self.assert_clean(mlp)


if __name__ == "__main__":
    unittest.main()
