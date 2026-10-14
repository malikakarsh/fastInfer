"""Diagnostic output must reproduce production before interpreting its stages."""

from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
try:
    from qwen_swiglu_stages import pytorch_stages, unrounded_swiglu
finally:
    sys.path.pop(0)

try:
    import triton
except ImportError:
    triton = None


class SwigluStageReferenceTest(unittest.TestCase):
    def test_known_values_and_rounding_boundaries(self):
        gate = torch.tensor([[0., 2., -2.]], dtype=torch.float16)
        up = torch.tensor([[100., 3., 4.]], dtype=torch.float16)
        stages = pytorch_stages(gate, up)
        self.assertEqual(stages["silu_fp32"].dtype, torch.float32)
        self.assertEqual(stages["silu_fp16"].dtype, torch.float16)
        torch.testing.assert_close(stages["product_fp16"], F.silu(gate) * up, rtol=0, atol=0)
        torch.testing.assert_close(unrounded_swiglu(gate, up),
                                   (F.silu(gate.float()) * up.float()).half(), rtol=0, atol=0)
        self.assertEqual(stages["product_fp16"][0, 0].item(), 0.)


@unittest.skipUnless(triton is not None and torch.cuda.is_available(), "CUDA Triton required")
class SwigluDiagnosticCudaTest(unittest.TestCase):
    def test_diagnostic_output_matches_production_on_tails_and_selected_shape(self):
        from fastinfer.kernels.swiglu_diagnostic import triton_swiglu_stages
        from fastinfer.kernels.triton_swiglu import swiglu
        torch.manual_seed(11)
        with torch.inference_mode():
            for shape in ((1, 65), (1, 8960), (16, 8960)):
                with self.subTest(shape=shape):
                    gate = 2 * torch.randn(shape, device="cuda", dtype=torch.float16)
                    up = torch.randn_like(gate)
                    stages = triton_swiglu_stages(gate, up)
                    torch.testing.assert_close(stages["product_fp16"], swiglu(gate, up), rtol=0, atol=0)
                    for value in stages.values():
                        self.assertTrue(torch.isfinite(value).all().item())
                    torch.testing.assert_close(stages["silu_fp16"], stages["silu_fp32"].half(), rtol=0, atol=0)

    def test_extremes_zero_and_diagnostic_input_contract(self):
        from fastinfer.kernels.swiglu_diagnostic import triton_swiglu_stages
        from fastinfer.kernels.triton_swiglu import swiglu
        gate = torch.tensor([[-1000., -2., 0., 2., 1000.]], device="cuda", dtype=torch.float16)
        up = torch.tensor([[2., -3., 4., 5., -2.]], device="cuda", dtype=torch.float16)
        stages = triton_swiglu_stages(gate, up)
        torch.testing.assert_close(stages["product_fp16"], swiglu(gate, up), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "CUDA FP16"):
            triton_swiglu_stages(gate.float(), up.float())
        with torch.enable_grad(), self.assertRaisesRegex(RuntimeError, "forward-only"):
            triton_swiglu_stages(gate.clone().requires_grad_(True), up)


if __name__ == "__main__":
    unittest.main()
