"""Accuracy experiments must retain boundaries and reproduce the original control."""

import unittest

import torch
import torch.nn.functional as F

try:
    import triton
except ImportError:
    triton = None


@unittest.skipUnless(triton is not None and torch.cuda.is_available(), "CUDA Triton required")
class SwigluArithmeticCandidateTest(unittest.TestCase):
    def test_all_variants_on_tails_and_extremes(self):
        from fastinfer.kernels.swiglu_arithmetic_candidate import VARIANTS, arithmetic_stages
        from fastinfer.kernels.triton_swiglu import swiglu
        torch.manual_seed(11)
        with torch.inference_mode():
            inputs = [(2 * torch.randn(1, 65, device="cuda", dtype=torch.float16),
                       torch.randn(1, 65, device="cuda", dtype=torch.float16)),
                      (2 * torch.randn(1, 8960, device="cuda", dtype=torch.float16),
                       torch.randn(1, 8960, device="cuda", dtype=torch.float16)),
                      (torch.tensor([[-1000., -20., -2., 0., 2., 20., 1000.]], device="cuda", dtype=torch.float16),
                       torch.tensor([[2., -3., 4., 5., -3., 0., -2.]], device="cuda", dtype=torch.float16))]
            for gate, up in inputs:
                expected = F.silu(gate) * up
                production = swiglu(gate, up)
                for variant in VARIANTS:
                    with self.subTest(shape=gate.shape, variant=variant):
                        stages, _ = arithmetic_stages(gate, up, variant=variant)
                        torch.testing.assert_close(stages["product_fp16"], expected, rtol=.002, atol=2e-4)
                        torch.testing.assert_close(stages["silu_fp16"], stages["silu_fp32"].half(), rtol=0, atol=0)
                        for stage in stages.values():
                            self.assertTrue(torch.isfinite(stage).all().item())
                        if variant == "fast_exp_reciprocal":
                            torch.testing.assert_close(stages["product_fp16"], production, rtol=0, atol=0)

    def test_rejects_invalid_variant_dtype_and_gradients(self):
        from fastinfer.kernels.swiglu_arithmetic_candidate import arithmetic_stages
        gate = torch.ones(1, 3, device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "unknown variant"):
            arithmetic_stages(gate, gate, variant="unknown")
        with self.assertRaisesRegex(ValueError, "CUDA FP16"):
            arithmetic_stages(gate.float(), gate.float(), variant="fast_exp_direct")
        with torch.enable_grad(), self.assertRaisesRegex(RuntimeError, "forward-only"):
            arithmetic_stages(gate.clone().requires_grad_(True), gate, variant="fast_exp_direct")


if __name__ == "__main__":
    unittest.main()
