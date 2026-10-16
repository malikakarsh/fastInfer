"""Exercise new inputs, rounding boundaries and the output-only arithmetic."""

import unittest

import torch

try:
    import triton
except ImportError:
    triton = None


@unittest.skipUnless(triton is not None and torch.cuda.is_available(), "CUDA Triton required")
class PreciseSwigluTest(unittest.TestCase):
    def check_pair(self, gate, up):
        from fastinfer.kernels.swiglu import reference_swiglu
        from fastinfer.kernels.swiglu_arithmetic_candidate import arithmetic_stages
        from fastinfer.kernels.swiglu_precise_candidate import swiglu_precise
        with torch.inference_mode():
            actual = swiglu_precise(gate, up)
            diagnostic, _ = arithmetic_stages(gate, up, variant="libdevice_exp_div_rn")
            torch.testing.assert_close(actual, diagnostic["product_fp16"], rtol=0, atol=0)
            activation = (gate.double() * torch.sigmoid(gate.double())).half()
            oracle = (activation.double() * up.double()).half()
            for expected in (reference_swiglu(gate, up), oracle):
                torch.testing.assert_close(actual, expected, rtol=.002, atol=2e-4)
            self.assertTrue(torch.isfinite(actual).all().item())
            self.assertEqual(actual.shape, gate.shape)
            self.assertEqual(actual.dtype, gate.dtype)

    def test_fresh_seeds_shapes_and_tails(self):
        for seed in (21, 22, 23):
            torch.manual_seed(seed)
            for shape in ((1,), (7,), (3, 65), (1, 8960), (16, 8960),
                          (128, 8960), (1, 1, 8960)):
                with self.subTest(seed=seed, shape=shape):
                    gate = 2 * torch.randn(shape, device="cuda", dtype=torch.float16)
                    self.check_pair(gate, torch.randn_like(gate))

    def test_all_finite_fp16_gate_values(self):
        # Includes subnormals, signed zeros and both sides of rounding boundaries.
        bits = torch.arange(-32768, 32768, device="cuda", dtype=torch.int32).to(torch.int16)
        gate = bits.view(torch.float16)
        gate = gate[torch.isfinite(gate)].contiguous()
        for multiplier in (-.75, .5, 1.):
            with self.subTest(multiplier=multiplier):
                self.check_pair(gate, torch.full_like(gate, multiplier))

    def test_contract(self):
        from fastinfer.kernels.swiglu_precise_candidate import swiglu_precise
        gate = torch.ones(2, 3, device="cuda", dtype=torch.float16)
        for a, b in ((gate.float(), gate.float()), (gate.cpu(), gate.cpu()),
                     (gate.t(), gate.t()), (gate, gate[:, :1])):
            with self.subTest(shape=a.shape, device=a.device, dtype=a.dtype):
                with self.assertRaises((ValueError, TypeError)):
                    swiglu_precise(a, b)
        with torch.enable_grad(), self.assertRaisesRegex(RuntimeError, "forward-only"):
            swiglu_precise(gate.clone().requires_grad_(True), gate)


if __name__ == "__main__":
    unittest.main()
