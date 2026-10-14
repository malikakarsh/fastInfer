"""GPU checks for fused activation/multiply, tails and rounding."""

import unittest

import torch

try:
    import triton
except ImportError:
    triton = None


@unittest.skipUnless(triton is not None and torch.cuda.is_available(), "CUDA Triton required")
class TritonSwigluTest(unittest.TestCase):
    def check_output(self, gate, up):
        from fastinfer.kernels.swiglu import reference_swiglu
        from fastinfer.kernels.triton_swiglu import swiglu

        output = swiglu(gate, up)
        activated = (gate.double() * torch.sigmoid(gate.double())).to(gate.dtype)
        oracle = (activated.double() * up.double()).to(gate.dtype)
        rtol, atol = (.002, 2e-4) if gate.dtype == torch.float16 else (1e-5, 2e-6)
        self.assertEqual(output.shape, gate.shape)
        self.assertEqual(output.dtype, gate.dtype)
        self.assertEqual(output.device, gate.device)
        self.assertTrue(torch.isfinite(output).all().item())
        for expected in (reference_swiglu(gate, up), oracle):
            torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
        return output

    def test_shapes_dtypes_and_masked_tails(self):
        for dtype in (torch.float16, torch.float32):
            for seed in (0, 1):
                torch.manual_seed(seed)
                for shape in ((1,), (7,), (3, 65), (1, 8960), (16, 8960), (1, 1, 8960)):
                    with self.subTest(dtype=dtype, seed=seed, shape=shape):
                        self.check_output(torch.randn(shape, device="cuda", dtype=dtype),
                                          torch.randn(shape, device="cuda", dtype=dtype))

    def test_signs_large_gaps_and_half_rounding(self):
        for dtype in (torch.float16, torch.float32):
            gate = torch.tensor([-1000., -20., -2., 0., 2., 20., 1000.], device="cuda", dtype=dtype)
            up = torch.tensor([2., -3., 4., 5., -3., 0., -2.], device="cuda", dtype=dtype)
            self.check_output(gate, up)
        # Exercise half rounding thresholds with a dense gate sweep.
        gate = torch.linspace(-8, 8, 4097, device="cuda", dtype=torch.float16)
        up = torch.full_like(gate, 3.125)
        self.check_output(gate, up)

    def test_rejects_unsupported_layout_device_and_gradients(self):
        from fastinfer.kernels.triton_swiglu import swiglu

        with self.assertRaisesRegex(ValueError, "CUDA"):
            swiglu(torch.ones(3), torch.ones(3))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            swiglu(torch.ones(2, 6, device="cuda")[:, ::2], torch.ones(2, 3, device="cuda"))
        for which in (0, 1):
            with torch.enable_grad():
                pair = [torch.ones(3, device="cuda"), torch.ones(3, device="cuda")]
                pair[which].requires_grad_(True)
                with self.assertRaisesRegex(RuntimeError, "forward-only"):
                    swiglu(*pair)


if __name__ == "__main__":
    unittest.main()
