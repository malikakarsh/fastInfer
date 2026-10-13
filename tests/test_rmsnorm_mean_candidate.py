"""Check candidate benchmark statistics and GPU arithmetic before model dispatch."""

from pathlib import Path
import sys
import unittest

import torch

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
try:
    from rmsnorm_mean_candidate_check import paired_savings
finally:
    sys.path.pop(0)


class CandidateTest(unittest.TestCase):
    def test_paired_savings_preserve_rounds(self):
        result = paired_savings([5, 2, 9], [4, 4, 6])
        self.assertEqual(result["samples_us"], [1, -2, 3])
        self.assertEqual(result["median_us"], 1)
        self.assertEqual(result["candidate_faster_rounds"], 2)

    def test_invalid_pairs_rejected(self):
        for baseline, candidate in (([], []), ([1], [1, 2])):
            with self.assertRaises(ValueError):
                paired_savings(baseline, candidate)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_output_and_mean_on_known_extreme_and_random_inputs(self):
        from fastinfer.kernels.rmsnorm import reference_rmsnorm
        from fastinfer.kernels.rmsnorm_mean_candidate import rmsnorm_candidate

        with torch.inference_mode():
            torch.manual_seed(11)
            weight = torch.randn(1536, device="cuda", dtype=torch.float16)
            inputs = [torch.zeros(1, 1, 1536, device="cuda", dtype=torch.float16),
                      torch.full((1, 1, 1536), 60000., device="cuda", dtype=torch.float16),
                      torch.randn(1, 1, 1536, device="cuda", dtype=torch.float16)]
            for x in inputs:
                actual, mean = rmsnorm_candidate(x, weight, return_mean=True)
                expected = reference_rmsnorm(x, weight)
                torch.testing.assert_close(actual, expected, rtol=.002, atol=.002)
                torch.testing.assert_close(actual, rmsnorm_candidate(x, weight), rtol=0, atol=0)
                # A double-precision mean independently checks coverage of all features.
                torch.testing.assert_close(mean.double(), x.double().square().mean().reshape(1),
                                           rtol=1e-6, atol=1e-12)
                self.assertTrue(torch.isfinite(actual).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_rejects_unsupported_shapes_and_gradients(self):
        from fastinfer.kernels.rmsnorm_mean_candidate import rmsnorm_candidate

        weight = torch.ones(1536, device="cuda", dtype=torch.float16)
        with self.assertRaises(ValueError):
            rmsnorm_candidate(torch.ones(2, 1536, device="cuda", dtype=torch.float16), weight)
        x = torch.ones(1, 1536, device="cuda", dtype=torch.float16, requires_grad=True)
        with torch.enable_grad(), self.assertRaises(RuntimeError):
            rmsnorm_candidate(x, weight)


if __name__ == "__main__":
    unittest.main()
