"""Scoped normalization dispatch, module identity and restoration checks."""

import unittest
from unittest.mock import Mock, patch

import torch
from fastinfer.decode_rmsnorm import eligible_norm_input, use_decode_rmsnorm
from fastinfer.kernels.rmsnorm import reference_rmsnorm


class ToyNorm(torch.nn.Module):
    def __init__(self, scale=1., eps=1e-6, *, hidden=2, device="cpu", dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((hidden,), scale, device=device, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return reference_rmsnorm(hidden_states, self.weight, eps=self.variance_epsilon)


class DecodeNormTest(unittest.TestCase):
    def test_cpu_fallback_keeps_gradients_and_restores(self):
        norm = ToyNorm()
        x = torch.tensor([[[3., 4.]]], requires_grad=True)
        expected = norm(x)
        kernel = Mock(side_effect=AssertionError("unexpected kernel call"))
        with use_decode_rmsnorm({"norm": norm}, kernel=kernel) as counts:
            actual = norm(x)
            actual.sum().backward()
        torch.testing.assert_close(actual, expected)
        self.assertIsNotNone(x.grad)
        self.assertEqual(counts["fallback_calls"], 1)
        self.assertEqual(counts["rmsnorm_calls"], 0)
        self.assertNotIn("forward", norm.__dict__)

    def test_each_module_uses_its_own_weight_and_epsilon(self):
        a, b = ToyNorm(1., 1e-6).eval(), ToyNorm(2., 1e-5).eval()
        x = torch.tensor([[[3., 4.]]])
        kernel = Mock(wraps=reference_rmsnorm)
        with torch.inference_mode(), patch("fastinfer.decode_rmsnorm.eligible_norm_input", return_value=True):
            expected = b(a(x))
            with use_decode_rmsnorm({"a": a, "b": b}, kernel=kernel) as counts:
                actual = b(a(x))
            torch.testing.assert_close(actual, expected)
        self.assertIs(kernel.call_args_list[0].args[1], a.weight)
        self.assertIs(kernel.call_args_list[1].args[1], b.weight)
        self.assertEqual(kernel.call_args_list[0].kwargs["eps"], 1e-6)
        self.assertEqual(kernel.call_args_list[1].kwargs["eps"], 1e-5)
        self.assertEqual(counts["rmsnorm_calls"], 2)
        for row in counts["modules"].values():
            self.assertEqual(row, {"rmsnorm_calls": 1, "fallback_calls": 0})
        self.assertNotIn("forward", a.__dict__)
        self.assertNotIn("forward", b.__dict__)

    def test_all_modules_restored_after_kernel_failure(self):
        a, b = ToyNorm(), ToyNorm()
        original = Mock(return_value=torch.ones(1, 1, 2))
        a.forward = original
        kernel = Mock(side_effect=RuntimeError("kernel failure"))
        with patch("fastinfer.decode_rmsnorm.eligible_norm_input", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "kernel failure"):
                with use_decode_rmsnorm({"a": a, "b": b}, kernel=kernel):
                    b(torch.ones(1, 1, 2))
        self.assertIs(a.forward, original)
        self.assertNotIn("forward", b.__dict__)

    def test_eligibility_rejects_other_shapes_and_devices(self):
        from types import SimpleNamespace
        weight = SimpleNamespace(shape=(1536,), dtype=torch.float16,
                                 device=torch.device("cuda:0"), is_contiguous=lambda: True)
        norm = SimpleNamespace(weight=weight, training=False)
        fields = dict(shape=(1, 1, 1536), dtype=torch.float16, is_cuda=True,
                      device=torch.device("cuda:0"), is_contiguous=lambda: True)
        with torch.inference_mode():
            self.assertTrue(eligible_norm_input(norm, SimpleNamespace(**fields)))
            for change in ({"shape": (1, 16, 1536)}, {"dtype": torch.float32},
                           {"device": torch.device("cuda:1")}, {"is_cuda": False},
                           {"is_contiguous": lambda: False}):
                self.assertFalse(eligible_norm_input(norm, SimpleNamespace(**(fields | change))))
            norm.training = True
            self.assertFalse(eligible_norm_input(norm, SimpleNamespace(**fields)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecodeNormCudaTest(unittest.TestCase):
    def test_real_kernel_dispatch_and_prefill_fallback(self):
        norm = ToyNorm(hidden=1536, device="cuda", dtype=torch.float16).eval()
        x = torch.randn(1, 1, 1536, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            expected = norm(x)
            with use_decode_rmsnorm({"norm": norm}) as counts:
                output = norm(x)
                prefill = x.expand(1, 2, 1536).contiguous()
                fallback = norm(prefill)
            torch.testing.assert_close(output, expected, rtol=.002, atol=.002)
            torch.testing.assert_close(fallback, norm(prefill))
        self.assertEqual(counts["rmsnorm_calls"], 1)
        self.assertEqual(counts["fallback_calls"], 1)
        self.assertNotIn("forward", norm.__dict__)


if __name__ == "__main__":
    unittest.main()
