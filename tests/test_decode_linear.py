"""Dispatch safety and restoration for the experimental decode integration."""

import unittest
from unittest.mock import Mock, patch

import torch
from fastinfer.decode_linear import eligible_decode_input, use_decode_gemv


class DecodeLinearTest(unittest.TestCase):
    def test_cpu_fallback_preserves_output_and_gradients(self):
        linear = torch.nn.Linear(3, 2)
        x = torch.randn(1, 1, 3, requires_grad=True)
        expected = linear(x)
        kernel = Mock(side_effect=AssertionError("must not dispatch"))
        with use_decode_gemv(linear, kernel=kernel) as counts:
            actual = linear(x)
            actual.sum().backward()
        torch.testing.assert_close(actual, expected)
        self.assertIsNotNone(x.grad)
        self.assertEqual(counts, {"gemv_calls": 0, "fallback_calls": 1})
        self.assertNotIn("forward", linear.__dict__)

    def test_dispatch_preserves_weight_and_output_shape(self):
        linear = torch.nn.Linear(1536, 8960, bias=False).eval()
        x = torch.ones(1, 1, 1536)
        kernel = Mock(return_value=torch.ones(1, 8960))
        # Exercise dispatch plumbing on CPU; eligibility is independently tested.
        with patch("fastinfer.decode_linear.eligible_decode_input", return_value=True):
            with use_decode_gemv(linear, kernel=kernel) as counts:
                result = linear(x)
        self.assertEqual(tuple(result.shape), (1, 1, 8960))
        self.assertIs(kernel.call_args.args[1], linear.weight)
        self.assertEqual(kernel.call_args.kwargs, {"block_rows": 1, "num_warps": 4})
        self.assertEqual(counts, {"gemv_calls": 1, "fallback_calls": 0})

    def test_restores_existing_forward_after_kernel_failure(self):
        linear = torch.nn.Linear(3, 2)
        original = Mock(return_value=torch.ones(1, 1, 2))
        linear.forward = original
        with patch("fastinfer.decode_linear.eligible_decode_input", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "kernel failure"):
                with use_decode_gemv(linear, kernel=Mock(side_effect=RuntimeError("kernel failure"))):
                    linear(torch.ones(1, 1, 1536))
        self.assertIs(linear.forward, original)

    def test_eligibility_rejects_unsupported_inputs(self):
        # Metadata doubles test all guards without allocating CUDA tensors.
        from types import SimpleNamespace
        weight = SimpleNamespace(shape=(8960, 1536), dtype=torch.float16,
                                 device=torch.device("cuda:0"), is_contiguous=lambda: True)
        linear = SimpleNamespace(weight=weight, training=False, bias=None)
        def tensor(**changes):
            fields = dict(shape=(1, 1, 1536), dtype=torch.float16, is_cuda=True,
                          device=torch.device("cuda:0"), is_contiguous=lambda: True)
            fields.update(changes)
            return SimpleNamespace(**fields)
        with torch.inference_mode():
            self.assertTrue(eligible_decode_input(linear, tensor()))
            for changes in ({"shape": (1, 2, 1536)}, {"shape": (2, 1, 1536)},
                            {"dtype": torch.float32}, {"is_cuda": False},
                            {"device": torch.device("cuda:1")},
                            {"is_contiguous": lambda: False}):
                self.assertFalse(eligible_decode_input(linear, tensor(**changes)))
            linear.training = True
            self.assertFalse(eligible_decode_input(linear, tensor()))
            linear.training = False
            linear.bias = object()
            self.assertFalse(eligible_decode_input(linear, tensor()))
            linear.bias = None
            weight.shape = (8192, 1536)
            self.assertFalse(eligible_decode_input(linear, tensor()))
            weight.shape = (8960, 1536)
        with torch.enable_grad():
            self.assertFalse(eligible_decode_input(linear, tensor()))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecodeLinearCudaTest(unittest.TestCase):
    def test_actual_kernel_in_linear_module(self):
        linear = torch.nn.Linear(1536, 8960, bias=False,
                                 device="cuda", dtype=torch.float16).eval()
        x = torch.randn(1, 1, 1536, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            reference = torch.nn.functional.linear(x.float(), linear.weight.float())
            with use_decode_gemv(linear) as counts:
                output = linear(x)
                prefill = x.expand(1, 2, 1536).contiguous()
                fallback = linear(prefill)
            torch.testing.assert_close(output.float(), reference, rtol=.01, atol=.01)
            torch.testing.assert_close(fallback, linear(prefill))
        self.assertEqual(counts, {"gemv_calls": 1, "fallback_calls": 1})
        self.assertNotIn("forward", linear.__dict__)


if __name__ == "__main__":
    unittest.main()
