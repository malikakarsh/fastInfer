"""MLP routing, fallback, capture and restoration of the scoped replacement."""

import unittest
from unittest.mock import Mock, patch

import torch

from fastinfer.decode_swiglu import eligible_swiglu_input, use_decode_swiglu
from fastinfer.kernels.swiglu import reference_swiglu


class TinyMLP(torch.nn.Module):
    def __init__(self, *, device="cpu", dtype=torch.float32, real_shapes=False):
        super().__init__()
        hidden, intermediate = (1536, 8960) if real_shapes else (3, 4)
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False, device=device, dtype=dtype)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False, device=device, dtype=dtype)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False, device=device, dtype=dtype)
        self.act_fn = torch.nn.SiLU()

    def forward(self, hidden_states):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class DecodeSwigluTest(unittest.TestCase):
    def test_cpu_fallback_preserves_gradients_and_restores(self):
        mlp = TinyMLP()
        x = torch.randn(1, 1, 3, requires_grad=True)
        kernel = Mock(side_effect=AssertionError("unexpected dispatch"))
        expected = mlp(x)
        with use_decode_swiglu({"mlp": mlp}, kernel=kernel) as counts:
            actual = mlp(x)
            actual.sum().backward()
        torch.testing.assert_close(actual, expected)
        self.assertIsNotNone(x.grad)
        self.assertEqual(counts["fallback_calls"], 1)
        self.assertEqual(counts["stage_calls"], 0)
        self.assertNotIn("forward", mlp.__dict__)

    def test_each_module_keeps_its_projections_and_captures_product(self):
        # Small output/input projections, but validated intermediate width for reshape.
        a, b = TinyMLP().eval(), TinyMLP().eval()
        for mlp in (a, b):
            mlp.gate_proj = torch.nn.Linear(3, 8960, bias=False)
            mlp.up_proj = torch.nn.Linear(3, 8960, bias=False)
            mlp.down_proj = torch.nn.Linear(8960, 3, bias=False)
        x = torch.randn(1, 1, 3)
        captures = []
        def capture(name, gate, up, product):
            captures.append((name, gate, up, product))
        with torch.inference_mode(), patch("fastinfer.decode_swiglu.eligible_swiglu_input", return_value=True):
            expected = b(a(x))
            with use_decode_swiglu({"a": a, "b": b}, kernel=reference_swiglu, capture=capture) as counts:
                actual = b(a(x))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for _, gate, up, product in captures:
                torch.testing.assert_close(product, reference_swiglu(gate, up), rtol=0, atol=0)
        self.assertEqual([r[0] for r in captures], ["a", "b"])
        self.assertEqual(counts["stage_calls"], 2)
        for row in counts["modules"].values():
            self.assertEqual(row, {"stage_calls": 1, "fallback_calls": 0})
        self.assertNotIn("forward", a.__dict__)
        self.assertNotIn("forward", b.__dict__)

    def test_restores_existing_override_and_all_modules_on_error(self):
        a, b = TinyMLP(), TinyMLP()
        a.forward = Mock(return_value=torch.ones(1, 1, 3))
        override = a.forward
        kernel = Mock(side_effect=RuntimeError("kernel failure"))
        b.gate_proj = torch.nn.Linear(3, 8960, bias=False)
        b.up_proj = torch.nn.Linear(3, 8960, bias=False)
        with patch("fastinfer.decode_swiglu.eligible_swiglu_input", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "kernel failure"):
                with use_decode_swiglu({"a": a, "b": b}, kernel=kernel):
                    b(torch.ones(1, 1, 3))
        self.assertIs(a.forward, override)
        self.assertNotIn("forward", b.__dict__)

    def test_eligibility_rejects_training_cpu_and_other_activation(self):
        mlp = TinyMLP().eval()
        self.assertFalse(eligible_swiglu_input(mlp, torch.ones(1, 1, 3)))
        # Simulated metadata checks eligibility without allocating large GPU weights.
        from types import SimpleNamespace
        device = torch.device("cuda:0")
        def linear(shape):
            weight = SimpleNamespace(shape=shape, dtype=torch.float16, device=device,
                                     is_contiguous=lambda: True)
            return SimpleNamespace(training=False, bias=None, weight=weight)
        fake = SimpleNamespace(training=False, act_fn=torch.nn.SiLU(),
                               gate_proj=linear((8960, 1536)), up_proj=linear((8960, 1536)),
                               down_proj=linear((1536, 8960)))
        x = SimpleNamespace(shape=(1, 1, 1536), dtype=torch.float16, device=device,
                            is_cuda=True, is_contiguous=lambda: True)
        with torch.inference_mode():
            self.assertTrue(eligible_swiglu_input(fake, x))
            fake.act_fn = torch.nn.ReLU()
            self.assertFalse(eligible_swiglu_input(fake, x))
            fake.act_fn = torch.nn.SiLU()
            fake.up_proj.bias = object()
            self.assertFalse(eligible_swiglu_input(fake, x))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class DecodeSwigluCudaTest(unittest.TestCase):
    def test_real_dispatch_and_prefill_fallback(self):
        from fastinfer.kernels.triton_swiglu import swiglu
        mlp = TinyMLP(device="cuda", dtype=torch.float16, real_shapes=True).eval()
        x = torch.randn(1, 1, 1536, device="cuda", dtype=torch.float16)
        with torch.inference_mode():
            expected = mlp(x)
            with use_decode_swiglu({"mlp": mlp}, kernel=swiglu) as counts:
                output = mlp(x)
                prefill = x.expand(1, 2, 1536).contiguous()
                fallback = mlp(prefill)
            torch.testing.assert_close(output, expected, rtol=.01, atol=.01)
            torch.testing.assert_close(fallback, mlp(prefill), rtol=0, atol=0)
        self.assertEqual(counts["stage_calls"], 1)
        self.assertEqual(counts["fallback_calls"], 1)
        self.assertNotIn("forward", mlp.__dict__)


if __name__ == "__main__":
    unittest.main()
