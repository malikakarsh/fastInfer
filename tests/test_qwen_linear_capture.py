"""Check decode-only capture and hook cleanup without downloading a model."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


path = Path(__file__).resolve().parents[1] / "benchmarks/qwen_linear_check.py"
spec = importlib.util.spec_from_file_location("qwen_linear_check", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TinyModel(torch.nn.Module):
    def __init__(self, fail_decode=False):
        super().__init__()
        self.projection = torch.nn.Linear(3, 2, bias=False)
        self.cache = object()
        self.calls = []
        self.fail_decode = fail_decode

    def forward(self, input_ids, attention_mask, past_key_values=None, **kwargs):
        self.calls.append((input_ids.clone(), attention_mask.clone(), past_key_values))
        x = input_ids.float().unsqueeze(-1).expand(-1, -1, 3)
        self.projection(x)
        if past_key_values is not None and self.fail_decode:
            raise RuntimeError("decode failure")
        logits = torch.zeros((*input_ids.shape, 5))
        logits[..., 3] = 1
        return SimpleNamespace(logits=logits, past_key_values=self.cache)


class DecodeCaptureTest(unittest.TestCase):
    def setUp(self):
        self.encoded = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def test_captures_only_decode_and_passes_cache(self):
        model = TinyModel()
        activation, output, token = module.capture_decode_projection(
            model, model.projection, self.encoded
        )
        self.assertEqual(token, 3)
        self.assertEqual(tuple(activation.shape), (1, 1, 3))
        torch.testing.assert_close(activation, torch.full((1, 1, 3), 3.0))
        with torch.inference_mode():
            torch.testing.assert_close(output, model.projection(activation))
        self.assertEqual(len(model.calls), 2)
        self.assertIs(model.calls[1][2], model.cache)
        self.assertEqual(model.calls[1][1].tolist(), [[1, 1, 1]])
        self.assertEqual(len(model.projection._forward_hooks), 0)

    def test_nonfinite_projection_stops_prefill_and_removes_hooks(self):
        model = TinyModel()
        with torch.no_grad():
            model.projection.weight.fill_(float("nan"))
        with self.assertRaises(module.NonfiniteTensorError) as caught:
            module.capture_decode_projection(model, model.projection, self.encoded)
        self.assertEqual(caught.exception.details["stage"], "prefill:projection.output")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(model.projection._forward_hooks), 0)
        self.assertEqual(len(model.projection._forward_pre_hooks), 0)

    def test_nonfinite_logits_rejected_before_token_selection(self):
        model = TinyModel()
        def corrupt_logits(module_, inputs, output):
            output.logits.fill_(float("inf"))
        handle = model.register_forward_hook(corrupt_logits)
        try:
            with self.assertRaises(module.NonfiniteTensorError) as caught:
                module.capture_decode_projection(model, model.projection, self.encoded)
            self.assertEqual(caught.exception.details["stage"], "prefill.logits")
            self.assertEqual(len(model.calls), 1)
        finally:
            handle.remove()

    def test_removes_hook_when_decode_fails(self):
        model = TinyModel(fail_decode=True)
        with self.assertRaisesRegex(RuntimeError, "decode failure"):
            module.capture_decode_projection(model, model.projection, self.encoded)
        self.assertEqual(len(model.projection._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
