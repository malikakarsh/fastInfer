"""Exercise real DynamicCache rollback and cached causal logit alignment on CPU."""

from types import SimpleNamespace
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from fastinfer.speculative_cached import cached_decode, crop_to


class ToyCachedModel:
    def __init__(self, rule, vocabulary=16):
        self.config = Qwen2Config(vocab_size=vocabulary, hidden_size=8,
            intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=1, use_sliding_window=False)
        self.rule = rule
        self.histories = []

    def __call__(self, input_ids, past_key_values, logits_to_keep=0, **kwargs):
        length = past_key_values.get_seq_length()
        if length:
            old = past_key_values.layers[0].keys[0, 0, :, 0].long().tolist()
        else:
            old = []
        new = input_ids[0].tolist()
        self.histories.append(old + new)
        values = input_ids.float().reshape(1, 1, -1, 1)
        past_key_values.update(values, values, 0)
        logits = torch.full((1, len(new), self.config.vocab_size), -10.)
        for i in range(len(new)):
            logits[0, i, self.rule(old + new[:i + 1])] = 10.
        return SimpleNamespace(logits=logits[:, -logits_to_keep:] if logits_to_keep else logits)


def rule(history):
    return (sum(history) + len(history)) % 11


class CachedSpeculativeTest(unittest.TestCase):
    def test_rollbacks_acceptance_and_different_padding_sizes(self):
        prompt = torch.tensor([[2, 4]])
        for draft_rule in (rule, lambda h: (rule(h) + 1) % 11,
                           lambda h: rule(h) if len(h) % 2 == 0 else (rule(h) + 1) % 11):
            for gamma in (1, 3, 12):
                for budget in (1, 2, 7, 12):
                    with self.subTest(rule=draft_rule, gamma=gamma, budget=budget):
                        expected, _ = cached_decode(ToyCachedModel(rule), prompt, max_new_tokens=budget)
                        actual, timing = cached_decode(ToyCachedModel(rule), prompt,
                            draft=ToyCachedModel(draft_rule, vocabulary=12), max_new_tokens=budget, gamma=gamma)
                        self.assertEqual(actual.generated, expected.generated)
                        self.assertEqual(len(actual.generated), budget)
                        self.assertGreaterEqual(timing["total_s"], timing["prefill_s"])

    def test_eos_and_discarded_draft_eos(self):
        prompt = torch.tensor([[0]])
        for draft_rule in (lambda h: len(h), lambda h: 3, lambda h: 0):
            for gamma in (1, 2, 3, 5):
                actual, _ = cached_decode(ToyCachedModel(lambda h: len(h)), prompt,
                    draft=ToyCachedModel(draft_rule), max_new_tokens=8, gamma=gamma, eos_token_ids=[3])
                self.assertEqual(actual.generated, [1, 2, 3])
        first, _ = cached_decode(ToyCachedModel(lambda h: 3), prompt,
            draft=ToyCachedModel(rule), max_new_tokens=8, eos_token_ids=[3])
        self.assertEqual(first.generated, [3])
        self.assertEqual(first.draft_calls, 0)

    def test_tiny_real_qwen_matches_uncached_greedy(self):
        torch.manual_seed(7)
        config = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            use_sliding_window=False, attention_dropout=0.)
        target, draft = Qwen2ForCausalLM(config).eval(), Qwen2ForCausalLM(config).eval()
        prompt = torch.tensor([[2, 5, 7]])
        history = prompt.clone()
        expected = []
        with torch.inference_mode():
            for _ in range(10):
                token = target(history, use_cache=False).logits[:, -1].argmax(-1)
                expected.append(int(token))
                history = torch.cat((history, token[:, None]), dim=1)
        for candidate in (None, target, draft):
            actual, _ = cached_decode(target, prompt, draft=candidate, max_new_tokens=10)
            self.assertEqual(actual.generated, expected)


if __name__ == "__main__":
    unittest.main()
