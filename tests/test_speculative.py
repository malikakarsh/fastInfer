"""Deterministic causal models test acceptance, logit alignment and restart."""

import unittest

import torch

from fastinfer.speculative import greedy_decode, speculative_decode


class ToyModel:
    def __init__(self, rule, vocabulary=11):
        self.rule = rule
        self.vocabulary = vocabulary
        self.histories = []

    def __call__(self, tokens):
        self.histories.append(tokens.tolist())
        logits = torch.full((tokens.numel(), self.vocabulary), -10.)
        for i in range(tokens.numel()):
            logits[i, self.rule(tokens[:i + 1].tolist())] = 10.
        return logits


def next_token(history):
    # Depends on the entire history, so incorrect restart histories are visible.
    return (sum(history) + len(history)) % 11


class SpeculativeTest(unittest.TestCase):
    def setUp(self):
        self.prompt = torch.tensor([2, 4], dtype=torch.long)

    def test_all_accepted_and_bonus(self):
        target, draft = ToyModel(next_token), ToyModel(next_token)
        result = speculative_decode(target, draft, self.prompt, max_new_tokens=8, gamma=3)
        baseline = greedy_decode(ToyModel(next_token), self.prompt, max_new_tokens=8)
        self.assertEqual(result.generated, baseline.generated)
        self.assertEqual((result.target_calls, result.draft_calls, result.accepted), (2, 6, 6))
        self.assertEqual(baseline.target_calls, 8)
        self.assertEqual(self.prompt.tolist(), [2, 4])

    def test_immediate_rejection_every_round(self):
        draft = ToyModel(lambda h: (next_token(h) + 1) % 11)
        result = speculative_decode(ToyModel(next_token), draft, self.prompt, max_new_tokens=6)
        baseline = greedy_decode(ToyModel(next_token), self.prompt, max_new_tokens=6)
        self.assertEqual(result.generated, baseline.generated)
        self.assertEqual((result.accepted, result.target_calls), (0, 6))
        self.assertGreater(result.draft_calls, 6)

    def test_partial_acceptance_discards_wrong_context(self):
        draft = ToyModel(lambda h: next_token(h) if len(h) % 2 == 0 else (next_token(h) + 1) % 11)
        result = speculative_decode(ToyModel(next_token), draft, self.prompt, max_new_tokens=7)
        baseline = greedy_decode(ToyModel(next_token), self.prompt, max_new_tokens=7)
        self.assertEqual(result.generated, baseline.generated)
        self.assertGreater(result.accepted, 0)
        self.assertLess(result.accepted, result.proposed)
        # Round two must draft from the committed correction, not rejected tokens.
        self.assertEqual(draft.histories[3], self.prompt.tolist() + baseline.generated[:2])

    def test_budget_and_proposal_lengths(self):
        for gamma in (1, 2, 3, 20):
            for budget in (0, 1, 2, 3, 4, 7):
                with self.subTest(gamma=gamma, budget=budget):
                    result = speculative_decode(ToyModel(next_token), ToyModel(next_token),
                                                self.prompt, max_new_tokens=budget, gamma=gamma)
                    baseline = greedy_decode(ToyModel(next_token), self.prompt, max_new_tokens=budget)
                    self.assertEqual(result.generated, baseline.generated)
                    self.assertEqual(len(result.generated), budget)
                    if budget == 0:
                        self.assertEqual(result.target_calls + result.draft_calls, 0)

    def test_eos_accepted_corrected_bonus_and_rejected(self):
        # Target emits 1, 2, 3, ...; EOS is token 3.
        rule = lambda h: len(h)
        prompt = torch.tensor([0])
        for gamma, draft_rule in ((3, rule), (2, rule), (3, lambda h: 0),
                                  (3, lambda h: 3)):
            with self.subTest(gamma=gamma, draft=draft_rule):
                result = speculative_decode(ToyModel(rule), ToyModel(draft_rule), prompt,
                                            max_new_tokens=8, gamma=gamma, eos_token_id=3)
                self.assertEqual(result.generated, [1, 2, 3])

    def test_invalid_inputs_and_model_contract(self):
        model = ToyModel(next_token)
        for prompt in (torch.tensor([]).long(), torch.tensor([1.]), torch.tensor([[1]]),
                       torch.tensor([-1])):
            with self.assertRaises(ValueError):
                speculative_decode(model, model, prompt, max_new_tokens=2)
        for kwargs in ({"gamma": 0}, {"max_new_tokens": -1}):
            with self.assertRaises(ValueError):
                speculative_decode(model, model, self.prompt, **({"max_new_tokens": 2} | kwargs))
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            speculative_decode(model, ToyModel(lambda h: 0, vocabulary=12),
                               self.prompt, max_new_tokens=2)
        with self.assertRaisesRegex(ValueError, "finite"):
            speculative_decode(lambda x: torch.full((len(x), 11), float("nan")),
                               model, self.prompt, max_new_tokens=2)


if __name__ == "__main__":
    unittest.main()
