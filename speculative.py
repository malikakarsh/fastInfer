"""Greedy speculative decoding reference, without KV caches or sampling.

A model is a callable taking CPU token IDs [length] and returning causal logits
[length, vocabulary]. Row i predicts the token after input position i. Draft
and target must share token IDs. This intentionally recomputes complete prefixes
to make verification logic inspectable; it is not a speed-optimized engine.
"""

from dataclasses import dataclass, field

import torch


@dataclass
class DecodeResult:
    generated: list[int] = field(default_factory=list)
    target_calls: int = 0
    draft_calls: int = 0
    proposed: int = 0
    accepted: int = 0


def _validate(prompt, max_new_tokens, eos_token_id):
    if (prompt.device.type != "cpu" or prompt.dtype != torch.long
            or prompt.ndim != 1 or prompt.numel() == 0):
        raise ValueError("prompt must be nonempty CPU int64 token IDs [length]")
    if (prompt < 0).any():
        raise ValueError("token IDs must be nonnegative")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if eos_token_id is not None and (not isinstance(eos_token_id, int) or eos_token_id < 0):
        raise ValueError("eos_token_id must be a nonnegative integer or None")


def _predictions(model, tokens):
    logits = model(tokens)
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 2
            or logits.shape[0] != tokens.numel() or logits.shape[1] == 0
            or logits.device.type != "cpu" or not logits.is_floating_point()):
        raise ValueError("model must return floating CPU logits [length, vocabulary]")
    if not torch.isfinite(logits).all():
        raise ValueError("model logits must be finite")
    return logits.argmax(dim=-1), logits.shape[1]


@torch.inference_mode()
def greedy_decode(target, prompt, *, max_new_tokens, eos_token_id=None):
    """Target-only baseline; EOS is included in the generated token list."""
    _validate(prompt, max_new_tokens, eos_token_id)
    result = DecodeResult()
    history = prompt.clone()
    for _ in range(max_new_tokens):
        predictions, _ = _predictions(target, history)
        result.target_calls += 1
        token = int(predictions[-1])
        result.generated.append(token)
        if token == eos_token_id:
            break
        history = torch.cat((history, history.new_tensor([token])))
    return result


@torch.inference_mode()
def speculative_decode(target, draft, prompt, *, max_new_tokens, gamma=3, eos_token_id=None):
    """Accept a greedy matching prefix, then emit a correction or bonus token.

    One target call verifies each proposal block. Rejected proposals never enter
    the committed history. Equal scores use torch.argmax's first-index policy.
    Causality and shared vocabulary semantics are caller requirements.
    """
    _validate(prompt, max_new_tokens, eos_token_id)
    if not isinstance(gamma, int) or gamma < 1:
        raise ValueError("gamma must be a positive integer")
    result = DecodeResult()
    history = prompt.clone()
    while len(result.generated) < max_new_tokens:
        prefix_length = history.numel()
        draft_history = history
        proposals = []
        draft_vocab = None
        remaining = max_new_tokens - len(result.generated)
        for _ in range(min(gamma, remaining)):
            predictions, vocabulary = _predictions(draft, draft_history)
            if draft_vocab is not None and vocabulary != draft_vocab:
                raise ValueError("draft vocabulary size changed during generation")
            draft_vocab = vocabulary
            token = int(predictions[-1])
            proposals.append(token)
            result.draft_calls += 1
            draft_history = torch.cat((draft_history, history.new_tensor([token])))
            if token == eos_token_id:
                break
        result.proposed += len(proposals)

        predictions, target_vocab = _predictions(target, draft_history)
        result.target_calls += 1
        if target_vocab != draft_vocab:
            raise ValueError("draft and target must share a vocabulary size and token IDs")
        # The last PROMPT row predicts proposal 0, not the first proposal row.
        verified = predictions[prefix_length - 1:]
        accepted = 0
        for index, proposal in enumerate(proposals):
            if proposal != int(verified[index]):
                break
            result.generated.append(proposal)
            result.accepted += 1
            accepted += 1
            if proposal == eos_token_id or len(result.generated) == max_new_tokens:
                return result

        # At rejection, this predicts the correction under the accepted history.
        # If all k proposals matched, row k supplies a bonus next token.
        token = int(verified[accepted])
        result.generated.append(token)
        if token == eos_token_id or len(result.generated) == max_new_tokens:
            return result
        committed = proposals[:accepted] + [token]
        history = torch.cat((history, history.new_tensor(committed)))
    return result
