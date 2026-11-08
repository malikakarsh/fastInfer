"""Batch-one greedy speculative decoding for ordinary Qwen2 DynamicCaches.

Both paths include fresh prefill. The first generated token comes from target
prefill. Thereafter target cache holds all committed history except its final
token; that pending token starts the next verification block.
"""

from time import perf_counter

import torch
from transformers import DynamicCache

from .speculative import DecodeResult


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def crop_to(cache, length):
    """Negative crop counts work with both old and new Transformers APIs."""
    remove = cache.get_seq_length() - length
    if remove < 0:
        raise ValueError("cannot grow a cache by cropping")
    if remove:
        cache.crop(-remove)
    if cache.get_seq_length() != length:
        raise RuntimeError("cache rollback did not restore the expected length")


def _forward(model, tokens, cache, keep=0):
    return model(input_ids=tokens, past_key_values=cache, use_cache=True,
                 logits_to_keep=keep, return_dict=True).logits


def _check(model, prompt, max_new_tokens):
    if (prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0
            or prompt.dtype != torch.long or max_new_tokens < 1):
        raise ValueError("nonempty batch-one int64 prompt and positive token budget required")
    if model.config.model_type != "qwen2" or getattr(model.config, "use_sliding_window", False):
        raise ValueError("this cache experiment supports ordinary full-attention Qwen2 models only")


@torch.inference_mode()
def cached_decode(target, prompt, *, max_new_tokens, eos_token_ids=(), draft=None,
                  gamma=3, clock=perf_counter, observer=None):
    """Target-only if draft=None; otherwise verify k proposals per target call.

    Models must have compatible token IDs (checked by the benchmark loader).
    All timing includes model calls, Python dispatch, draft prefill and rollback.
    Target and draft output vocabularies may differ in unused padding rows.
    Optional observer(event, payload) is an accuracy-diagnostic hook, disabled
    for benchmarks. It must not mutate tensors/caches; exceptions can stop replay.
    """
    _check(target, prompt, max_new_tokens)
    if gamma < 1:
        raise ValueError("gamma must be positive")
    if draft is not None:
        _check(draft, prompt, max_new_tokens)
    result = DecodeResult()
    eos = set(eos_token_ids)
    target_cache = DynamicCache(config=target.config)
    device = prompt.device
    synchronize(device)
    start = clock()
    if observer is not None:
        observer("target_before", {"phase": "prefill", "cache": target_cache,
                                   "input_ids": prompt, "history": prompt})
    logits = _forward(target, prompt, target_cache, keep=1)
    if observer is not None:
        observer("target_after", {"cache": target_cache, "logits": logits})
    first = int(logits[0, -1].argmax())
    result.generated.append(first)
    result.target_calls = 1
    if observer is not None:
        observer("emit", {"index": 0, "token": first, "row": 0, "logits": logits[0, -1]})
    history = torch.cat((prompt, prompt.new_tensor([[first]])), dim=1)
    draft_cache = None
    if draft is not None and first not in eos and max_new_tokens > 1:
        draft_cache = DynamicCache(config=draft.config)
        if observer is not None:
            observer("draft_before", {"cache": draft_cache, "input_ids": prompt, "history": prompt})
        _forward(draft, prompt, draft_cache, keep=1)
        if observer is not None:
            observer("draft_after", {"cache": draft_cache})
        result.draft_calls += 1
    synchronize(device)
    prefill_end = clock()

    while len(result.generated) < max_new_tokens and result.generated[-1] not in eos:
        if draft is None:
            if observer is not None:
                observer("target_before", {"phase": "decode", "cache": target_cache,
                                           "input_ids": history[:, -1:], "history": history})
            logits = _forward(target, history[:, -1:], target_cache, keep=1)
            if observer is not None:
                observer("target_after", {"cache": target_cache, "logits": logits})
            result.target_calls += 1
            token = int(logits[0, -1].argmax())
            result.generated.append(token)
            if observer is not None:
                observer("emit", {"index": len(result.generated) - 1, "token": token,
                                  "row": 0, "logits": logits[0, -1]})
            history = torch.cat((history, history.new_tensor([[token]])), dim=1)
            continue

        length = history.shape[1]
        proposal_history = history
        proposals = []
        remaining = max_new_tokens - len(result.generated)
        for _ in range(min(gamma, remaining)):
            suffix = proposal_history[:, draft_cache.get_seq_length():]
            if observer is not None:
                observer("draft_before", {"cache": draft_cache, "input_ids": suffix,
                                          "history": proposal_history})
            logits = _forward(draft, suffix, draft_cache, keep=1)
            if observer is not None:
                observer("draft_after", {"cache": draft_cache})
            result.draft_calls += 1
            token = int(logits[0, -1].argmax())
            proposals.append(token)
            proposal_history = torch.cat((proposal_history, history.new_tensor([[token]])), dim=1)
            # Only target EOS defines stopping, including when draft was instruction-tuned.
            if token in eos:
                break
        result.proposed += len(proposals)
        # Row 0 (pending committed token) predicts proposal 0; final row predicts bonus.
        if observer is not None:
            observer("target_before", {"phase": "verification", "cache": target_cache,
                                       "input_ids": proposal_history[:, length - 1:],
                                       "history": history, "proposals": list(proposals)})
        logits = _forward(target, proposal_history[:, length - 1:], target_cache)
        if observer is not None:
            observer("target_after", {"cache": target_cache, "logits": logits})
        result.target_calls += 1
        predictions = logits[0].argmax(dim=-1).tolist()
        accepted = 0
        finished = False
        for index, proposal in enumerate(proposals):
            if proposal != predictions[index]:
                break
            result.generated.append(proposal)
            accepted += 1
            result.accepted += 1
            if observer is not None:
                observer("emit", {"index": len(result.generated) - 1, "token": proposal,
                                  "row": index, "logits": logits[0, index], "source": "accepted_proposal"})
            if proposal in eos or len(result.generated) == max_new_tokens:
                finished = True
                break
        if finished:
            break
        correction_or_bonus = predictions[accepted]
        result.generated.append(correction_or_bonus)
        if observer is not None:
            observer("emit", {"index": len(result.generated) - 1, "token": correction_or_bonus,
                              "row": accepted, "logits": logits[0, accepted],
                              "source": "bonus" if accepted == len(proposals) else "correction"})
        committed = proposals[:accepted] + [correction_or_bonus]
        history = torch.cat((history, history.new_tensor([committed])), dim=1)
        # Keep accepted context but leave the final committed token pending.
        if observer is not None:
            observer("crop_before", {"target_cache": target_cache, "draft_cache": draft_cache,
                                     "history": history, "target_length": history.shape[1] - 1,
                                     "draft_length": min(draft_cache.get_seq_length(), history.shape[1] - 1)})
        crop_to(target_cache, history.shape[1] - 1)
        crop_to(draft_cache, min(draft_cache.get_seq_length(), history.shape[1] - 1))
        if observer is not None:
            observer("crop_after", {"target_cache": target_cache, "draft_cache": draft_cache,
                                    "history": history})

    synchronize(device)
    end = clock()
    return result, {"total_s": end - start, "prefill_s": prefill_end - start,
                    "decode_s": end - prefill_end,
                    "decode_tokens": len(result.generated) - 1}
