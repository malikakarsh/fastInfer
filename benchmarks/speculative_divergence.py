"""Replay a saved first-block token divergence; accuracy only, no timing.

Loads only the saved target snapshot, never the draft. Uses recorded proposed
tokens only when every proposal was accepted, so the first block is recoverable
without pretending to reconstruct a missing draft trace. All paths use the same
history before the divergent prediction; only execution shape/cache policy varies.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


def locate_divergence(case):
    validation = case.get("validation", {})
    baseline = validation.get("baseline", {}).get("generated", [])
    candidate = validation.get("speculative", {}).get("generated", [])
    difference = next((i for i, pair in enumerate(zip(baseline, candidate)) if pair[0] != pair[1]), None)
    if difference is None:
        raise ValueError("no differing token in the overlapping saved output; length-only failures need another diagnostic")
    return difference, baseline, candidate


def recover_first_block(case, gamma):
    difference, baseline, candidate = locate_divergence(case)
    counts = case["validation"]["speculative"]
    if not 1 <= difference <= gamma:
        raise ValueError("this diagnostic requires divergence in the first verification block after prefill")
    if counts["proposed"] != counts["accepted"] or len(candidate) < gamma + 1:
        raise ValueError("saved proposals cannot be recovered: need a fully accepted first block trace")
    # Target prefill emitted candidate[0]; next gamma tokens were draft proposals.
    return difference, baseline, candidate, candidate[:gamma + 1]


def describe_logits(vector, tokenizer, watched):
    values = vector.detach().float().cpu()
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise ValueError("expected one finite vocabulary logit vector")
    scores, indices = values.topk(min(10, values.numel()))
    # topk does not guarantee argmax's first-index tie policy.
    winner = int(values.argmax())
    baseline_token, candidate_token = watched
    return {
        "original_dtype": str(vector.dtype), "vocabulary_size": values.numel(),
        "argmax_id": winner, "argmax_text": tokenizer.decode([winner]),
        "top_two_gap": float(scores[0] - scores[1]),
        "baseline_token_score": float(values[baseline_token]),
        "speculative_token_score": float(values[candidate_token]),
        "baseline_minus_speculative_score": float(values[baseline_token] - values[candidate_token]),
        "top10": [{"id": int(index), "text": tokenizer.decode([int(index)]), "logit": float(score)}
                  for index, score in zip(indices, scores)],
    }, values


def compare_logits(actual, expected):
    difference = (actual - expected).abs()
    return {"max_abs_logit_difference": float(difference.max()),
            "mean_abs_logit_difference": float(difference.mean()),
            "different_logit_values": int((actual != expected).sum()),
            "argmax_matches": int(actual.argmax()) == int(expected.argmax())}


@torch.inference_mode()
def replay(model, prompt, common, block, row, path):
    cache = DynamicCache(config=model.config)
    if path == "uncached_prefix":
        inputs = torch.cat((prompt, prompt.new_tensor([common])), dim=1)
        output = model(input_ids=inputs, use_cache=False, logits_to_keep=1, return_dict=True)
        return output.logits[0, -1], {"cache": False, "input_tokens": inputs.shape[1]}

    prefill = model(input_ids=prompt, past_key_values=cache, use_cache=True,
                    logits_to_keep=1, return_dict=True)
    metadata = {"prefill_argmax": int(prefill.logits[0, -1].argmax()),
                "cache_length_after_prefill": cache.get_seq_length()}
    if path == "sequential_cached":
        for token in common:
            output = model(input_ids=prompt.new_tensor([[token]]), past_key_values=cache,
                           use_cache=True, logits_to_keep=1, return_dict=True)
        metadata["cache_length_after_forward"] = cache.get_seq_length()
        return output.logits[0, -1], metadata

    forward_tokens = list(block)
    if path == "cached_prefix_block":
        forward_tokens = list(common)
    elif path == "changed_future_block":
        # Preserve length and all causal predecessors. Change only tokens after
        # the row whose logits predict the divergence; correct causal attention
        # must not read these future tokens.
        forward_tokens = list(common) + [0] * (len(block) - len(common))
    keep = prompt.new_tensor([row]) if path == "verification_single_head_row" else 0
    output = model(input_ids=prompt.new_tensor([forward_tokens]), past_key_values=cache,
                   use_cache=True, logits_to_keep=keep, return_dict=True)
    metadata.update({"forward_token_ids": forward_tokens,
                     "cache_length_after_forward": cache.get_seq_length(),
                     "selected_row": row,
                     "lm_head_rows": 1 if path == "verification_single_head_row" else len(forward_tokens)})
    index = 0 if path == "verification_single_head_row" else row
    return output.logits[0, index], metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--case-id", type=int, default=151)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"))
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two repeats are required to check stability")
    saved = json.loads(args.report.read_text())
    cases = [case for case in saved["cases"] if case["id"] == args.case_id]
    if len(cases) != 1:
        parser.error("case ID must identify exactly one saved case")
    case = cases[0]
    try:
        index, baseline, candidate, block = recover_first_block(case, saved["settings"]["gamma"])
    except ValueError as exc:
        parser.error(str(exc))
    device = torch.device(args.device or saved["environment"]["device"])
    dtype = getattr(torch, args.dtype or saved["environment"]["dtype"].split(".")[-1])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output or args.report.with_name(f"speculative_divergence_{args.case_id}_{stamp}.json")
    if destination.exists():
        parser.error("output exists; choose a new filename")
    report = {"status": "running", "source_report": str(args.report.resolve()),
              "source_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
              "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "case_id": args.case_id, "divergent_generated_token_1_based": index + 1,
              "recovered_first_verification_block": block,
              "common_generated_history": baseline[:index],
              "original_baseline_repeat_matches": case.get("baseline_repeat_matches"),
              "original_target_snapshot": saved["target_snapshot"],
              "original_environment": saved["environment"],
              "environment": {"device": str(device), "dtype": str(dtype),
                              "attention": args.attention, "torch": torch.__version__,
                              "transformers": transformers.__version__, "python": platform.python_version()},
              "scope": "Accuracy replay, no performance measurements. Target-only resident, "
                       "unlike original two-model benchmark. No draft calls or cache rollback. "
                       "Future-token perturbation tests causal masking at fixed input shape. "
                       "Exact logits and top-token gaps diagnose ranking, not task quality.",
              "paths": {}}

    def save():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)

    save()
    try:
        tokenizer = AutoTokenizer.from_pretrained(saved["target_snapshot"], local_files_only=True)
        text = case["prompt"] if saved["settings"]["prompt_format"] == "raw" else f"Question: {case['prompt']}\nAnswer:"
        prompt = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        if prompt.shape[1] != case["input_tokens"]:
            raise ValueError("reconstructed prompt length differs from original report")
        report["prompt_token_ids"] = prompt[0].tolist()
        report["watched_tokens"] = [{"id": token, "text": tokenizer.decode([token])}
                                    for token in (baseline[index], candidate[index])]
        print("Watched tokens:", report["watched_tokens"], flush=True)
        print(f"Loading target only on {device} as {dtype}; no generation benchmark", flush=True)
        model = AutoModelForCausalLM.from_pretrained(saved["target_snapshot"], local_files_only=True,
                    dtype=dtype, attn_implementation=args.attention).to(device).eval()
        vectors = {}
        paths = ("sequential_cached", "verification_block", "cached_prefix_block",
                 "changed_future_block", "verification_single_head_row", "uncached_prefix")
        for path in paths:
            report["paths"][path] = []
            vectors[path] = []
            for _ in range(args.repeats):
                vector, metadata = replay(model, prompt, baseline[:index], block, index - 1, path)
                description, cpu_values = describe_logits(vector, tokenizer, (baseline[index], candidate[index]))
                report["paths"][path].append({**metadata, **description})
                vectors[path].append(cpu_values)
                save()
            first = report["paths"][path][0]
            print(f"{path}: winner={first['argmax_text']!r}, "
                  f"top-two gap={first['top_two_gap']:.8f}, "
                  f"baseline-minus-speculative={first['baseline_minus_speculative_score']:.8f}", flush=True)
        report["repeat_comparisons"] = {path: [compare_logits(v, values[0]) for v in values[1:]]
                                         for path, values in vectors.items()}
        report["versus_sequential"] = {path: compare_logits(values[0], vectors["sequential_cached"][0])
                                        for path, values in vectors.items()}
        report["future_token_perturbation"] = compare_logits(vectors["changed_future_block"][0],
                                                              vectors["verification_block"][0])
        winners = {path: report["paths"][path][0]["argmax_id"] for path in paths}
        stable = all(comparison["different_logit_values"] == 0
                     for comparisons in report["repeat_comparisons"].values() for comparison in comparisons)
        reproduced = winners["sequential_cached"] == baseline[index] and winners["verification_block"] == candidate[index]
        report["findings"] = {"original_ranking_flip_reproduced": reproduced,
                              "all_paths_repeat_exactly": stable,
                              "same_shape_future_tokens_change_logits": report["future_token_perturbation"]["different_logit_values"] > 0,
                              "single_head_row_changes_block_winner": winners["verification_single_head_row"] != winners["verification_block"],
                              "uncached_winner_matches_sequential": winners["uncached_prefix"] == winners["sequential_cached"],
                              "note": "A reproduced stable flip with unchanged same-shape future-token logits "
                                      "supports execution-shape numerical sensitivity. It does not prove the exact "
                                      "operator/layer responsible. If perturbation changes logits, investigate causality "
                                      "and repeat noise before attributing it to rounding."}
        report["status"] = "complete"
        print(json.dumps(report["findings"], indent=2), flush=True)
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
        print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
