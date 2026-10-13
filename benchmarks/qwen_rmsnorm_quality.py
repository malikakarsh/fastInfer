"""Broader numerical-behavior check for the scoped Qwen RMSNorm replacement."""

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from fastinfer.decode_rmsnorm import use_decode_rmsnorm
from qwen_decode_compare import MODEL, REVISION, ROOT
from qwen_linear_check import require_finite

# Fixed raw-text continuations for the base model, not a scored task benchmark.
PROMPT_CASES = (
    ("original_fact", "The capital of France is"),
    ("original_explanation", "Explain why the sky appears blue in one sentence."),
    ("original_code", "def add(a, b):\n    return"),
    ("arithmetic", "Question: What is 17 multiplied by 23?\nAnswer:"),
    ("arithmetic_steps", "Calculate (12 + 8) * 5 and explain each step.\n"),
    ("python", "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n"),
    ("code_explanation", "The Python expression [x * x for x in range(5)] evaluates to"),
    ("science", "Photosynthesis converts light energy into chemical energy. The main steps are"),
    ("story", "Maya opened the old wooden box and found a map. At its center was"),
    ("dialogue", "Alex: How do I organize a busy week?\nSam:"),
    ("translation", "English: The small cat sleeps on the chair.\nFrench:"),
    ("few_shot", "red -> color\nblue -> color\ncat -> animal\ndog ->"),
    ("structured_text", 'A JSON object containing name "Ravi", age 24, and city "Delhi":\n'),
    ("summary", "A library extended its opening hours, added study rooms, and hired two librarians. "
                "Visitors increased, but the budget remained unchanged.\nSummary:"),
    ("long_context", "A garden has four sections. North: tomatoes and carrots. "
                     "East: beans and peas. South: roses and lavender. West: apples and pears. "
                     "The north section gets morning sunlight. The east section needs daily watering. "
                     "The south section attracts bees. The west section has the oldest trees. "
                     "Each section has a bench and a path.\nThe fruits grow in the"),
    ("repeated_context", "The station opens at six. Trains leave every twenty minutes. " * 12
                         + "\nA concise description of the train schedule:"),
)


def first_divergence(a, b):
    """Return the first differing token position, using one-based positions."""
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index + 1
    return min(len(a), len(b)) + 1 if len(a) != len(b) else None


def probability_comparison(baseline, candidate):
    """Compare logits only when evaluated on the same token histories."""
    a, b = baseline["scores"].float(), candidate["scores"].float()
    if a.shape != b.shape or baseline["tokens"] != candidate["tokens"]:
        raise ValueError("equal score shapes and identical fed token histories required")
    require_finite(a, "baseline.scores")
    require_finite(b, "candidate.scores")
    error = (a - b).abs()
    mismatches = error > .01 + .01 * a.abs()
    pa, pb = a.softmax(dim=-1), b.softmax(dim=-1)
    tv = .5 * (pa - pb).abs().sum(dim=-1)
    baseline_choices = a.argmax(dim=-1)
    candidate_choices = b.argmax(dim=-1)
    top_two = a.topk(2, dim=-1).values
    per_token = []
    for index in range(a.shape[0]):
        per_token.append({
            "position": index + 1,
            "logit_mismatches": int(mismatches[index].sum().item()),
            "max_abs_logit_error": error[index].max().item(),
            "probability_total_variation": tv[index].item(),
            "baseline_argmax": baseline_choices[index].item(),
            "candidate_argmax": candidate_choices[index].item(),
            "baseline_top_two_score_gap": (top_two[index, 0] - top_two[index, 1]).item(),
        })
    return {"positions": a.shape[0], "strict_logit_mismatches": int(mismatches.sum().item()),
            "strict_logits_passed": not mismatches.any().item(),
            "logits_exactly_equal": torch.equal(a, b),
            "max_abs_logit_error": error.max().item(),
            "worst_probability_total_variation": tv.max().item(),
            "mean_probability_total_variation": tv.mean().item(),
            "shared_history_argmax_agreement": int((baseline_choices == candidate_choices).sum().item()),
            "per_token": per_token}


def rollout(model, encoded, norms, limit, eos_ids, *, custom, teacher_tokens=None,
            normalization_kernel=None):
    """Fresh original prefill, then greedy or baseline-token-fed cached decoding."""
    from transformers import DynamicCache

    if teacher_tokens is not None and not teacher_tokens:
        raise ValueError("teacher token history must be nonempty")
    limit = len(teacher_tokens) if teacher_tokens is not None else limit
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        prefill = model(**encoded, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
        cache = prefill.past_key_values
        next_scores = prefill.logits[:, -1, :]
        mask = encoded["attention_mask"]
        del prefill
        tokens, predictions, scores = [], [], []
        decode_calls = 0
        if not custom:
            context = nullcontext(None)
        elif normalization_kernel is None:
            context = use_decode_rmsnorm(norms)
        else:
            context = use_decode_rmsnorm(norms, kernel=normalization_kernel)
        with context as counts:
            for position in range(limit):
                require_finite(next_scores, "generation.scores")
                scores.append(next_scores[0].detach().float().cpu().clone())
                predicted = int(next_scores.argmax(dim=-1).item())
                chosen = predicted if teacher_tokens is None else teacher_tokens[position]
                predictions.append(predicted)
                tokens.append(chosen)
                if chosen in eos_ids or position + 1 == limit:
                    break
                token = torch.tensor([[chosen]], device=encoded["input_ids"].device, dtype=torch.long)
                mask = torch.cat((mask, torch.ones_like(token)), dim=1)
                output = model(input_ids=token, attention_mask=mask, past_key_values=cache,
                               use_cache=True, return_dict=True)
                cache = output.past_key_values
                next_scores = output.logits[:, -1, :]
                decode_calls += 1
                del output
        if custom:
            if (counts["rmsnorm_calls"] != decode_calls * len(norms) or counts["fallback_calls"] != 0
                    or any(row != {"rmsnorm_calls": decode_calls, "fallback_calls": 0}
                           for row in counts["modules"].values())):
                raise RuntimeError(f"unexpected normalization dispatch: {counts}")
        return {"tokens": tokens, "predictions": predictions, "scores": torch.stack(scores),
                "terminated_on_eos": tokens[-1] in eos_ids, "decode_calls": decode_calls,
                "dispatch": counts}


def archive_evidence(destination, *, kernel="current"):
    """Preserve available raw timing files and this check in a downloadable ZIP."""
    originals = [ROOT / "results/raw" / name for name in (
        "qwen_rmsnorm_decode_20261013T065623423837Z.json",
        "qwen_rmsnorm_decode_20261013T070551021237Z.json",
    )]
    if kernel == "mean_candidate":
        originals = []
        kernel_source = "kernels/rmsnorm_mean_candidate.py"
        expected_hash = json.loads(destination.read_text()).get("source_sha256", {}).get(kernel_source)
        for path in sorted((ROOT / "results/raw").glob("qwen_rmsnorm_decode_mean_candidate_*.json")):
            timing = json.loads(path.read_text())
            cases = timing.get("cases", [])
            if (not timing.get("failure") and timing.get("rmsnorm", {}).get("kernel") == kernel
                    and (expected_hash is None or timing.get("source_sha256", {}).get(kernel_source) == expected_hash)
                    and len(cases) == 3
                    and all(case.get("timing_status") == "strict_accuracy_passed"
                            and case.get("samples") for case in cases)):
                originals.append(path)
        originals = originals[-2:]
    archive = destination.with_suffix(".evidence.zip")
    missing = [str(path) for path in originals if not path.exists()]
    if kernel == "mean_candidate" and len(originals) < 2:
        missing.append(f"Expected two completed mean_candidate timing runs; found {len(originals)}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in [destination, *originals]:
            if path.exists():
                bundle.write(path, arcname=path.name)
        bundle.writestr("archive_manifest.json", json.dumps({"kernel": kernel,
                         "included_timing_files": [path.name for path in originals if path.exists()],
                         "missing_timing_files": missing}, indent=2))
    return archive, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--archive", action="store_true", help="also ZIP this result and available raw timing evidence")
    parser.add_argument("--kernel", choices=("current", "mean_candidate"), default="current")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or not torch.cuda.is_available():
        parser.error("positive max-new-tokens and CUDA required")
    selected_kernel = None
    kernel_path = "kernels/triton_rmsnorm.py"
    if args.kernel == "mean_candidate":
        if torch.__version__.split("+")[0] != "2.5.1":
            parser.error("mean_candidate targets pinned PyTorch 2.5.1")
        from fastinfer.kernels.rmsnorm_mean_candidate import rmsnorm_candidate
        selected_kernel = rmsnorm_candidate
        kernel_path = "kernels/rmsnorm_mean_candidate.py"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    print(f"Loading {MODEL}; FP16, math SDPA; numerical behavior check; kernel={args.kernel}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16,
        attn_implementation="sdpa", output_attentions=False,
        low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False,
    ).to("cuda").eval()
    norms = {name: module for name, module in model.named_modules()
             if module.__class__.__name__ == "Qwen2RMSNorm"}
    if len(norms) != 2 * model.config.num_hidden_layers + 1:
        raise RuntimeError("unexpected normalization count")
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("no model/tokenizer EOS ID available")
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    suffix = "mean_candidate_" if args.kernel == "mean_candidate" else ""
    destination = ROOT / "results/raw" / f"qwen_rmsnorm_quality_{suffix}{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "qwen_rmsnorm_broader_numerical_behavior", "model": MODEL, "revision": REVISION,
        "dtype": "float16", "attention": "sdpa_math", "cache": "DynamicCache",
        "max_new_tokens": args.max_new_tokens, "eos_ids": sorted(eos_ids),
        "prompt_count": len(PROMPT_CASES), "rmsnorm_num_warps": 4, "normalization_count": len(norms),
        "kernel": args.kernel, "kernel_source": kernel_path,
        "strict_logits_tolerance": {"rtol": .01, "atol": .01},
        "allow_fp16_reduced_precision_reduction": True,
        "scope": "Fixed raw-text continuations on the base model, not a scored task-quality "
                 "benchmark or sampled generation. Original prefill; normalization replacement "
                 "during decode only. Compare independent greedy continuations and aligned "
                 "probabilities on the baseline's token history. No latency measurement, "
                 "no relaxation of the existing logits criterion or probability acceptance threshold.",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("benchmarks/qwen_rmsnorm_quality.py", "decode_rmsnorm.py",
                                       kernel_path)},
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "cases": [],
    }

    def save():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for index, (category, prompt) in enumerate(PROMPT_CASES):
            print(f"\n[{index+1}/{len(PROMPT_CASES)}] {category}", flush=True)
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
            original = rollout(model, encoded, norms, args.max_new_tokens, eos_ids, custom=False)
            repeat = rollout(model, encoded, norms, args.max_new_tokens, eos_ids, custom=False)
            candidate = rollout(model, encoded, norms, args.max_new_tokens, eos_ids, custom=True,
                                normalization_kernel=selected_kernel)
            baseline_tokens_match = original["tokens"] == repeat["tokens"]
            baseline_scores_match = (original["scores"].shape == repeat["scores"].shape
                                     and torch.equal(original["scores"], repeat["scores"]))
            divergence = first_divergence(original["tokens"], candidate["tokens"])
            case = {"category": category, "prompt": prompt,
                    "baseline_repeat_tokens_identical": baseline_tokens_match,
                    "baseline_repeat_logits_identical": baseline_scores_match,
                    "greedy_tokens_identical": divergence is None,
                    "first_divergence_token_position": divergence,
                    "baseline_token_ids": original["tokens"], "candidate_token_ids": candidate["tokens"],
                    "baseline_text": tokenizer.decode(original["tokens"], skip_special_tokens=False),
                    "candidate_text": tokenizer.decode(candidate["tokens"], skip_special_tokens=False),
                    "baseline_stopped_on_eos": original["terminated_on_eos"],
                    "candidate_stopped_on_eos": candidate["terminated_on_eos"],
                    "candidate_decode_calls": candidate["decode_calls"],
                    "candidate_dispatch": candidate["dispatch"]}
            report["cases"].append(case)
            if baseline_tokens_match and baseline_scores_match:
                aligned = candidate
                if divergence is not None:
                    aligned = rollout(model, encoded, norms, args.max_new_tokens, eos_ids,
                                      custom=True, teacher_tokens=original["tokens"],
                                      normalization_kernel=selected_kernel)
                case["baseline_history_replay_required"] = divergence is not None
                case["aligned_probability_comparison"] = probability_comparison(original, aligned)
                if divergence is not None:
                    case["aligned_replay_dispatch"] = aligned["dispatch"]
                comparison = case["aligned_probability_comparison"]
                print(f"  Generated tokens: original {len(original['tokens'])}, custom {len(candidate['tokens'])} | "
                      f"identical {divergence is None} | first divergence {divergence}", flush=True)
                print(f"  Shared-history worst TV {comparison['worst_probability_total_variation']:.6f} | "
                      f"strict logit mismatches {comparison['strict_logit_mismatches']} | "
                      f"exact logits {comparison['logits_exactly_equal']}", flush=True)
                del aligned
            else:
                case["comparison_skipped"] = "baseline_repeat_unstable"
                print("  Baseline repeat unstable; numerical comparison skipped.", flush=True)
            del original, repeat, candidate
            save()
        usable = [case for case in report["cases"] if "aligned_probability_comparison" in case]
        report["summary"] = {
            "cases": len(report["cases"]), "baseline_stable_cases": len(usable),
            "identical_greedy_cases": sum(case["greedy_tokens_identical"] for case in usable),
            "divergent_greedy_cases": sum(not case["greedy_tokens_identical"] for case in usable),
            "custom_decode_exercised_cases": sum(case["candidate_decode_calls"] > 0 for case in usable),
            "identical_greedy_cases_with_custom_decode": sum(
                case["greedy_tokens_identical"] and case["candidate_decode_calls"] > 0 for case in usable),
            "strict_logits_passed_cases": sum(case["aligned_probability_comparison"]["strict_logits_passed"]
                                                for case in usable),
            "exact_logits_cases": sum(case["aligned_probability_comparison"]["logits_exactly_equal"]
                                      for case in usable),
            "worst_aligned_probability_total_variation": max(
                (case["aligned_probability_comparison"]["worst_probability_total_variation"] for case in usable),
                default=None),
        }
        save()
        print(f"\nSummary: {json.dumps(report['summary'])}", flush=True)
        if args.archive:
            archive, missing = archive_evidence(destination, kernel=args.kernel)
            print(f"Evidence ZIP: {archive}", flush=True)
            if missing:
                print(f"Timing files unavailable for archive: {missing}", flush=True)
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise
    finally:
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
