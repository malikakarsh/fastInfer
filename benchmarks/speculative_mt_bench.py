"""Paired ordinary/speculative greedy timing on first-turn MT-Bench prompts.

Local cached, unquantized Hugging Face Qwen2 checkpoints only. No downloads,
chat template, compilation, or custom Triton kernels. Both models stay resident.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from fastinfer.speculative_cached import cached_decode


def load_prompts(path, limit=0):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = set()
    prompts = []
    for row in rows:
        identifier = row.get("question_id", row.get("prompt_id"))
        turns = row.get("prompt", row.get("turns"))
        if (identifier is None or identifier in ids or not isinstance(turns, list)
                or not turns or not isinstance(turns[0], str) or not turns[0].strip()
                or not isinstance(row.get("category"), str)):
            raise ValueError("expected unique MT-Bench IDs, categories and nonempty prompt lists")
        ids.add(identifier)
        prompts.append({"id": identifier, "category": row["category"], "prompt": turns[0]})
    if not prompts:
        raise ValueError("dataset contains no prompts")
    # Small pilots rotate categories rather than selecting writing alone.
    groups = defaultdict(list)
    for prompt in prompts:
        groups[prompt["category"]].append(prompt)
    balanced = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                balanced.append(group.pop(0))
    return balanced[:limit] if limit else balanced


def resolve_model(identifier):
    path = Path(identifier).expanduser()
    if path.is_dir():
        return path.resolve()
    return Path(snapshot_download(identifier, local_files_only=True)).resolve()


def summarize(cases):
    valid = [case for case in cases if case.get("correctness_passed") and case.get("samples")]
    result = {"cases": len(cases), "timed_cases": len(valid),
              "failed_cases": [case["id"] for case in cases if not case.get("correctness_passed")]}
    if not valid:
        return result
    baseline = sum(statistics.median(s["baseline"]["total_s"] for s in c["samples"]) for c in valid)
    result["sum_baseline_prompt_medians_s"] = baseline
    if "speculative" in valid[0]["samples"][0]:
        candidate = sum(statistics.median(s["speculative"]["total_s"] for s in c["samples"]) for c in valid)
        result["sum_speculative_prompt_medians_s"] = candidate
        result["aggregate_speedup"] = baseline / candidate
        ratios = [statistics.median(s["baseline"]["total_s"] for s in c["samples"]) /
                  statistics.median(s["speculative"]["total_s"] for s in c["samples"]) for c in valid]
        result["prompts_faster"] = sum(ratio > 1 for ratio in ratios)
        result["median_prompt_speedup"] = statistics.median(ratios)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--draft", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--mode", choices=("baseline", "compare"), default="compare")
    parser.add_argument("--data", type=Path, default=ROOT / "benchmarks/data/mt_bench/raw/question.jsonl")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 means all 80; pilots alternate categories")
    parser.add_argument("--prompt-format", choices=("qa", "raw"), default="qa")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.max_new_tokens, args.gamma, args.pairs) < 1 or args.limit < 0:
        parser.error("token count, gamma and pairs must be positive; limit must be nonnegative")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_name)
    dtype = getattr(torch, args.dtype)
    prompts = load_prompts(args.data, args.limit)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output or ROOT / "results/raw" / f"speculative_mt_bench_{stamp}.json"
    if destination.exists():
        parser.error(f"output exists: {destination}; choose a new filename")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    report = {"status": "running", "created_utc": stamp,
              "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                              "python": platform.python_version(), "platform": platform.platform(),
                              "device": str(device), "dtype": str(dtype), "cpu_threads": torch.get_num_threads()},
              "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in ("speculative_cached.py", "speculative.py",
                                              "benchmarks/speculative_mt_bench.py")},
              "category_counts": dict(Counter(p["category"] for p in prompts)),
              "scope": "First-turn MT-Bench latency workload, not official quality scoring. "
                       "Greedy, batch one, EOS enabled, fresh DynamicCaches, both models resident. "
                       "Total includes prefill and speculative draft work; loading/tokenization excluded. "
                       "Decode timings exclude first-token prefill, which includes draft prefill on candidate. "
                       "No quantization, sampling, compilation, custom kernels or prompt truncation.", "cases": []}

    def save():
        report["summary"] = summarize(report["cases"])
        report["by_category"] = {category: summarize([c for c in report["cases"] if c["category"] == category])
                                 for category in report["category_counts"]}
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(destination)

    save()
    try:
        target_path = resolve_model(args.target)
        tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
        report["target_snapshot"] = str(target_path)
        print(f"Loading target {target_path.name} on {device} as {dtype}", flush=True)
        target = AutoModelForCausalLM.from_pretrained(target_path, local_files_only=True,
                    dtype=dtype, attn_implementation="sdpa").to(device).eval()
        if getattr(target.config, "quantization_config", None):
            raise ValueError("this first experiment requires an unquantized target")
        draft = None
        if args.mode == "compare":
            draft_path = resolve_model(args.draft)
            draft_tokenizer = AutoTokenizer.from_pretrained(draft_path, local_files_only=True)
            if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
                raise ValueError("draft/target token-to-ID mappings differ")
            report["draft_snapshot"] = str(draft_path)
            print(f"Loading draft {draft_path.name}", flush=True)
            draft = AutoModelForCausalLM.from_pretrained(draft_path, local_files_only=True,
                        dtype=dtype, attn_implementation="sdpa").to(device).eval()
        eos = target.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        report["eos_token_ids"] = eos
        operations = {"baseline": lambda ids: cached_decode(target, ids,
                            max_new_tokens=args.max_new_tokens, eos_token_ids=eos)}
        if draft is not None:
            operations["speculative"] = lambda ids: cached_decode(target, ids, draft=draft,
                            max_new_tokens=args.max_new_tokens, eos_token_ids=eos, gamma=args.gamma)
        for prompt in prompts:
            text = prompt["prompt"] if args.prompt_format == "raw" else f"Question: {prompt['prompt']}\nAnswer:"
            ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            case = {**prompt, "input_tokens": ids.shape[1], "samples": []}
            report["cases"].append(case)
            # Independent repeat validates stability and warms each path, outside timed pairs.
            checked = {name: operation(ids)[0] for name, operation in operations.items()}
            repeat = operations["baseline"](ids)[0]
            expected = checked["baseline"].generated
            case["validation"] = {name: asdict(output) for name, output in checked.items()}
            case["baseline_repeat_matches"] = repeat.generated == expected
            case["correctness_passed"] = case["baseline_repeat_matches"] and all(
                output.generated == expected for output in checked.values())
            case["generated_text"] = tokenizer.decode(expected, skip_special_tokens=False)
            if not case["correctness_passed"]:
                print(f"{prompt['id']} {prompt['category']}: token validation FAILED; no timing", flush=True)
                save()
                continue
            for pair in range(args.pairs):
                names = list(operations)
                if pair % 2:
                    names.reverse()
                sample = {"order": names}
                for name in names:
                    output, timings = operations[name](ids)
                    sample[name] = {**timings, **asdict(output)}
                    if output.generated != expected:
                        case["correctness_passed"] = False
                case["samples"].append(sample)
                save()
                if not case["correctness_passed"]:
                    break
            case["generated_tokens"] = len(expected)
            if draft is not None and case["correctness_passed"]:
                case["paired_total_savings_s"] = [s["baseline"]["total_s"] - s["speculative"]["total_s"]
                                                   for s in case["samples"]]
                case["speculative_faster_pairs"] = sum(saving > 0 for saving in case["paired_total_savings_s"])
                case["total_speedup"] = statistics.median(s["baseline"]["total_s"] for s in case["samples"]) / statistics.median(
                    s["speculative"]["total_s"] for s in case["samples"])
                baseline_decode = statistics.median(s["baseline"]["decode_s"] for s in case["samples"])
                speculative_decode = statistics.median(s["speculative"]["decode_s"] for s in case["samples"])
                case["decode_speedup"] = baseline_decode / speculative_decode if len(expected) > 1 and speculative_decode > 0 else None
                proposed = sum(s["speculative"]["proposed"] for s in case["samples"])
                case["acceptance_rate"] = sum(s["speculative"]["accepted"] for s in case["samples"]) / proposed if proposed else None
                print(f"{prompt['id']} {prompt['category']}: {len(expected)} tokens, "
                      f"speedup {case['total_speedup']:.3f}x, acceptance {case['acceptance_rate']}", flush=True)
            else:
                print(f"{prompt['id']} {prompt['category']}: {len(expected)} tokens, baseline recorded", flush=True)
            save()
        report["status"] = "complete" if all(c["correctness_passed"] for c in report["cases"]) else "correctness_failed"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
        print(f"Saved {destination}", flush=True)
    if report["status"] != "complete":
        raise SystemExit("Token correctness failed; no accepted speedup claim.")


if __name__ == "__main__":
    main()
