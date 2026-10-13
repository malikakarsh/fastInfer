"""Full Qwen decode comparison with all single-token RMSNorm calls fused."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch

from qwen_decode_compare import MODEL, REVISION, ROOT, decode_trial, compare_outputs
from qwen_linear_check import PROMPTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    if args.steps < 1 or args.repetitions < 2 or not torch.cuda.is_available():
        parser.error("CUDA, positive steps and at least two repetitions required")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    print(f"Loading {MODEL}; FP16, math SDPA; RMSNorm replacement only", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16,
        attn_implementation="sdpa", output_attentions=False,
        low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False,
    ).to("cuda").eval()
    norms = {name: module for name, module in model.named_modules()
             if module.__class__.__name__ == "Qwen2RMSNorm"}
    if len(norms) != 2 * model.config.num_hidden_layers + 1:
        raise RuntimeError("unexpected normalization module count")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"qwen_rmsnorm_decode_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "qwen_all_rmsnorm_full_decode", "model": MODEL, "revision": REVISION,
        "dtype": "float16", "attention": "sdpa_math", "cache": "DynamicCache",
        "replacement_modules": list(norms), "gemv_replacement": False,
        "rmsnorm": {"num_warps": 4, "enable_fp_fusion": False},
        "steps": args.steps, "repetitions": args.repetitions,
        "warmup_trajectories_per_implementation": 2,
        "allow_fp16_reduced_precision_reduction": True,
        "accuracy": {"rtol": .01, "atol": .01, "identical_greedy_tokens_required": True},
        "timing": "eager_cached_greedy_decode_loop_no_cuda_graph",
        "order": "alternating_baseline_candidate",
        "scope": "All Qwen RMSNorm modules use the frozen kernel on eligible single-token "
                 "decode inputs. Original prefill, independent fresh cache per trial. "
                 "Includes model, LM head, growing cache and argmax; fixed steps after EOS. "
                 "Strict logits/token and baseline-repeat checks before timing. "
                 "Uncompiled original model baseline; no attention or GEMV replacement.",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("decode_rmsnorm.py", "kernels/triton_rmsnorm.py",
                                       "benchmarks/qwen_rmsnorm_decode_compare.py",
                                       "benchmarks/qwen_decode_compare.py")},
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "cases": [],
    }

    def trial(encoded, custom, validate=False):
        return decode_trial(model, encoded, args.steps, custom=custom, validate=validate,
                            replacement="rmsnorm", norms=norms)

    def save():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for prompt in PROMPTS:
            print(f"\nPrompt: {prompt!r}", flush=True)
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
            baseline = trial(encoded, False, True)
            repeat = trial(encoded, False, True)
            candidate = trial(encoded, True, True)
            accuracy = compare_outputs(baseline, candidate)
            baseline_accuracy = compare_outputs(baseline, repeat)
            case = {"prompt": prompt, "accuracy": accuracy,
                    "baseline_repeat_accuracy": baseline_accuracy,
                    "validation_dispatch": candidate["dispatch"]}
            report["cases"].append(case)
            del baseline, repeat, candidate
            calls = case["validation_dispatch"]
            print(f"  RMSNorm calls: {calls['rmsnorm_calls']} | fallback {calls['fallback_calls']} "
                  f"| modules {len(norms)}", flush=True)
            print(f"  Baseline repeat mismatches: {baseline_accuracy['logit_mismatches']}", flush=True)
            print(f"  Logit mismatches: {accuracy['logit_mismatches']} | max error "
                  f"{accuracy['max_abs_logit_error']:.6f} | identical tokens "
                  f"{accuracy['identical_greedy_tokens']}", flush=True)
            if not accuracy["passed"] or not baseline_accuracy["passed"]:
                case["timing_skipped"] = True
                print("  Timing skipped: strict accuracy or baseline-repeat check failed.", flush=True)
                save()
                continue
            for _ in range(2):
                for custom in (False, True):
                    trial(encoded, custom)
            samples = {"pytorch": [], "fused_rmsnorm": []}
            for repetition in range(args.repetitions):
                order = (False, True) if repetition % 2 == 0 else (True, False)
                for custom in order:
                    name = "fused_rmsnorm" if custom else "pytorch"
                    samples[name].append(trial(encoded, custom))
            case["samples"] = samples
            case["medians"] = {
                name: {metric: statistics.median(row[metric] for row in rows)
                       for metric in ("wall_ms_per_step", "cuda_event_ms_per_step")}
                for name, rows in samples.items()
            }
            deltas = [(pt["wall_ms_per_step"] - fused["wall_ms_per_step"]) * 1000
                      for pt, fused in zip(samples["pytorch"], samples["fused_rmsnorm"])]
            case["paired_wall_delta_us_per_step"] = {
                "samples": deltas, "median": statistics.median(deltas),
                "candidate_faster_rounds": sum(delta > 0 for delta in deltas),
                "rounds": args.repetitions,
            }
            case["timed_candidate_dispatch_total"] = {
                field: sum(row["dispatch"][field] for row in samples["fused_rmsnorm"])
                for field in ("rmsnorm_calls", "fallback_calls")
            }
            pt = case["medians"]["pytorch"]["wall_ms_per_step"]
            fused = case["medians"]["fused_rmsnorm"]["wall_ms_per_step"]
            case["wall_speedup"] = pt / fused
            paired = case["paired_wall_delta_us_per_step"]
            print(f"  Full decode: PyTorch {pt:.3f} ms/step | fused RMSNorm {fused:.3f} ms/step "
                  f"| PT/custom {pt/fused:.3f}x", flush=True)
            print(f"  Paired median saving {paired['median']:.2f} us/step | custom faster "
                  f"{paired['candidate_faster_rounds']}/{args.repetitions}", flush=True)
            print(f"  Timed dispatch: {case['timed_candidate_dispatch_total']}", flush=True)
            save()
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise
    finally:
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
