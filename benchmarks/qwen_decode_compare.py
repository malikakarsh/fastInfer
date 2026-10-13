"""Compare full cached Qwen decode with one opt-in GEMV projection."""

import argparse
from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from fastinfer.decode_linear import use_decode_gemv
from qwen_linear_check import MODEL, PROMPTS, require_finite

ROOT = Path(__file__).resolve().parents[1]
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"


def decode_trial(model, encoded, steps, *, custom, validate=False, capture_projection=False):
    """Fresh prefill/cache per trial; time only the cached greedy decode loop.

    Fixed length, including tokens after EOS, so paired workloads have equal
    lengths. CUDA-event elapsed time includes host submission gaps; wall time
    includes Python dispatch, cache growth, argmax and final synchronization.
    """
    from transformers import DynamicCache

    projection = model.model.layers[0].mlp.up_proj
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        # Prefill always uses the original projection, even for a one-token prompt.
        prefill = model(**encoded, past_key_values=DynamicCache(),
                        use_cache=True, return_dict=True)
        require_finite(prefill.logits, "prefill.logits")
        cache = prefill.past_key_values
        token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        initial_token = token
        mask = encoded["attention_mask"]
        del prefill
        logits, tokens, projections = [], [], []
        context = use_decode_gemv(projection) if custom else nullcontext(None)
        with context as counts, ExitStack() as cleanup:
            if capture_projection:
                def capture(module, inputs, output):
                    projections.append((inputs[0].detach().clone(), output.detach().clone()))
                handle = projection.register_forward_hook(capture)
                cleanup.callback(handle.remove)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            start.record()
            for _ in range(steps):
                mask = torch.cat((mask, torch.ones_like(token)), dim=1)
                output = model(input_ids=token, attention_mask=mask,
                               past_key_values=cache, use_cache=True, return_dict=True)
                cache = output.past_key_values
                token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if validate:
                    require_finite(output.logits, "decode.logits")
                    logits.append(output.logits[:, -1, :].clone())
                    tokens.append(token.clone())
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000
            event_ms = start.elapsed_time(end)
        if custom and counts != {"gemv_calls": steps, "fallback_calls": 0}:
            raise RuntimeError(f"Expected one GEMV dispatch per decode step: {counts}")
        result = {"wall_ms_per_step": wall_ms / steps,
                  "cuda_event_ms_per_step": event_ms / steps,
                  "dispatch": counts}
        if validate:
            result["logits"] = torch.stack(logits).float()
            result["tokens"] = torch.cat([initial_token, *tokens], dim=1).cpu().tolist()
        if capture_projection:
            result["projections"] = projections
        return result


def compare_outputs(baseline, candidate):
    """Same greedy trajectory plus elementwise logits tolerance are required."""
    a, b = baseline["logits"], candidate["logits"]
    require_finite(a, "baseline.logits")
    require_finite(b, "candidate.logits")
    error = (a - b).abs()
    mismatches = int((error > .01 + .01 * a.abs()).sum().item())
    same_tokens = baseline["tokens"] == candidate["tokens"]
    per_step = []
    for index in range(a.shape[0]):
        delta = error[index]
        mismatch = delta > .01 + .01 * a[index].abs()
        # Probability differences supplement the unchanged raw-logit gate.
        pa, pb = a[index].softmax(dim=-1), b[index].softmax(dim=-1)
        per_step.append({
            "step": index + 1,
            "mismatches": int(mismatch.sum().item()),
            "max_abs_error": delta.max().item(),
            "mean_abs_error": delta.mean().item(),
            "probability_total_variation_max": (.5 * (pa - pb).abs().sum(dim=-1)).max().item(),
        })
    return {"logit_mismatches": mismatches, "logit_elements": a.numel(),
            "max_abs_logit_error": error.max().item(),
            "mean_abs_logit_error": error.mean().item(),
            "mismatch_fraction": mismatches / a.numel(),
            "per_step": per_step,
            "identical_greedy_tokens": same_tokens,
            "baseline_token_ids": baseline["tokens"],
            "candidate_token_ids": candidate["tokens"],
            "passed": same_tokens and mismatches == 0}


def projection_diagnostics(candidate, weight):
    """Compare GEMV with PyTorch on the same actual input at every step."""
    import torch.nn.functional as F

    rows = []
    with torch.inference_mode():
        weight32 = weight.float()
        for index, (x, actual) in enumerate(candidate["projections"]):
            pytorch = F.linear(x, weight)
            reference = F.linear(x.float(), weight32)
            require_finite(reference, "projection_fp32_reference")
            values = {}
            for name, output in (("pytorch", pytorch), ("gemv", actual)):
                require_finite(output, name + ".projection")
                error = (output.float() - reference).abs()
                values[name] = {
                    "fp32_mismatches": int((error > .01 + .01 * reference.abs()).sum().item()),
                    "max_abs_fp32_error": error.max().item(),
                }
            difference = (actual.float() - pytorch.float()).abs()
            rows.append({"step": index + 1, "accuracy": values,
                         "pytorch_gemv_max_abs_difference": difference.max().item(),
                         "pytorch_gemv_different_elements": int((difference != 0).sum().item())})
    return rows


def timing_decision(case, *, diagnose, exploratory, steps):
    """Keep strict accuracy separate from explicitly requested exploratory timing."""
    if diagnose:
        return False, "diagnostic_only"
    if case["accuracy"]["passed"]:
        return True, "strict_accuracy_passed"
    if not exploratory:
        return False, "strict_accuracy_failed"
    if not case["accuracy"]["identical_greedy_tokens"]:
        return False, "greedy_tokens_differ"
    if not case.get("baseline_repeat_accuracy", {}).get("passed", False):
        return False, "baseline_repeat_failed"
    projections = case.get("projection_diagnostics", [])
    if len(projections) != steps or any(
        row["accuracy"][name]["fp32_mismatches"] != 0
        for row in projections for name in ("pytorch", "gemv")
    ):
        return False, "projection_accuracy_failed_or_missing"
    return True, "exploratory_strict_logits_failed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--diagnose", action="store_true",
                        help="compare a baseline repeat and per-step projections; skip timing")
    mode.add_argument("--exploratory-timing", action="store_true",
                      help="allow labeled timing despite failed strict logits, only after "
                           "matching tokens, baseline repeat and projection checks")
    args = parser.parse_args()
    inspect_projection = args.diagnose or args.exploratory_timing
    if args.steps < 1 or args.repetitions < 2:
        parser.error("steps >= 1 and repetitions >= 2 required")
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    print(f"Loading {MODEL} revision {REVISION}; FP16, math SDPA", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16,
        attn_implementation="sdpa", output_attentions=False,
        low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False,
    ).to("cuda").eval()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output or ROOT / "results/raw" / f"qwen_decode_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "qwen_one_projection_full_decode",
        "model": MODEL, "revision": REVISION, "dtype": "float16",
        "attention": "sdpa_math_fp32_intermediates", "cache": "DynamicCache",
        "projection": "model.layers.0.mlp.up_proj",
        "gemv": {"block_rows": 1, "num_warps": 4},
        "allow_fp16_reduced_precision_reduction": True,
        "diagnostic_mode": args.diagnose,
        "exploratory_timing_requested": args.exploratory_timing,
        "steps": args.steps, "repetitions": args.repetitions,
        "warmup_trajectories_per_implementation": 2,
        "order": "alternating_baseline_candidate_per_round",
        "timing": "eager_cached_greedy_decode_loop_no_cuda_graph",
        "scope": "Full model decode, including LM head, argmax and growing cache; "
                 "fresh untimed prefill before every trial; fixed steps regardless of EOS. "
                 "Only layer 0 up_proj replaced during decode. Not full request latency.",
        "accuracy": "candidate vs baseline logits, rtol=atol=0.01; identical greedy tokens",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("decode_linear.py", "kernels/triton_gemv.py",
                                       "benchmarks/qwen_decode_compare.py", "benchmarks/qwen_linear_check.py")},
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "cases": [],
    }

    def save():
        destination.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for prompt in PROMPTS:
            print(f"\nPrompt: {prompt!r}", flush=True)
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
            baseline = decode_trial(model, encoded, args.steps, custom=False, validate=True)
            candidate = decode_trial(model, encoded, args.steps, custom=True, validate=True,
                                     capture_projection=inspect_projection)
            accuracy = compare_outputs(baseline, candidate)
            case = {"prompt": prompt, "accuracy": accuracy,
                    "validation_dispatch": candidate["dispatch"]}
            report["cases"].append(case)
            dispatch = case["validation_dispatch"]
            print(f"  Candidate decode calls: GEMV {dispatch['gemv_calls']} | "
                  f"original fallback {dispatch['fallback_calls']}", flush=True)
            if inspect_projection:
                repeat = decode_trial(model, encoded, args.steps, custom=False, validate=True)
                case["baseline_repeat_accuracy"] = compare_outputs(baseline, repeat)
                case["projection_diagnostics"] = projection_diagnostics(
                    candidate, model.model.layers[0].mlp.up_proj.weight
                )
                repeat_accuracy = case["baseline_repeat_accuracy"]
                print(f"  Baseline repeat: {repeat_accuracy['logit_mismatches']} mismatches | "
                      f"max error {repeat_accuracy['max_abs_logit_error']:.6f}", flush=True)
                for row, projection in zip(accuracy["per_step"], case["projection_diagnostics"]):
                    print(f"  Step {row['step']:2}: logits max/mean error "
                          f"{row['max_abs_error']:.6f}/{row['mean_abs_error']:.6f} | "
                          f"mismatches {row['mismatches']} | "
                          f"projection PT/GEMV FP32 mismatches "
                          f"{projection['accuracy']['pytorch']['fp32_mismatches']}/"
                          f"{projection['accuracy']['gemv']['fp32_mismatches']}", flush=True)
                del repeat
            del baseline, candidate
            save()
            print(f"  Logit mismatches: {accuracy['logit_mismatches']} | "
                  f"identical tokens: {accuracy['identical_greedy_tokens']}", flush=True)
            allowed, status = timing_decision(
                case, diagnose=args.diagnose, exploratory=args.exploratory_timing,
                steps=args.steps,
            )
            case["timing_status"] = status
            print(f"  Timing status: {status}", flush=True)
            if not allowed:
                case["timing_skipped"] = True
                save()
                continue
            for _ in range(2):
                for custom in (False, True):
                    decode_trial(model, encoded, args.steps, custom=custom)
            samples = {"pytorch": [], "one_gemv": []}
            for repetition in range(args.repetitions):
                order = (False, True) if repetition % 2 == 0 else (True, False)
                for custom in order:
                    name = "one_gemv" if custom else "pytorch"
                    samples[name].append(decode_trial(model, encoded, args.steps, custom=custom))
            case["samples"] = samples
            case["timed_candidate_dispatch_total"] = {
                field: sum(row["dispatch"][field] for row in samples["one_gemv"])
                for field in ("gemv_calls", "fallback_calls")
            }
            deltas_us = [
                (pt["wall_ms_per_step"] - custom["wall_ms_per_step"]) * 1000
                for pt, custom in zip(samples["pytorch"], samples["one_gemv"])
            ]
            case["paired_wall_delta_us_per_step"] = {
                "samples": deltas_us,
                "median": statistics.median(deltas_us),
                "min": min(deltas_us), "max": max(deltas_us),
                "candidate_faster_rounds": sum(delta > 0 for delta in deltas_us),
                "rounds": args.repetitions,
            }
            case["medians"] = {
                name: {metric: statistics.median(row[metric] for row in rows)
                       for metric in ("wall_ms_per_step", "cuda_event_ms_per_step")}
                for name, rows in samples.items()
            }
            pt = case["medians"]["pytorch"]["wall_ms_per_step"]
            tt = case["medians"]["one_gemv"]["wall_ms_per_step"]
            case["wall_speedup"] = pt / tt
            print(f"  Full decode wall time: PyTorch {pt:.3f} ms/step | "
                  f"one GEMV {tt:.3f} ms/step | PT/custom {pt / tt:.3f}x", flush=True)
            paired = case["paired_wall_delta_us_per_step"]
            calls = case["timed_candidate_dispatch_total"]
            print(f"  Paired median saving {paired['median']:.2f} us/step | "
                  f"custom faster in {paired['candidate_faster_rounds']}/{args.repetitions} rounds", flush=True)
            print(f"  Timed candidate calls: GEMV {calls['gemv_calls']} | "
                  f"original fallback {calls['fallback_calls']}", flush=True)
            save()
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise
    finally:
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
