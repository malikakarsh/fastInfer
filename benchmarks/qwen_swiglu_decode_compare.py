"""Compare eager Qwen with only its decode SwiGLU stage compiled or custom."""

import argparse
from contextlib import nullcontext
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

from fastinfer.decode_swiglu import use_decode_swiglu
from qwen_decode_compare import MODEL, REVISION, ROOT, compare_outputs
from qwen_linear_check import PROMPTS, require_finite
from softmax_compare import COMPILED_SETTINGS, measurement_orders, native_triton_comparison


def decode_trial(model, encoded, mlps, kernel, steps, *, validate=False):
    """Original fresh-cache prefill; optional stage replacement only during decode."""
    from transformers import DynamicCache

    captured = []
    def capture(name, gate, up, product):
        captured.append((name, gate.detach().clone(), up.detach().clone(), product.detach().clone()))
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        prefill = model(**encoded, past_key_values=DynamicCache(), use_cache=True, return_dict=True)
        require_finite(prefill.logits, "prefill.logits")
        token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        initial_token = token
        cache, mask = prefill.past_key_values, encoded["attention_mask"]
        del prefill
        context = (nullcontext(None) if kernel is None else
                   use_decode_swiglu(mlps, kernel=kernel, capture=capture if validate else None))
        logits, tokens = [], []
        with context as counts:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            start.record()
            for _ in range(steps):
                mask = torch.cat((mask, torch.ones_like(token)), dim=1)
                output = model(input_ids=token, attention_mask=mask, past_key_values=cache,
                               use_cache=True, return_dict=True)
                cache = output.past_key_values
                token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if validate:
                    require_finite(output.logits, "decode.logits")
                    logits.append(output.logits[:, -1, :].clone())
                    tokens.append(token.clone())
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - wall_start) * 1000
        if kernel is not None:
            expected = steps * len(mlps)
            if (counts["stage_calls"] != expected or counts["fallback_calls"] != 0
                    or any(row != {"stage_calls": steps, "fallback_calls": 0}
                           for row in counts["modules"].values())):
                raise RuntimeError(f"Expected every MLP stage to dispatch each step: {counts}")
            if validate and len(captured) != expected:
                raise RuntimeError("Missing validation captures")
        result = {"wall_ms_per_step": wall_ms / steps,
                  "cuda_event_ms_per_step": start.elapsed_time(end) / steps, "dispatch": counts}
        if validate:
            result.update(logits=torch.stack(logits).float(),
                          tokens=torch.cat([initial_token, *tokens], dim=1).cpu().tolist(),
                          captured=captured)
        return result


def local_checks(captured):
    """Check each replacement against eager and staged FP64 on the same input."""
    from fastinfer.kernels.swiglu import reference_swiglu
    rows = []
    with torch.inference_mode():
        for name, gate, up, output in captured:
            for label, value in (("gate", gate), ("up", up), ("product", output)):
                require_finite(value, name + "." + label)
            native = reference_swiglu(gate, up)
            activated = (gate.double() * torch.sigmoid(gate.double())).to(gate.dtype)
            oracle = (activated.double() * up.double()).to(gate.dtype)
            comparisons = {}
            for label, expected in (("eager", native), ("staged_fp64", oracle)):
                require_finite(expected, name + "." + label)
                error = (output.double() - expected.double()).abs()
                comparisons[label] = {
                    "mismatches": int((error > 2e-4 + .002 * expected.double().abs()).sum().item()),
                    "max_abs_error": error.max().item(),
                    "different_elements": int((output != expected).sum().item()),
                }
            rows.append({"module": name, "comparisons": comparisons,
                         "passed": all(v["mismatches"] == 0 for v in comparisons.values())})
    return {"calls_checked": len(rows), "all_passed": all(r["passed"] for r in rows), "per_call": rows}


def timing_allowed(case, *, diagnose, expected_calls):
    """Require stable baseline and every candidate's model/local/counter checks."""
    if diagnose or not case["baseline_repeat_accuracy"]["passed"]:
        return False
    for row in case["candidates"].values():
        if (not row["accuracy"]["passed"] or not row["local_checks"]["all_passed"]
                or row["local_checks"]["calls_checked"] != expected_calls
                or row["dispatch"]["stage_calls"] != expected_calls
                or row["dispatch"]["fallback_calls"] != 0):
            return False
    return set(case["candidates"]) == {"compiled_swiglu", "triton_swiglu"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--diagnose", action="store_true", help="check model outputs, skip timing")
    args = parser.parse_args()
    if not torch.cuda.is_available() or args.steps < 1 or args.repetitions < 2:
        parser.error("CUDA, positive steps and at least two repetitions required")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fastinfer.kernels.swiglu import reference_swiglu
    from fastinfer.kernels.triton_swiglu import swiglu

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    print(f"Loading {MODEL}; FP16, math SDPA; only decode SwiGLU stage varies", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16, attn_implementation="sdpa",
        output_attentions=False, low_cpu_mem_usage=True, use_safetensors=True,
        trust_remote_code=False,
    ).to("cuda").eval()
    if model.config.hidden_act != "silu":
        raise RuntimeError("Expected original SiLU MLP")
    mlps = {name: module for name, module in model.named_modules()
            if module.__class__.__name__ == "Qwen2MLP"}
    if len(mlps) != model.config.num_hidden_layers or not mlps:
        raise RuntimeError("Unexpected MLP count")
    # One shared compiled stage for identical single-token shapes across all layers.
    torch.compiler.reset()
    compiled = torch.compile(reference_swiglu, **COMPILED_SETTINGS)
    kernels = {"eager": None, "compiled_swiglu": compiled, "triton_swiglu": swiglu}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"qwen_swiglu_decode_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "qwen_three_way_swiglu_decode", "model": MODEL, "revision": REVISION,
        "dtype": "float16", "attention": "sdpa_math", "cache": "DynamicCache",
        "steps": args.steps, "repetitions": args.repetitions, "diagnose": args.diagnose,
        "replacement_modules": list(mlps), "compiled_settings": COMPILED_SETTINGS,
        "accuracy": {"logits_rtol": .01, "logits_atol": .01, "local_rtol": .002,
                     "local_atol": 2e-4, "identical_greedy_tokens_required": True},
        "scope": "Original eager model versus only SiLU(gate)*up compiled or custom, "
                 "on all eligible single-token decode MLP calls. Projections, RMSNorm and attention "
                 "stay original. Original fresh-cache prefill excluded from timing, no CUDA graphs. "
                 "Fixed decode steps even after EOS; model+cache+argmax+dispatch included. "
                 "Compile and validation captures excluded from timed trials. "
                 "Strict baseline-repeat/logit/token/local/counter checks gate timing. "
                 "This is NOT a whole-model torch.compile comparison.",
        "timing": {"warmup_trajectories": 2, "order": "all_six_provider_permutations",
                   "primary_metric": "wall_ms_per_step"},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("decode_swiglu.py", "kernels/swiglu.py", "kernels/triton_swiglu.py",
                                       "kernels/swiglu_config.py", "benchmarks/qwen_swiglu_decode_compare.py",
                                       "benchmarks/qwen_decode_compare.py", "benchmarks/softmax_compare.py")},
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "cases": [],
    }
    try:
        for prompt in PROMPTS:
            print(f"\nPrompt: {prompt!r}", flush=True)
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
            baseline = decode_trial(model, encoded, mlps, None, args.steps, validate=True)
            repeat = decode_trial(model, encoded, mlps, None, args.steps, validate=True)
            case = {"prompt": prompt, "baseline_repeat_accuracy": compare_outputs(baseline, repeat),
                    "candidates": {}}
            report["cases"].append(case)
            for name in ("compiled_swiglu", "triton_swiglu"):
                candidate = decode_trial(model, encoded, mlps, kernels[name], args.steps, validate=True)
                accuracy = compare_outputs(baseline, candidate)
                local = local_checks(candidate["captured"])
                case["candidates"][name] = {"accuracy": accuracy, "local_checks": local,
                                             "dispatch": candidate["dispatch"]}
                print(f"  {name}: stage calls {candidate['dispatch']['stage_calls']} | "
                      f"fallback {candidate['dispatch']['fallback_calls']} | "
                      f"local checks PASS {local['all_passed']} | "
                      f"logit mismatches {accuracy['logit_mismatches']} | "
                      f"identical tokens {accuracy['identical_greedy_tokens']}", flush=True)
                del candidate
            del baseline, repeat
            allowed = timing_allowed(case, diagnose=False,
                                     expected_calls=args.steps * len(mlps))
            case["timing_status"] = "diagnostic_only" if args.diagnose else (
                "strict_accuracy_passed" if allowed else "accuracy_failed_timing_skipped"
            )
            print(f"  Timing status: {case['timing_status']}", flush=True)
            if allowed and not args.diagnose:
                for _ in range(2):
                    for kernel in kernels.values():
                        decode_trial(model, encoded, mlps, kernel, args.steps)
                orders = measurement_orders(kernels, args.repetitions)
                samples = {name: [] for name in kernels}
                for order in orders:
                    for name in order:
                        samples[name].append(decode_trial(model, encoded, mlps, kernels[name], args.steps))
                case["samples"], case["measurement_orders"] = samples, orders
                case["medians_ms_per_step"] = {
                    name: statistics.median(row["wall_ms_per_step"] for row in rows)
                    for name, rows in samples.items()}
                case["paired_comparisons"] = {}
                for name, base, candidate in (("eager_vs_compiled", "eager", "compiled_swiglu"),
                                               ("eager_vs_triton", "eager", "triton_swiglu"),
                                               ("compiled_vs_triton", "compiled_swiglu", "triton_swiglu")):
                    comparison = native_triton_comparison(
                        [row["wall_ms_per_step"] * 1000 for row in samples[base]],
                        [row["wall_ms_per_step"] * 1000 for row in samples[candidate]])
                    comparison.update(baseline=base, candidate=candidate)
                    case["paired_comparisons"][name] = comparison
                    print(f"  {name}: {comparison['ratio_of_separate_medians']:.3f}x | "
                          f"paired saving {comparison['paired_median_saving_us']:+.2f}us/step | "
                          f"candidate faster {comparison['triton_faster_rounds']}/{args.repetitions}", flush=True)
                case["timed_dispatch_totals"] = {
                    name: {field: sum(row["dispatch"][field] for row in samples[name])
                           for field in ("stage_calls", "fallback_calls")}
                    for name in ("compiled_swiglu", "triton_swiglu")
                }
                print("  Full decode ms/step:", case["medians_ms_per_step"], flush=True)
                print("  Timed dispatch totals:", case["timed_dispatch_totals"], flush=True)
            destination.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
