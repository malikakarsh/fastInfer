"""Compare full cached Qwen decode with one opt-in GEMV projection."""

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

from fastinfer.decode_linear import use_decode_gemv
from qwen_linear_check import MODEL, PROMPTS, require_finite

ROOT = Path(__file__).resolve().parents[1]
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"


def decode_trial(model, encoded, steps, *, custom, validate=False):
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
        logits, tokens = [], []
        context = use_decode_gemv(projection) if custom else nullcontext(None)
        with context as counts:
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
        return result


def compare_outputs(baseline, candidate):
    """Same greedy trajectory plus elementwise logits tolerance are required."""
    a, b = baseline["logits"], candidate["logits"]
    require_finite(a, "baseline.logits")
    require_finite(b, "candidate.logits")
    error = (a - b).abs()
    mismatches = int((error > .01 + .01 * a.abs()).sum().item())
    same_tokens = baseline["tokens"] == candidate["tokens"]
    return {"logit_mismatches": mismatches, "logit_elements": a.numel(),
            "max_abs_logit_error": error.max().item(),
            "identical_greedy_tokens": same_tokens,
            "baseline_token_ids": baseline["tokens"],
            "candidate_token_ids": candidate["tokens"],
            "passed": same_tokens and mismatches == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
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
            candidate = decode_trial(model, encoded, args.steps, custom=True, validate=True)
            accuracy = compare_outputs(baseline, candidate)
            case = {"prompt": prompt, "accuracy": accuracy,
                    "validation_dispatch": candidate["dispatch"]}
            report["cases"].append(case)
            del baseline, candidate
            save()
            print(f"  Logit mismatches: {accuracy['logit_mismatches']} | "
                  f"identical tokens: {accuracy['identical_greedy_tokens']}", flush=True)
            if not accuracy["passed"]:
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
            save()
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        save()
        raise
    finally:
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
