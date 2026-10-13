"""Check fused RMSNorm on original Qwen decode inputs and learned weights."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from fastinfer.kernels.rmsnorm import reference_rmsnorm
from qwen_decode_compare import MODEL, REVISION, ROOT
from qwen_decode_profile import prepare_decode, decode_steps
from qwen_linear_check import PROMPTS, require_finite


def capture_norm_calls(norms, execute):
    """Capture every selected module once; always remove hooks on exit."""
    captured = {name: [] for name in norms}
    handles = []

    def capture(name, inputs, output):
        captured[name].append((inputs[0].detach().clone(), output.detach().clone()))

    try:
        for name, module in norms.items():
            handles.append(module.register_forward_hook(
                lambda module, inputs, output, name=name: capture(name, inputs, output)
            ))
        result = execute()
    finally:
        for handle in handles:
            handle.remove()
    if any(len(values) != 1 for values in captured.values()):
        counts = {name: len(values) for name, values in captured.items()}
        raise RuntimeError(f"Expected one decode call per normalization module: {counts}")
    return result, {name: values[0] for name, values in captured.items()}


def error_report(value, reference):
    require_finite(value, "normalization_output")
    require_finite(reference, "normalization_reference")
    difference = (value.float() - reference.float()).abs()
    return {"mismatches": int((difference > .002 + .002 * reference.float().abs()).sum().item()),
            "elements": reference.numel(), "max_abs_error": difference.max().item()}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fastinfer.kernels.triton_rmsnorm import rmsnorm

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    print(f"Loading {MODEL}; FP16, math SDPA, original model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16,
        attn_implementation="sdpa", output_attentions=False,
        low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False,
    ).to("cuda").eval()
    norms = {name: module for name, module in model.named_modules()
             if module.__class__.__name__ == "Qwen2RMSNorm"}
    expected_count = 2 * model.config.num_hidden_layers + 1
    if len(norms) != expected_count:
        raise RuntimeError(f"Expected {expected_count} RMSNorm modules, found {len(norms)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"qwen_real_rmsnorm_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "qwen_real_decode_rmsnorm_accuracy", "model": MODEL, "revision": REVISION,
        "dtype": "float16", "attention": "sdpa_math", "cache": "DynamicCache",
        "module_count": len(norms), "rtol": .002, "atol": .002,
        "kernel": {"num_warps": 4, "enable_fp_fusion": False},
        "scope": "Original model performs prefill and one cached decode per prompt. "
                 "Captured normalization inputs are replayed separately with real learned weights. "
                 "No model replacement, timing or full-model quality claim.",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("kernels/rmsnorm.py", "kernels/triton_rmsnorm.py",
                                       "benchmarks/qwen_rmsnorm_check.py")},
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "cases": [],
    }
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            for prompt in PROMPTS:
                encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
                cache, token, mask = prepare_decode(model, encoded)
                logits, captured = capture_norm_calls(
                    norms, lambda: decode_steps(model, cache, token, mask, 1)
                )
                require_finite(logits, "original_decode.logits")
                case = {"prompt": prompt, "capture_calls": len(captured), "modules": []}
                report["cases"].append(case)
                for name, module in norms.items():
                    x, model_output = captured[name]
                    if tuple(x.shape) != (1, 1, 1536):
                        raise RuntimeError(f"Unexpected input at {name}: {x.shape}")
                    require_finite(x, name + ".input")
                    weight = module.weight.detach()
                    require_finite(weight, name + ".weight")
                    eps = module.variance_epsilon
                    x = x.contiguous()
                    actual = rmsnorm(x, weight, eps=eps)
                    reference = reference_rmsnorm(x, weight, eps=eps)
                    oracle = reference_rmsnorm(x.float(), weight.float(), eps=eps)
                    comparisons = {
                        "captured_vs_fp32": error_report(model_output, oracle),
                        "reference_vs_fp32": error_report(reference, oracle),
                        "triton_vs_fp32": error_report(actual, oracle),
                        "triton_vs_captured": error_report(actual, model_output),
                        "reference_vs_captured": error_report(reference, model_output),
                    }
                    case["modules"].append({
                        "name": name, "shape": list(x.shape), "eps": eps,
                        "comparisons": comparisons,
                        "triton_different_elements_from_captured": int((actual != model_output).sum().item()),
                        "passed": all(row["mismatches"] == 0 for row in comparisons.values()),
                    })
                failed = [row["name"] for row in case["modules"] if not row["passed"]]
                case["passed"] = not failed
                largest_difference = max(row["comparisons"]["triton_vs_captured"]["max_abs_error"]
                                         for row in case["modules"])
                print(f"Prompt {prompt!r}: {len(norms)-len(failed)}/{len(norms)} modules PASS | "
                      f"largest Triton/model difference {largest_difference:.6f}", flush=True)
                destination.write_text(json.dumps(report, indent=2) + "\n")
                if failed:
                    raise RuntimeError(f"Normalization accuracy failed: {failed}")
                del captured, logits, cache
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        destination.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
