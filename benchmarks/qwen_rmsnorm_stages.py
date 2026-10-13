"""Find arithmetic differences on an unchanged Qwen model's decode inputs."""

from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from qwen_decode_compare import MODEL, REVISION, ROOT
from qwen_decode_profile import prepare_decode, decode_steps
from qwen_linear_check import PROMPTS, require_finite
from qwen_rmsnorm_check import capture_norm_calls


def pytorch_stages(x, weight, *, eps):
    """Qwen's original eager arithmetic order, with intermediate values retained."""
    squared = x.float().pow(2)
    mean_square = squared.mean(dim=-1, keepdim=True)
    variance_with_eps = mean_square + eps
    inverse_rms = torch.rsqrt(variance_with_eps)
    normalized_fp32 = x.float() * inverse_rms
    rounded = normalized_fp32.to(x.dtype)
    return {
        "squared_values": squared,
        "mean_square": mean_square,
        "mean_square_plus_eps": variance_with_eps,
        "inverse_rms": inverse_rms,
        "normalized_fp32": normalized_fp32,
        "normalized_fp16": rounded,
        "weighted_output_fp16": rounded * weight,
    }


def difference_report(actual, reference):
    """Exact differences, not a changed accuracy tolerance; include one example."""
    require_finite(actual, "stage.actual")
    require_finite(reference, "stage.reference")
    actual = actual.flatten().double()
    reference = reference.flatten().double()
    if actual.shape != reference.shape:
        raise ValueError("stage shapes must match")
    difference = (actual - reference).abs()
    index = int(difference.argmax().item())
    return {
        "different_elements": int((actual != reference).sum().item()),
        "elements": actual.numel(),
        "max_abs_difference": difference[index].item(),
        "largest_difference_example": {
            "flat_index": index,
            "pytorch": reference[index].item(),
            "triton_or_hybrid": actual[index].item(),
        },
    }


def compare_stages(actual, reference):
    comparisons = {name: difference_report(actual[name], reference[name])
                   for name in reference}
    first = next((name for name, row in comparisons.items()
                  if row["different_elements"]), None)
    return {"first_different_stage": first, "stages": comparisons}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; run with the Kaggle Python 3.12 environment")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fastinfer.kernels.triton_rmsnorm import rmsnorm
    from fastinfer.kernels.rmsnorm_diagnostic import triton_stages

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    print(f"Loading {MODEL}; original FP16 model, math SDPA; arithmetic diagnosis", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16,
        attn_implementation="sdpa", output_attentions=False,
        low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False,
    ).to("cuda").eval()
    norms = {name: module for name, module in model.named_modules()
             if module.__class__.__name__ == "Qwen2RMSNorm"}
    if len(norms) != 2 * model.config.num_hidden_layers + 1:
        raise RuntimeError("Unexpected normalization module count")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"qwen_rmsnorm_stages_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "qwen_rmsnorm_stage_diagnosis", "model": MODEL, "revision": REVISION,
        "scope": "Original prefill and first cached decode only. For each prompt, inspect "
                 "the module with the largest production-kernel output difference. "
                 "Instrumented intermediate stores may affect code generation. "
                 "No timing, tolerance change, or full-generation equivalence claim.",
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("kernels/triton_rmsnorm.py", "kernels/rmsnorm_diagnostic.py",
                                       "benchmarks/qwen_rmsnorm_stages.py")},
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
                candidates = []
                for name, (x, model_output) in captured.items():
                    output = rmsnorm(x.contiguous(), norms[name].weight,
                                     eps=norms[name].variance_epsilon)
                    candidates.append({"module": name,
                                       **difference_report(output, model_output)})
                selected = max(candidates, key=lambda row: row["max_abs_difference"])
                name = selected["module"]
                module = norms[name]
                x, model_output = captured[name]
                x = x.contiguous()
                weight = module.weight.detach()
                eps = module.variance_epsilon
                original = pytorch_stages(x, weight, eps=eps)
                diagnostic = triton_stages(x, weight, eps=eps)
                production = rmsnorm(x, weight, eps=eps)
                matches_model = torch.equal(original["weighted_output_fp16"], model_output)
                matches_production = torch.equal(diagnostic["weighted_output_fp16"], production)
                comparison = compare_stages(diagnostic, original)

                # Keep PyTorch's remaining operations, replacing just one intermediate.
                from_mean_inverse = torch.rsqrt(diagnostic["mean_square"] + eps)
                from_mean = (x.float() * from_mean_inverse).half() * weight
                from_inverse = (x.float() * diagnostic["inverse_rms"]).half() * weight
                snapshots = destination.with_name(destination.stem + f"_case{len(report['cases']) + 1}.pt")
                torch.save({"module": name, "input": x.cpu(), "weight": weight.cpu(), "eps": eps,
                            "captured_output": model_output.cpu(), "production_output": production.cpu(),
                            "pytorch": {k: v.cpu() for k, v in original.items()},
                            "triton_diagnostic": {k: v.cpu() for k, v in diagnostic.items()}}, snapshots)
                case = {
                    "prompt": prompt, "selected_module": name, "eps": eps,
                    "modules_screened": candidates,
                    "pytorch_stages_exactly_match_captured_model": matches_model,
                    "diagnostic_exactly_matches_production_kernel": matches_production,
                    "interpretation_status": "output_cross_checks_passed" if matches_model and matches_production
                                             else "do_not_attribute_production_difference_from_these_stages",
                    **comparison,
                    "hybrid_output_vs_model": {
                        "pytorch_using_triton_mean": difference_report(from_mean, model_output),
                        "pytorch_using_triton_inverse_rms": difference_report(from_inverse, model_output),
                    },
                    "tensor_snapshot": str(snapshots),
                }
                report["cases"].append(case)
                print(f"\nPrompt: {prompt!r}\n  Selected: {name}\n"
                      f"  PyTorch reconstruction matches model: {matches_model}\n"
                      f"  Diagnostic matches production kernel: {matches_production}\n"
                      f"  First differing diagnostic stage: {comparison['first_different_stage']}", flush=True)
                for stage, row in comparison["stages"].items():
                    print(f"  {stage}: {row['different_elements']}/{row['elements']} different | "
                          f"max difference {row['max_abs_difference']:.9g}", flush=True)
                for label, row in case["hybrid_output_vs_model"].items():
                    print(f"  {label}: {row['different_elements']} output elements different | "
                          f"max difference {row['max_abs_difference']:.9g}", flush=True)
                if not matches_model or not matches_production:
                    print("  Cross-check failed: these stages cannot explain the production "
                          "kernel difference reliably.", flush=True)
                destination.write_text(json.dumps(report, indent=2) + "\n")
                del captured, cache, logits
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
