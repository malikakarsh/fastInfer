"""Diagnose the first differing Qwen MLP product on one original decode step."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from qwen_decode_compare import MODEL, REVISION, ROOT
from qwen_decode_profile import prepare_decode, decode_steps
from qwen_linear_check import PROMPTS, require_finite
from qwen_swiglu_check import capture_mlp_branches
from qwen_rmsnorm_stages import difference_report, compare_stages
from softmax_compare import COMPILED_SETTINGS


def pytorch_stages(gate, up):
    """Native FP32 SiLU, then explicit half rounding and FP32 product stages.

    torch.sigmoid is a diagnostic comparator, not an asserted internal step
    of native F.silu: the pinned CUDA source calculates x/(1+exp(-x)).
    """
    silu_fp32 = F.silu(gate.float())
    silu_fp16 = silu_fp32.half()
    product_fp32 = silu_fp16.float() * up.float()
    return {"sigmoid_fp32": torch.sigmoid(gate.float()), "silu_fp32": silu_fp32,
            "silu_fp16": silu_fp16, "product_fp32": product_fp32,
            "product_fp16": product_fp32.half()}


def unrounded_swiglu(gate, up):
    """Hypothesis proxy: retain FP32 activation through multiplication."""
    return (F.silu(gate.float()) * up.float()).to(gate.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-index", type=int, choices=range(len(PROMPTS)), default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.__version__.split("+")[0] != "2.5.1":
        parser.error("CUDA and pinned PyTorch2.5.1 required")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fastinfer.kernels.swiglu import reference_swiglu
    from fastinfer.kernels.triton_swiglu import swiglu
    from fastinfer.kernels.swiglu_diagnostic import triton_swiglu_stages

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.compiler.reset()
    compiled = torch.compile(reference_swiglu, **COMPILED_SETTINGS)
    compiled_unrounded = torch.compile(unrounded_swiglu, **COMPILED_SETTINGS)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    print(f"Loading {MODEL}; original FP16 model, math SDPA; one-step arithmetic diagnosis", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16, attn_implementation="sdpa",
        output_attentions=False, low_cpu_mem_usage=True, use_safetensors=True,
        trust_remote_code=False,
    ).to("cuda").eval()
    mlps = {name: module for name, module in model.named_modules()
            if module.__class__.__name__ == "Qwen2MLP"}
    if len(mlps) != model.config.num_hidden_layers or not mlps:
        raise RuntimeError("Unexpected MLP count")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"qwen_swiglu_stages_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "qwen_swiglu_stage_diagnosis", "model": MODEL, "revision": REVISION,
        "prompt": PROMPTS[args.prompt_index], "prompt_index": args.prompt_index,
        "scope": "One unchanged original-model prefill and first cached decode. "
                 "Screen all28MLPs; select first exact product difference for each candidate. "
                 "Inspect selected original inputs, not later candidate trajectory inputs. "
                 "Instrumented final output must exactly match production before attribution. "
                 "Sigmoid comparator is not an asserted internal native SiLU step. "
                 "Unrounded proxy tests a hypothesis; does not by itself establish generated code. "
                 "No tolerance changes, timing, production changes or full-model equivalence claim.",
        "compiled_settings": COMPILED_SETTINGS,
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("kernels/triton_swiglu.py", "kernels/swiglu_diagnostic.py",
                                       "benchmarks/qwen_swiglu_stages.py", "benchmarks/qwen_swiglu_check.py",
                                       "benchmarks/qwen_rmsnorm_stages.py")},
        "screened": [], "selected": {},
    }
    try:
        with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            encoded = tokenizer(PROMPTS[args.prompt_index], return_tensors="pt").to("cuda")
            cache, token, mask = prepare_decode(model, encoded)
            logits, captured = capture_mlp_branches(
                mlps, lambda: decode_steps(model, cache, token, mask, 1))
            require_finite(logits, "original_decode.logits")
            report["input_ids"] = encoded["input_ids"].tolist()
            report["decode_input_token"] = token.item()
            selections = {}
            for name, tensors in captured.items():
                gate = tensors["gate"].reshape(1, 8960).contiguous()
                up = tensors["up"].reshape(1, 8960).contiguous()
                product = tensors["product"].reshape(1, 8960)
                require_finite(gate, name + ".gate")
                require_finite(up, name + ".up")
                native = reference_swiglu(gate, up)
                if not torch.equal(native, product):
                    raise RuntimeError(f"Native reference does not exactly match captured product at {name}")
                outputs = {"compiled": compiled(gate, up), "triton": swiglu(gate, up)}
                row = {"module": name, "differences": {}}
                for candidate, output in outputs.items():
                    diff = difference_report(output, product)
                    row["differences"][candidate] = diff
                    if diff["different_elements"] and candidate not in selections:
                        selections[candidate] = (name, gate, up, product, output)
                report["screened"].append(row)
            for candidate, (name, gate, up, product, output) in selections.items():
                native_stages = pytorch_stages(gate, up)
                diagnostic = triton_swiglu_stages(gate, up)
                production = swiglu(gate, up)
                model_match = torch.equal(native_stages["product_fp16"], product)
                activation_match = torch.equal(native_stages["silu_fp16"], F.silu(gate))
                diagnostic_match = torch.equal(diagnostic["product_fp16"], production)
                stages = compare_stages(diagnostic, native_stages)
                # Keep all eager operations after the selected intermediate.
                hybrid = diagnostic["silu_fp16"] * up
                no_round_eager = unrounded_swiglu(gate, up)
                no_round_compiled = compiled_unrounded(gate, up)
                native = reference_swiglu(gate, up)
                snapshots = destination.with_name(destination.stem + f"_{candidate}.pt")
                torch.save({"module": name, "candidate": candidate, "prompt": report["prompt"],
                            "gate": gate.cpu(), "up": up.cpu(), "captured_product": product.cpu(),
                            "candidate_product": output.cpu(), "production_triton": production.cpu(),
                            "pytorch_stages": {k: v.cpu() for k, v in native_stages.items()},
                            "triton_stages": {k: v.cpu() for k, v in diagnostic.items()},
                            "unrounded_eager_product": no_round_eager.cpu(),
                            "unrounded_compiled_product": no_round_compiled.cpu()}, snapshots)
                case = {
                    "selected_module": name,
                    "native_stage_product_matches_captured": model_match,
                    "native_fp32_then_half_silu_matches_native_half_silu": activation_match,
                    "diagnostic_matches_production_triton": diagnostic_match,
                    "interpretation_status": "cross_checks_passed" if model_match and activation_match and diagnostic_match
                                             else "cross_check_failed_do_not_attribute",
                    "triton_vs_native_stages": stages,
                    "candidate_vs_captured": difference_report(output, product),
                    "hybrid_triton_silu16_then_eager_multiply_vs_production": difference_report(hybrid, production),
                    "compiled_vs_unrounded_eager_proxy": difference_report(compiled(gate, up), no_round_eager),
                    "compiled_vs_unrounded_compiled_proxy": difference_report(compiled(gate, up), no_round_compiled),
                    "unrounded_eager_vs_original": difference_report(no_round_eager, native),
                    "tensor_snapshot": str(snapshots),
                }
                report["selected"][candidate] = case
                print(f"\nCandidate: {candidate} | first differing MLP: {name}", flush=True)
                print(f"  Native stage reconstruction matches model: {model_match}", flush=True)
                print(f"  Native FP32-to-half SiLU matches native half SiLU: {activation_match}", flush=True)
                print(f"  Diagnostic matches production Triton: {diagnostic_match}", flush=True)
                print("  Triton versus native stage comparators on this selected pair:", flush=True)
                for stage, diff in stages["stages"].items():
                    print(f"  {stage}: {diff['different_elements']}/{diff['elements']} different | "
                          f"max difference {diff['max_abs_difference']:.9g}", flush=True)
                for key in ("candidate_vs_captured", "hybrid_triton_silu16_then_eager_multiply_vs_production",
                            "compiled_vs_unrounded_eager_proxy", "compiled_vs_unrounded_compiled_proxy"):
                    diff = case[key]
                    print(f"  {key}: {diff['different_elements']} different | "
                          f"max difference {diff['max_abs_difference']:.9g}", flush=True)
                print(f"  Interpretation: {case['interpretation_status']} | snapshot: {snapshots}", flush=True)
            for candidate in ("compiled", "triton"):
                if candidate not in selections:
                    print(f"{candidate}: no differing product found in this original first decode step", flush=True)
            report["missing_differences"] = [name for name in ("compiled", "triton") if name not in selections]
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
