"""Capture real Qwen gate/up tensors; check all MLPs and time one isolated stage."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from qwen_linear_check import MODEL, PROMPTS, require_finite
from swiglu_compare import measure_case

ROOT = Path(__file__).resolve().parents[1]
REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
PREFILL_PROMPT = (
    "Explain how plants convert sunlight into chemical energy through photosynthesis. "
    "Describe the roles of chlorophyll, water, carbon dioxide, oxygen, and glucose, "
    "and explain why this process matters for living organisms on Earth."
)


def capture_mlp_branches(mlps, execute):
    """Capture gate/up outputs and the product supplied to down_proj; restore hooks."""
    captured = {name: {key: [] for key in ("gate", "up", "product")} for name in mlps}
    handles = []

    def save(name, key, tensor):
        captured[name][key].append(tensor.detach().clone())

    try:
        for name, mlp in mlps.items():
            handles.append(mlp.gate_proj.register_forward_hook(
                lambda module, inputs, output, name=name: save(name, "gate", output)
            ))
            handles.append(mlp.up_proj.register_forward_hook(
                lambda module, inputs, output, name=name: save(name, "up", output)
            ))
            handles.append(mlp.down_proj.register_forward_pre_hook(
                lambda module, inputs, name=name: save(name, "product", inputs[0])
            ))
        result = execute()
    finally:
        for handle in handles:
            handle.remove()
    counts = {name: {key: len(values) for key, values in branches.items()}
              for name, branches in captured.items()}
    if any(count != 1 for branches in counts.values() for count in branches.values()):
        raise RuntimeError(f"Expected one gate/up/product capture per MLP: {counts}")
    return result, {name: {key: values[0] for key, values in branches.items()}
                    for name, branches in captured.items()}


def check_captured(captured):
    from fastinfer.kernels.swiglu import reference_swiglu
    from fastinfer.kernels.triton_swiglu import swiglu, default_swiglu_config

    records = []
    for name, tensors in captured.items():
        for key, value in tensors.items():
            require_finite(value, name + "." + key)
        gate, up, product = (tensors[key] for key in ("gate", "up", "product"))
        if gate.shape != up.shape or gate.shape != product.shape or gate.shape[-1] != 8960:
            raise RuntimeError(f"Unexpected capture shapes at {name}")
        if gate.dtype != torch.float16 or up.dtype != gate.dtype or product.dtype != gate.dtype:
            raise RuntimeError(f"Expected FP16 branches at {name}")
        gate = gate.reshape(-1, 8960).contiguous()
        up = up.reshape(-1, 8960).contiguous()
        product = product.reshape(-1, 8960)
        native = reference_swiglu(gate, up)
        actual = swiglu(gate, up)
        activation = (gate.double() * torch.sigmoid(gate.double())).to(gate.dtype)
        oracle = (activation.double() * up.double()).to(gate.dtype)
        comparisons = {}
        for label, value, expected in (
            ("native_vs_captured", native, product),
            ("triton_vs_captured", actual, product),
            ("native_vs_oracle", native, oracle),
            ("triton_vs_oracle", actual, oracle),
        ):
            require_finite(value, label)
            require_finite(expected, label + ".reference")
            error = (value.double() - expected.double()).abs()
            mismatches = int((error > 2e-4 + .002 * expected.double().abs()).sum().item())
            comparisons[label] = {"mismatches": mismatches,
                                  "different_elements": int((value != expected).sum().item()),
                                  "max_abs_error": error.max().item()}
            torch.testing.assert_close(value, expected, rtol=.002, atol=2e-4)
        records.append({"name": name, "shape": list(gate.shape),
                        "selected_config": list(default_swiglu_config(gate)),
                        "comparisons": comparisons, "passed": True})
    return records


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from qwen_decode_profile import prepare_decode, decode_steps

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    print(f"Loading {MODEL}; FP16, math SDPA; original model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float16,
        attn_implementation="sdpa", output_attentions=False,
        low_cpu_mem_usage=True, use_safetensors=True, trust_remote_code=False,
    ).to("cuda").eval()
    mlps = {name: module for name, module in model.named_modules()
            if module.__class__.__name__ == "Qwen2MLP"}
    if len(mlps) != model.config.num_hidden_layers or not mlps:
        raise RuntimeError("Unexpected MLP count")
    target = "model.layers.0.mlp"
    if target not in mlps:
        raise RuntimeError("Expected first Qwen MLP")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"qwen_real_swiglu_{stamp}.json"
    report = {
        "benchmark": "qwen_captured_elementwise_swiglu", "model": MODEL, "revision": REVISION,
        "dtype": "float16", "attention": "sdpa_math", "module_count": len(mlps),
        "rtol": .002, "atol": 2e-4, "timed_module": target,
        "scope": "Original model remains unchanged. Captured branches use learned Qwen projections. "
                 "All MLP products checked; isolated stage replay timed only for layer0. "
                 "One actual16-token prefill and three one-token cached decode captures. "
                 "Capture hooks are removed before replay timing; model stays resident. "
                 "No full-model speedup or attention/softmax integration measured.",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("benchmarks/qwen_swiglu_check.py", "benchmarks/swiglu_compare.py",
                                       "benchmarks/qwen_decode_profile.py", "kernels/swiglu.py",
                                       "kernels/triton_swiglu.py", "kernels/swiglu_config.py")},
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "cases": [],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            for phase, prompt in [("prefill16", PREFILL_PROMPT),
                                  *(("decode1", prompt) for prompt in PROMPTS)]:
                encoded = tokenizer(prompt, return_tensors="pt",
                                    **({"truncation": True, "max_length": 16}
                                       if phase == "prefill16" else {})).to("cuda")
                with sdpa_kernel(SDPBackend.MATH):
                    if phase == "prefill16":
                        if encoded["input_ids"].shape != (1, 16):
                            raise RuntimeError("Expected16tokens without padding")
                        result, captured = capture_mlp_branches(
                            mlps, lambda: model(**encoded, use_cache=False, return_dict=True)
                        )
                        require_finite(result.logits, "original_prefill.logits")
                    else:
                        cache, token, mask = prepare_decode(model, encoded)
                        result, captured = capture_mlp_branches(
                            mlps, lambda: decode_steps(model, cache, token, mask, 1)
                        )
                        require_finite(result, "original_decode.logits")
                records = check_captured(captured)
                case = {"phase": phase, "prompt": prompt,
                        "input_ids": encoded["input_ids"].tolist(), "modules": records}
                report["cases"].append(case)
                largest = max(row["comparisons"]["triton_vs_captured"]["max_abs_error"] for row in records)
                print(f"{phase}: {len(records)}/{len(mlps)} MLP checks PASS | "
                      f"largest Triton/model product difference {largest:.6f}", flush=True)
                tensors = captured[target]
                gate = tensors["gate"].reshape(-1, 8960).contiguous()
                up = tensors["up"].reshape(-1, 8960).contiguous()
                expected_shape = (16 if phase == "prefill16" else 1, 8960)
                if tuple(gate.shape) != expected_shape:
                    raise RuntimeError(f"Unexpected timing shape {gate.shape}")
                case["isolated_timing"] = measure_case(gate, up, include_compiled=True)
                timing = case["isolated_timing"]
                medians = {name: row["latency_us"]["median"]
                           for name, row in timing["implementations"].items()}
                print(f"  Layer0 {list(gate.shape)}: eager {medians['pytorch_native']:.3f}us | "
                      f"compiled {medians['pytorch_compiled']:.3f}us | "
                      f"Triton {medians['triton']:.3f}us | "
                      f"config {timing['triton_resources']['block_size']}/"
                      f"{timing['triton_resources']['num_warps']}", flush=True)
                for key in ("native_vs_triton", "compiled_vs_triton"):
                    comparison = timing[key]
                    print(f"  {key}: paired saving {comparison['paired_median_saving_us']:+.3f}us | "
                          f"Triton faster {comparison['triton_faster_rounds']}/30", flush=True)
                destination.write_text(json.dumps(report, indent=2) + "\n")
                del captured, result
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
