"""Check a frozen GEMV on real Qwen weights and captured decode activations."""

import argparse
from contextlib import contextmanager
import hashlib
import json
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-1.5B"
PROMPTS = (
    "The capital of France is",
    "Explain why the sky appears blue in one sentence.",
    "def add(a, b):\n    return",
)


class NonfiniteTensorError(RuntimeError):
    """A numerical failure with JSON-safe diagnostic details."""

    def __init__(self, details):
        self.details = details
        super().__init__(f"Nonfinite tensor at {details['stage']}: {details}")


def require_finite(value, stage):
    import torch

    if not torch.isfinite(value).all().item():
        raise NonfiniteTensorError({
            "stage": stage,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "nan_count": torch.isnan(value).sum().item(),
            "inf_count": torch.isinf(value).sum().item(),
            "elements": value.numel(),
        })


@contextmanager
def trace_finite_modules(model, phase):
    """Check leaf-module boundaries; hooks are removed before any timing.

    Inspect the first tensor only: attention masks can legitimately contain -inf.
    This locates the first observed failure, not necessarily its internal operation.
    """
    import torch

    handles = []

    def check(value, stage):
        if isinstance(value, (tuple, list)) and value:
            value = value[0]
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            require_finite(value, stage)

    try:
        for name, child in model.named_modules():
            if not name or any(child.children()):
                continue
            stage = f"{phase}:{name}"
            handles.append(child.register_forward_pre_hook(
                lambda module, inputs, stage=stage: check(inputs, stage + ".input")
            ))
            handles.append(child.register_forward_hook(
                lambda module, inputs, output, stage=stage: check(output, stage + ".output")
            ))
        yield
    finally:
        for handle in handles:
            handle.remove()


def capture_decode_projection(model, projection, encoded):
    """Run prefill, then capture this layer's input/output for one decoded token."""
    import torch

    captured = []

    def capture(module, inputs, output):
        captured.append((inputs[0].detach().clone(), output.detach().clone()))

    with torch.inference_mode():
        with trace_finite_modules(model, "prefill"):
            prefill = model(**encoded, use_cache=True, return_dict=True)
        require_finite(prefill.logits, "prefill.logits")
        token = prefill.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        mask = torch.cat((encoded["attention_mask"], torch.ones_like(token)), dim=1)
        # Register only after prefill: the captured input must be a single token.
        handle = projection.register_forward_hook(capture)
        try:
            with trace_finite_modules(model, "decode"):
                decode = model(
                    input_ids=token,
                    attention_mask=mask,
                    past_key_values=prefill.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            require_finite(decode.logits, "decode.logits")
        finally:
            handle.remove()

    if len(captured) != 1 or captured[0][0].shape[:2] != (1, 1):
        raise RuntimeError("expected exactly one [batch=1, token=1, hidden] capture")
    return captured[0][0], captured[0][1], token.item()


def time_pair(a, weight, helpers, gemv, round_index):
    import torch
    import torch.nn.functional as F

    operations = {
        "pytorch_linear": lambda x, w: F.linear(x, w),
        "gemv_rows1": lambda x, w: gemv(x, w, block_rows=1, num_warps=4),
    }
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            for operation in operations.values():
                operation(a, weight)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graphs, outputs = {}, {}
    names = list(operations)
    for name in (names if round_index % 2 == 0 else names[::-1]):
        graphs[name], outputs[name] = helpers["capture_graph"](operations[name], a, weight, 100)
    for _ in range(10):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in names}
    for repetition in range(30):
        order = names if (repetition + round_index) % 2 == 0 else names[::-1]
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 100)
    return {name: helpers["summarize"](values) for name, values in samples.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from fastinfer.kernels.triton_gemv import gemv

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    helpers = runpy.run_path(str(ROOT / "benchmarks/matmul_compare.py"))

    config = AutoConfig.from_pretrained(MODEL, revision=args.revision, trust_remote_code=False)
    revision = config._commit_hash
    if not revision:
        raise RuntimeError("could not resolve model revision for reproducibility")
    print(f"Loading {MODEL} revision {revision} in FP16...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=revision,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=False,
        attn_implementation="eager",
    ).to("cuda").eval()
    projection = model.model.layers[0].mlp.up_proj
    weight = projection.weight.detach()
    if weight.shape != (8960, 1536) or weight.dtype != torch.float16:
        raise RuntimeError(f"unexpected projection weights: {weight.shape}, {weight.dtype}")
    if projection.bias is not None or not weight.is_contiguous():
        raise RuntimeError("expected bias-free projection with contiguous stored weights")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output or ROOT / "results/raw" / f"qwen_real_linear_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "qwen_captured_decode_linear",
        "model": MODEL,
        "resolved_revision": revision,
        "projection": "model.layers.0.mlp.up_proj",
        "model_dtype": "float16",
        "checkpoint_config_dtype": str(config.torch_dtype),
        "attention_implementation": "eager",
        "shape": {"m": 1, "k": 1536, "n": 8960},
        "weight_stride": list(weight.stride()),
        "tile": {"block_rows": 1, "num_warps": 4},
        "allow_fp16_reduced_precision_reduction": True,
        "timing_mode": "cuda_graph_replay_amortized",
        "graph_ops": 100, "warmup": 10, "graph_warmup": 10, "repetitions": 30,
        "measurement_order": "alternating_pytorch_gemv_pairs",
        "accuracy_reference": "float32_matmul_of_loaded_fp16_weights_and_activation",
        "rtol": .01, "atol": .01,
        "scope": "Isolated projection replay on real weights and captured decode inputs. "
                 "Model stays resident. Model loading, prefill, decode and capture excluded "
                 "from timing. No model replacement or end-to-end speedup measured.",
        "kernel_sha256": hashlib.sha256((ROOT / "kernels/triton_gemv.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "cases": [],
    }
    with torch.inference_mode():
        for index, prompt in enumerate(PROMPTS):
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda")
            try:
                require_finite(weight, "projection.weight")
                activation, model_output, token = capture_decode_projection(model, projection, encoded)
                require_finite(activation, "captured_activation")
                require_finite(model_output, "captured_model_output")
                a = activation.reshape(1, -1).contiguous()
                captured_output = model_output.reshape(1, -1)
                reference = F.linear(a.float(), weight.float())
                require_finite(reference, "float32_reference")
            except NonfiniteTensorError as error:
                report["cases"].append({
                    "prompt": prompt,
                    "status": "nonfinite_model_or_reference",
                    "diagnostics": error.details,
                    "timing_skipped": True,
                })
                destination.write_text(json.dumps(report, indent=2) + "\n")
                print(f"Diagnosis saved: {destination}", flush=True)
                raise
            actual = gemv(a, weight, block_rows=1, num_warps=4)
            pytorch_output = F.linear(a, weight)
            accuracy = {
                name: helpers["accuracy_against_fp32"](value, reference)
                for name, value in (
                    ("captured_model_output", captured_output),
                    ("pytorch_linear", pytorch_output),
                    ("gemv_rows1", actual),
                )
            }
            capture_match = torch.allclose(pytorch_output, captured_output, rtol=.01, atol=.01)
            passed = capture_match and all(v["mismatched_elements"] == 0 for v in accuracy.values())
            case = {
                "prompt": prompt, "prompt_tokens": encoded.input_ids.shape[1],
                "decoded_token_id": token, "captured_input_shape": list(activation.shape),
                "input_stride": list(a.stride()), "accuracy": accuracy,
                "replay_matches_captured_output": capture_match,
                "all_accuracy_checks_passed": passed,
            }
            report["cases"].append(case)
            print(f"\nPrompt {index + 1}: {prompt!r}", flush=True)
            for name, values in accuracy.items():
                print(f"  {name}: {values['mismatched_elements']} mismatches, "
                      f"max error {values['max_abs_error']:.6f}", flush=True)
            if passed:
                del reference, actual, pytorch_output
                torch.cuda.synchronize()
                case["implementations"] = time_pair(a, weight, helpers, gemv, index)
                pt = case["implementations"]["pytorch_linear"]["latency_us"]["median"]
                tt = case["implementations"]["gemv_rows1"]["latency_us"]["median"]
                print(f"  PyTorch {pt:.2f} us | GEMV {tt:.2f} us | PT/GEMV {pt / tt:.3f}x", flush=True)
            else:
                print("  Timing skipped: accuracy check failed; see saved diagnostics.", flush=True)
            destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
