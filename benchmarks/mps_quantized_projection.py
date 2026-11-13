"""Isolate NF4 reconstruction and projection costs; no full model or generation.

Reads only selected real checkpoint tensors. Uses controlled synthetic inputs,
not captured model activations. User-run MPS timings; cached files only.
"""

import argparse
from collections import Counter
from contextlib import ExitStack
import copy
from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import statistics
import time
from unittest.mock import patch

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def read_projection(snapshot, module):
    """Read one projection and its quantization state, including sharded files."""
    from safetensors import safe_open

    snapshot = Path(snapshot)
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
        files = sorted({snapshot / filename for key, filename in mapping.items()
                        if key.startswith(module + ".")})
    else:
        files = sorted(snapshot.glob("*.safetensors"))
    result = {}
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(module + "."):
                    short = key[len(module) + 1:]
                    if short in result:
                        raise ValueError(f"duplicate tensor: {key}")
                    result[short] = handle.get_tensor(key)
    if "weight" not in result:
        raise ValueError(f"projection {module} not found in {snapshot}")
    return result


def make_quantized_layer(tensors, bnb, device):
    stats = {key[len("weight."):]: value for key, value in tensors.items()
             if key.startswith("weight.")}
    state = bnb.functional.QuantState.from_dict(dict(stats), device="cpu")
    if state.quant_type != "nf4" or len(state.shape) != 2:
        raise ValueError("expected packed NF4 linear weights")
    out_features, in_features = state.shape
    layer = bnb.nn.Linear4bit(in_features, out_features, bias="bias" in tensors,
                             compute_dtype=torch.float16, quant_type="nf4", device="meta")
    layer.weight = bnb.nn.Params4bit.from_prequantized(
        tensors["weight"], quantized_stats=dict(stats), device=device, module=layer)
    if "bias" in tensors:
        layer.bias = torch.nn.Parameter(tensors["bias"].to(device=device, dtype=torch.float16),
                                        requires_grad=False)
    return layer.eval()


def difference(actual, expected):
    # CPU conversion precedes FP64: MPS does not support FP64 tensors.
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    if a.shape != b.shape:
        raise ValueError("output shapes differ")
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {"finite": finite, "different_values": int((a != b).sum()),
            "max_abs": float((a - b).abs().max()) if finite else None,
            "within_tolerance": finite and bool(torch.allclose(a, b, rtol=0.02, atol=0.02))}


def classify_path(counts):
    if counts.get("hub_gemv", 0):
        return "specialized_hub_gemv_observed"
    if counts.get("weight_reconstruction", 0) and counts.get("linear", 0):
        return "weight_reconstruction_then_linear_observed"
    return "unresolved"


def trace_dispatch(operation, backend, synchronize):
    """Instrument a single untimed call, then restore every wrapper."""
    required = ("_get_kernel", "_dequantize_4bit_impl")
    if not all(hasattr(backend, name) for name in required):
        return {"available": False, "reason": "installed backend internals differ"}, None
    counts = Counter()
    original_get = backend._get_kernel
    original_dequantize = backend._dequantize_4bit_impl
    original_linear = F.linear

    class HubProxy:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getattr__(self, name):
            item = getattr(self.kernel, name)
            if name not in ("gemv_4bit", "dequantize_4bit"):
                return item

            def invoke(*args, **kwargs):
                counts["hub_gemv" if name == "gemv_4bit" else "hub_dequantize"] += 1
                return item(*args, **kwargs)
            return invoke

    def get_kernel():
        counts["hub_kernel_lookup"] += 1
        kernel = original_get()
        if kernel is None:
            counts["hub_kernel_unavailable"] += 1
            return None
        return HubProxy(kernel)

    def reconstruct(*args, **kwargs):
        counts["weight_reconstruction"] += 1
        return original_dequantize(*args, **kwargs)

    def linear(*args, **kwargs):
        counts["linear"] += 1
        return original_linear(*args, **kwargs)

    synchronize()
    with ExitStack() as stack:
        stack.enter_context(patch.object(backend, "_get_kernel", get_kernel))
        stack.enter_context(patch.object(backend, "_dequantize_4bit_impl", reconstruct))
        stack.enter_context(patch.object(F, "linear", linear))
        output = operation()
        synchronize()
    return {"available": True, "counts": dict(counts), "path": classify_path(counts)}, output


def measure_operations(operations, synchronize, warmup, samples, ops_per_sample):
    """Rotate variant order and time completed work; no trace wrappers in timing."""
    names = list(operations)
    for name in names:
        for _ in range(warmup):
            output = operations[name]()
            del output
    synchronize()
    result = {name: [] for name in names}
    orders = []
    for sample in range(samples):
        offset = sample % len(names)
        order = names[offset:] + names[:offset]
        orders.append(order)
        for name in order:
            synchronize()
            start = time.perf_counter()
            for _ in range(ops_per_sample):
                output = operations[name]()
                del output
            synchronize()
            result[name].append((time.perf_counter() - start) * 1000 / ops_per_sample)
    return {"samples_ms_per_call": result, "orders": orders,
            "median_ms_per_call": {name: statistics.median(values) for name, values in result.items()}}


def comparison_ratios(medians):
    return {
        "normal_4bit_over_predecoded": medians["normal_4bit"] / medians["predecoded_fp16"],
        "reconstruct_each_over_predecoded": medians["reconstruct_each_call"] / medians["predecoded_fp16"],
        "normal_4bit_over_original_fp16": medians["normal_4bit"] / medians["original_fp16"],
        "reconstruction_only_over_reconstruct_each": medians["reconstruction_only"] / medians["reconstruct_each_call"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="unsloth/Qwen2.5-7B-bnb-4bit")
    parser.add_argument("--original", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--module", action="append", help="repeat for multiple projections")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--ops-per-sample", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.rows + [args.warmup, args.samples, args.ops_per_sample]) < 1:
        parser.error("rows, warmup, samples and operations per sample must be positive")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        parser.error("unset PYTORCH_ENABLE_MPS_FALLBACK to avoid CPU fallback in MPS timing")
    if not torch.backends.mps.is_available():
        parser.error("MPS unavailable; run from your normal Apple Silicon terminal")
    import bitsandbytes as bnb
    from huggingface_hub import snapshot_download

    backend = importlib.import_module("bitsandbytes.backends.mps.ops")
    def resolve(identifier):
        path = Path(identifier).expanduser()
        return path.resolve() if path.is_dir() else Path(snapshot_download(identifier, local_files_only=True)).resolve()

    device = torch.device("mps")
    target, original = resolve(args.target), resolve(args.original)
    modules = args.module or ["model.layers.0.self_attn.q_proj"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output or ROOT / "results/raw" / f"mps_quantized_projection_{stamp}.json"
    if destination.exists():
        parser.error(f"output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    backend_file = Path(backend.__file__)
    report = {
        "status": "running", "created_utc": stamp,
        "environment": {"torch": torch.__version__, "python": platform.python_version(),
                        "platform": platform.platform(), "bitsandbytes": package_version("bitsandbytes"),
                        "kernels": package_version("kernels"), "device": "mps", "dtype": "float16",
                        "cpu_threads": torch.get_num_threads(),
                        "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"]},
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "target_snapshot": str(target), "original_snapshot": str(original),
        "backend_file": str(backend_file),
        "backend_sha256": hashlib.sha256(backend_file.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Real isolated projection weights, controlled synthetic inputs; no full model, KV cache, "
                 "draft or generation. Synchronized wall time includes launch/allocation overhead. "
                 "Offline backend availability may differ from the earlier run. No whole-model causal attribution.",
        "cases": [],
    }

    def save():
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(destination)

    try:
        with torch.inference_mode():
            for module in modules:
                print(f"Loading projection only: {module}", flush=True)
                layer = make_quantized_layer(read_projection(target, module), bnb, device)
                original_tensors = read_projection(original, module)
                w_original = original_tensors["weight"].to(device=device, dtype=torch.float16)
                b_original = original_tensors.get("bias")
                if b_original is not None:
                    b_original = b_original.to(device=device, dtype=torch.float16)
                if tuple(w_original.shape) != tuple(layer.weight.quant_state.shape):
                    raise ValueError("original and quantized projection dimensions differ")
                # Match the backend's FP16 reconstruction output, avoiding BF16→FP16 double rounding.
                state = copy.copy(layer.weight.quant_state)
                state.dtype = torch.float16

                def reconstruct():
                    return bnb.functional.dequantize_4bit(layer.weight.data, quant_state=state)

                w_decoded = reconstruct()
                generator = torch.Generator(device="cpu").manual_seed(0)
                inputs = torch.randn(max(args.rows), layer.in_features, generator=generator).half().to(device)
                for rows in args.rows:
                    x = inputs[:rows].unsqueeze(0).contiguous()
                    operations = {
                        "normal_4bit": lambda: layer(x),
                        "reconstruct_each_call": lambda: F.linear(x, reconstruct(), layer.bias),
                        "predecoded_fp16": lambda: F.linear(x, w_decoded, layer.bias),
                        "original_fp16": lambda: F.linear(x, w_original, b_original),
                        "reconstruction_only": reconstruct,
                    }
                    # Initialize lazy kernels before tracing; all instrumentation is outside timing.
                    reference = operations["predecoded_fp16"]()
                    normal = operations["normal_4bit"]()
                    rebuilt = operations["reconstruct_each_call"]()
                    trace, traced_output = trace_dispatch(operations["normal_4bit"], backend, torch.mps.synchronize)
                    controls = {"normal_vs_predecoded": difference(normal, reference),
                                "reconstruct_each_vs_predecoded": difference(rebuilt, reference),
                                "normal_repeat": difference(operations["normal_4bit"](), normal)}
                    if traced_output is not None:
                        controls["trace_vs_normal"] = difference(traced_output, normal)
                    case = {"module": module, "rows": rows, "input_shape": list(x.shape),
                            "weight_shape": list(w_original.shape),
                            "packed_weight_bytes": layer.weight.numel() * layer.weight.element_size(),
                            "decoded_weight_bytes": w_decoded.numel() * w_decoded.element_size(),
                            "quant_type": state.quant_type, "blocksize": state.blocksize,
                            "double_quantization": state.nested,
                            "checkpoint_reconstruction_dtype": str(layer.weight.quant_state.dtype),
                            "compute_and_explicit_reconstruction_dtype": "torch.float16",
                            "dispatch_trace": trace, "controls": controls}
                    report["cases"].append(case)
                    case["controls_passed"] = all(item["within_tolerance"] for item in controls.values())
                    case["controls_passed"] &= controls["normal_repeat"]["different_values"] == 0
                    if "trace_vs_normal" in controls:
                        case["controls_passed"] &= controls["trace_vs_normal"]["different_values"] == 0
                    if not case["controls_passed"]:
                        print(f"{module} rows={rows}: numerical controls FAILED; timing skipped", flush=True)
                        save()
                        continue
                    case["timing"] = measure_operations(operations, torch.mps.synchronize,
                                                        args.warmup, args.samples, args.ops_per_sample)
                    medians = case["timing"]["median_ms_per_call"]
                    case["ratios"] = comparison_ratios(medians)
                    print(f"{module} rows={rows}: {trace.get('path', 'unresolved')}", flush=True)
                    for name, value in medians.items():
                        print(f"  {name}: {value:.4f} ms/call", flush=True)
                    save()
                del layer, w_original, w_decoded, inputs, original_tensors
            report["status"] = "complete" if all(c["controls_passed"] for c in report["cases"]) else "controls_failed"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
        print(f"Saved {destination}", flush=True)
    if report["status"] != "complete":
        raise SystemExit("Projection controls failed; inspect the saved report")


if __name__ == "__main__":
    main()
