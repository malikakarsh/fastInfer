"""Compare eager/compiled PyTorch and fused Triton SiLU(gate) * up."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from softmax_compare import COMPILED_SETTINGS, measurement_orders, native_triton_comparison
from matmul_compare import capture_graph, summarize

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((1, 65), (1, 1536), (1, 8960), (16, 8960), (128, 8960))


def measure_case(gate, up, *, include_compiled):
    import triton
    from fastinfer.kernels.swiglu import reference_swiglu
    from fastinfer.kernels.triton_swiglu import swiglu, _swiglu, BLOCK_SIZE, NUM_WARPS

    operations = {"pytorch_native": reference_swiglu}
    if include_compiled:
        torch.compiler.reset()
        operations["pytorch_compiled"] = torch.compile(reference_swiglu, **COMPILED_SETTINGS)
    operations["triton"] = swiglu
    native = reference_swiglu(gate, up)
    # The oracle includes the SiLU intermediate's input-dtype rounding.
    activated = (gate.double() * torch.sigmoid(gate.double())).to(gate.dtype)
    oracle = (activated.double() * up.double()).to(gate.dtype)
    rtol, atol = (.002, 2e-4) if gate.dtype == torch.float16 else (1e-5, 2e-6)
    case = {"shape": list(gate.shape), "elements": gate.numel(),
            "accuracy": {"rtol": rtol, "atol": atol,
                         "oracle": "float64 arithmetic with input-dtype rounding after SiLU and multiply",
                         "implementations": {}}, "first_call_setup_and_execution_seconds": {}}
    for name, operation in operations.items():
        print(f"  Preparing {name}...", flush=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = operation(gate, up)
        torch.cuda.synchronize()
        case["first_call_setup_and_execution_seconds"][name] = time.perf_counter() - start
        for expected in (native, oracle):
            torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)
        if not torch.isfinite(output).all().item():
            raise RuntimeError(f"Nonfinite output from {name}")
        error = (output.double() - oracle.double()).abs()
        case["accuracy"]["implementations"][name] = {
            "passed": True, "max_abs_oracle_error": error.max().item(),
            "different_elements_from_native": int((output != native).sum().item()),
        }

    resource_output = torch.empty_like(gate)
    programs = triton.cdiv(gate.numel(), BLOCK_SIZE)
    handle = _swiglu[(programs,)](
        gate, up, resource_output, gate.numel(), BLOCK_SIZE,
        num_warps=NUM_WARPS, enable_fp_fusion=False,
    )
    torch.testing.assert_close(resource_output, swiglu(gate, up), rtol=0, atol=0)
    case["triton_resources"] = {
        "programs": programs, "block_size": BLOCK_SIZE, "num_warps": NUM_WARPS,
        "registers_per_thread": handle.n_regs, "spills": handle.n_spills,
        "shared_bytes": handle.metadata.shared,
    }
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            for operation in operations.values():
                operation(gate, up)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graphs, outputs = {}, {}
    for name, operation in operations.items():
        graphs[name], outputs[name] = capture_graph(operation, gate, up, 100)
    for _ in range(10):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    orders = measurement_orders(operations)
    samples = {name: [] for name in operations}
    for order in orders:
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 100)
    case["measurement_orders"] = orders
    case["implementations"] = {name: summarize(values) for name, values in samples.items()}
    case["native_vs_triton"] = native_triton_comparison(samples["pytorch_native"], samples["triton"])
    if include_compiled:
        case["compiled_vs_triton"] = native_triton_comparison(samples["pytorch_compiled"], samples["triton"])
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-compiled", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"swiglu_comparison_{args.dtype}_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "elementwise_swiglu_comparison", "dtype": args.dtype,
        "seed": 2, "input_distribution": "gate=2*randn, up=randn",
        "shapes": [list(shape) for shape in SHAPES],
        "timing_mode": "cuda_graph_replay_amortized", "operations_per_sample": 100,
        "warmup": 10, "graph_warmup": 10, "repetitions": 30,
        "compiled_settings": COMPILED_SETTINGS if args.include_compiled else None,
        "scope": "Synthetic elementwise stage only; no learned projections or model integration. "
                 "Native means F.silu(gate)*up, not a built-in fused SwiGLU. "
                 "Compiled reference is independent of our Triton kernel. Setup/compilation excluded. "
                 "Accuracy against eager PyTorch and staged FP64 oracle precedes timing; no eager fallback. "
                 "Fixed block256/warps4, no tuning or exact-output claim. "
                 "Spills is compiler local-storage metadata, not runtime transaction count.",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("kernels/swiglu.py", "kernels/triton_swiglu.py",
                                       "benchmarks/swiglu_compare.py", "benchmarks/softmax_compare.py",
                                       "benchmarks/matmul_compare.py")},
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "cases": [],
    }
    shape = None
    try:
        with torch.inference_mode():
            torch.manual_seed(2)
            for shape in SHAPES:
                print(f"Preparing shape={shape}...", flush=True)
                gate = 2 * torch.randn(shape, device="cuda", dtype=getattr(torch, args.dtype))
                up = torch.randn_like(gate)
                case = measure_case(gate, up, include_compiled=args.include_compiled)
                report["cases"].append(case)
                for name, values in case["implementations"].items():
                    print(f"  {name}: {values['latency_us']['median']:.3f} us", flush=True)
                for key in ("native_vs_triton", "compiled_vs_triton"):
                    if key in case:
                        row = case[key]
                        print(f"  {key}: {row['ratio_of_separate_medians']:.3f}x | paired saving "
                              f"{row['paired_median_saving_us']:+.3f} us | "
                              f"Triton faster {row['triton_faster_rounds']}/30", flush=True)
                print(f"  Resources: {case['triton_resources']} | accuracy PASS", flush=True)
                destination.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        report["failure"] = {"shape": shape, "type": type(error).__name__, "message": str(error)}
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
