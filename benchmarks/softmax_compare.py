"""Compare readable, native PyTorch, and Triton softmax on fixed synthetic rows."""

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import permutations
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((1, 65), (1, 128), (16, 128), (128, 128), (16, 1024),
          (128, 1024), (1, 1536), (128, 1536), (1, 8192))
COMPILED_SETTINGS = {
    "backend": "inductor", "fullgraph": True, "dynamic": False,
    "options": {"triton.cudagraphs": False},
}


def compile_reference():
    """Compile our PyTorch reference, independently of the custom Triton kernel."""
    from fastinfer.kernels.softmax import reference_softmax

    # Nine shapes exceed the default per-function recompilation limit in 2.5.1.
    # Prior cases have finished; reset capture state before compiling this case.
    torch.compiler.reset()
    return torch.compile(reference_softmax, **COMPILED_SETTINGS)


def measurement_orders(names, repetitions=30):
    """Keep the original three-provider order; balance four-provider positions."""
    names = tuple(names)
    orders = list(permutations(names))
    if len(names) == 4:
        rotations = [names[index:] + names[:index] for index in range(4)]
        preferred = rotations + [tuple(reversed(order)) for order in rotations]
        orders = preferred + [order for order in orders if order not in preferred]
    return [orders[index % len(orders)] for index in range(repetitions)]


def native_triton_comparison(native, triton):
    """Positive paired savings mean Triton took less time in the same round."""
    if not native or len(native) != len(triton) or min([*native, *triton]) <= 0:
        raise ValueError("positive, nonempty paired samples of equal length required")
    differences = [a - b for a, b in zip(native, triton)]
    return {
        "ratio_of_separate_medians": statistics.median(native) / statistics.median(triton),
        "paired_saving_us_samples": differences,
        "paired_median_saving_us": statistics.median(differences),
        "triton_faster_rounds": sum(delta > 0 for delta in differences),
        "tied_rounds": sum(delta == 0 for delta in differences),
        "rounds": len(differences),
    }


def measure_case(x, *, include_compiled=False):
    import triton
    from fastinfer.kernels.softmax import reference_softmax
    from fastinfer.kernels.triton_softmax import softmax, _softmax
    from matmul_compare import capture_graph, summarize

    operations = {
        "readable_reference": lambda a, unused: reference_softmax(a),
        "pytorch_native": lambda a, unused: torch.softmax(a, dim=-1),
    }
    if include_compiled:
        compiled = compile_reference()
        operations["compiled_reference"] = lambda a, unused: compiled(a)
    operations["triton"] = lambda a, unused: softmax(a)
    rtol, atol = (2e-3, 2e-6) if x.dtype == torch.float16 else (1e-5, 1e-7)
    oracle = torch.softmax(x.double(), dim=-1).float()
    native = operations["pytorch_native"](x, None)
    row = {"shape": list(x.shape), "accuracy": {"rtol": rtol, "atol": atol,
            "oracle": "float64_softmax_of_same_input_values", "implementations": {}}}
    row["first_call_setup_and_execution_seconds"] = {}
    for name, operation in operations.items():
        print(f"  Preparing {name}...", flush=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = operation(x, None)
        torch.cuda.synchronize()
        row["first_call_setup_and_execution_seconds"][name] = time.perf_counter() - started
        torch.testing.assert_close(output, native, rtol=rtol, atol=atol)
        torch.testing.assert_close(output.float(), oracle, rtol=rtol, atol=atol)
        if not torch.isfinite(output).all().item() or not (output >= 0).all().item():
            raise RuntimeError(f"Nonfinite or negative probabilities from {name}")
        torch.testing.assert_close(output.float().sum(dim=-1),
                                   torch.ones(x.shape[0], device=x.device), rtol=0,
                                   atol=5e-4 if x.dtype == torch.float16 else 1e-6)
        error = (output.float() - oracle).abs()
        row["accuracy"]["implementations"][name] = {
            "passed": True, "max_abs_oracle_error": error.max().item(),
            "different_elements_from_native": int((output != native).sum().item()),
            "oracle_mismatches": int((error > atol + rtol * oracle.abs()).sum().item()),
        }

    resource_output = torch.empty_like(x)
    handle = _softmax[(x.shape[0],)](
        x, resource_output, x.shape[1], triton.next_power_of_2(x.shape[1]),
        num_warps=4, enable_fp_fusion=False,
    )
    row["triton_resources"] = {
        "programs": x.shape[0], "block": triton.next_power_of_2(x.shape[1]),
        "num_warps": 4, "registers_per_thread": handle.n_regs,
        "spills": handle.n_spills, "shared_bytes": handle.metadata.shared,
    }

    # The shared capture helper accepts two tensors; softmax uses only its first.
    unused = torch.empty(0, device=x.device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            for operation in operations.values():
                operation(x, unused)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graphs, outputs = {}, {}
    for name, operation in operations.items():
        graphs[name], outputs[name] = capture_graph(operation, x, unused, 100)
    for _ in range(10):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()

    orders = measurement_orders(operations)
    samples = {name: [] for name in operations}
    for repetition in range(30):
        for name in orders[repetition]:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 100)
    row["implementations"] = {name: summarize(values) for name, values in samples.items()}
    row["native_vs_triton"] = native_triton_comparison(samples["pytorch_native"], samples["triton"])
    if include_compiled:
        row["compiled_vs_triton"] = native_triton_comparison(samples["compiled_reference"], samples["triton"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--include-compiled", action="store_true",
                        help="also measure torch.compile of our readable PyTorch reference")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prefix = "softmax_compiled_comparison" if args.include_compiled else "softmax_comparison"
    destination = ROOT / "results/raw" / f"{prefix}_{args.dtype}_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "readable_vs_native_vs_triton_softmax", "dtype": args.dtype,
        "seed": 2, "shapes": [list(shape) for shape in SHAPES],
        "timing_mode": "cuda_graph_replay_amortized", "operations_per_sample": 100,
        "warmup": 10, "graph_warmup": 10, "repetitions": 30,
        "order": ("balanced_four_provider_orders_30_rounds" if args.include_compiled
                  else "all_six_implementation_orders_repeated_five_times"),
        "compiled_settings": COMPILED_SETTINGS if args.include_compiled else None,
        "compiler_reset_between_shapes": args.include_compiled,
        "scope": "Fixed synthetic input per case, one program per row, four warps. "
                 "Compilation, accuracy checks, and warmup excluded. Native torch.softmax "
                 "is the optimized baseline; the readable reference is separate. "
                 "Optional compiled reference uses default Inductor settings, independently "
                 "of our kernel, with internal CUDA graphs disabled. Fullgraph errors stop "
                 "the run; no eager fallback. Manual kernel stays at four warps. "
                 "No model integration, tuning, or exact-output guarantee.",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("kernels/softmax.py", "kernels/triton_softmax.py",
                                       "benchmarks/softmax_compare.py", "benchmarks/matmul_compare.py")},
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "cases": [],
    }
    try:
        with torch.inference_mode():
            torch.manual_seed(2)
            for shape in SHAPES:
                x = torch.randn(shape, device="cuda", dtype=getattr(torch, args.dtype))
                print(f"Preparing rows={shape[0]}, columns={shape[1]}...", flush=True)
                case = measure_case(x, include_compiled=args.include_compiled)
                report["cases"].append(case)
                medians = {name: value["latency_us"]["median"]
                           for name, value in case["implementations"].items()}
                paired = case["native_vs_triton"]
                compiled_text = ""
                if args.include_compiled:
                    comparison = case["compiled_vs_triton"]
                    compiled_text = (
                        f"compiled reference {medians['compiled_reference']:.3f} us | "
                        f"compiled/Triton {comparison['ratio_of_separate_medians']:.3f}x | "
                        f"compiled-minus-Triton paired saving {comparison['paired_median_saving_us']:+.3f} us | "
                        f"Triton faster than compiled {comparison['triton_faster_rounds']}/30 | "
                    )
                print(f"  Reference {medians['readable_reference']:.3f} us | "
                      f"native PyTorch {medians['pytorch_native']:.3f} us | "
                      f"Triton {medians['triton']:.3f} us | "
                      f"native/Triton {paired['ratio_of_separate_medians']:.3f}x | "
                      f"paired saving {paired['paired_median_saving_us']:+.3f} us | "
                      f"Triton faster {paired['triton_faster_rounds']}/30 | "
                      + compiled_text + "accuracy PASS", flush=True)
                destination.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        report["failure"] = {"shape": list(shape), "type": type(error).__name__, "message": str(error)}
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
