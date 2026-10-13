"""Paired CUDA-graph comparison of fused RMSNorm and the Qwen-style reference."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
from fastinfer.kernels.rmsnorm import reference_rmsnorm

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((1, 1536), (16, 1536), (128, 1536), (1, 65),
          (1, 1024), (1, 4096), (1, 8192))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-compiled", action="store_true",
                        help="also benchmark torch.compile of the same PyTorch reference")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    from fastinfer.kernels.triton_rmsnorm import rmsnorm
    from matmul_compare import capture_graph, summarize

    torch.manual_seed(2)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"rmsnorm_{'compiled_comparison' if args.include_compiled else 'first_comparison'}_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "qwen_style_pytorch_vs_fused_rmsnorm", "dtype": "float16",
        "seed": 2, "eps": 1e-6, "rtol": .002, "atol": .002,
        "kernel": {"programs": "one per token", "num_warps": 4, "enable_fp_fusion": False},
        "timing": "cuda_graph_replay_amortized", "operations_per_sample": 100,
        "repetitions": 30, "warmup": 10, "graph_warmup": 10,
        "order": "rotating_pytorch_compiled_triton" if args.include_compiled else "alternating_pytorch_triton",
        "compiled_settings": ({"backend": "inductor", "fullgraph": True, "dynamic": False,
                               "options": {"triton.cudagraphs": False},
                               "first_call_excluded_from_timing": True}
                              if args.include_compiled else None), "cases": [],
        "scope": "Synthetic isolated RMSNorm; PyTorch reference matches Qwen arithmetic "
                 "order. Optional compiled reference uses default Inductor optimization settings, "
                 "with internal CUDA graphs disabled. Not native BF16 or full model performance.",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("kernels/rmsnorm.py", "kernels/triton_rmsnorm.py",
                                       "benchmarks/rmsnorm_compare.py")},
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
    }
    operations = {"pytorch": reference_rmsnorm}
    if args.include_compiled:
        operations["pytorch_compiled"] = torch.compile(
            reference_rmsnorm, backend="inductor", fullgraph=True, dynamic=False,
            options={"triton.cudagraphs": False},
        )
    operations["triton"] = rmsnorm
    names = list(operations)
    with torch.inference_mode():
        for index, shape in enumerate(SHAPES):
            x = torch.randn(shape, device="cuda", dtype=torch.float16)
            weight = torch.randn(shape[-1], device="cuda", dtype=torch.float16)
            values, setup_seconds = {}, {}
            for name, operation in operations.items():
                print(f"Preparing {name}: tokens={shape[0]}, hidden={shape[1]}...", flush=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                try:
                    values[name] = operation(x, weight)
                    torch.cuda.synchronize()
                except Exception as error:
                    report["failure"] = {"shape": list(shape), "implementation": name,
                                         "type": type(error).__name__, "message": str(error)}
                    destination.write_text(json.dumps(report, indent=2) + "\n")
                    print(f"Failure saved: {destination}", flush=True)
                    raise
                setup_seconds[name] = time.perf_counter() - started
            expected = values["pytorch"]
            oracle = reference_rmsnorm(x.float(), weight.float())
            assert torch.isfinite(oracle).all().item(), "nonfinite FP32 reference"
            accuracy = {}
            for name, value in values.items():
                error = (value.float() - oracle).abs()
                accuracy[name] = {
                    "fp32_mismatches": int(((error > .002 + .002 * oracle.abs())
                                            | ~torch.isfinite(value)).sum().item()),
                    "max_abs_fp32_error": error.max().item(),
                    "max_abs_pytorch_difference": (value.float()-expected.float()).abs().max().item(),
                }
            case = {"shape": list(shape), "accuracy": accuracy,
                    "first_call_setup_and_execution_seconds": setup_seconds,
                    "max_abs_pytorch_triton_difference": accuracy["triton"]["max_abs_pytorch_difference"]}
            report["cases"].append(case)
            try:
                for name, value in values.items():
                    torch.testing.assert_close(value.float(), oracle, rtol=.002, atol=.002)
                    torch.testing.assert_close(value, expected, rtol=.002, atol=.002)
            except AssertionError as error:
                case["accuracy_passed"] = False
                case["timing_skipped"] = True
                case["accuracy_failure"] = str(error)
                destination.write_text(json.dumps(report, indent=2) + "\n")
                print(f"Accuracy failure saved: {destination}", flush=True)
                raise
            case["accuracy_passed"] = True
            del values, expected, oracle, value
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(10):
                    for operation in operations.values():
                        operation(x, weight)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graphs, outputs = {}, {}
            capture_order = (names[index % len(names):] + names[:index % len(names)]
                             if args.include_compiled else names if index % 2 == 0 else names[::-1])
            for name in capture_order:
                graphs[name], outputs[name] = capture_graph(operations[name], x, weight, 100)
            for _ in range(10):
                for graph in graphs.values():
                    graph.replay()
            torch.cuda.synchronize()
            samples = {name: [] for name in names}
            for repetition in range(30):
                if args.include_compiled:
                    offset = (index + repetition) % len(names)
                    order = names[offset:] + names[:offset]
                else:
                    order = names if (index + repetition) % 2 == 0 else names[::-1]
                for name in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    graphs[name].replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) * 1000 / 100)
            case["implementations"] = {name: summarize(values) for name, values in samples.items()}
            pt = case["implementations"]["pytorch"]["latency_us"]["median"]
            tt = case["implementations"]["triton"]["latency_us"]["median"]
            case["speedup"] = pt / tt
            compiled_text = ""
            if args.include_compiled:
                compiled = case["implementations"]["pytorch_compiled"]["latency_us"]["median"]
                case["compiled_over_triton_speedup"] = compiled / tt
                compiled_text = (f"Compiled PyTorch {compiled:.2f} us | "
                                 f"Compiled/Triton {compiled/tt:.3f}x | ")
            print(f"Tokens={shape[0]}, hidden={shape[1]} | PyTorch {pt:.2f} us | "
                  f"Triton {tt:.2f} us | PT/Triton {pt/tt:.3f}x | "
                  + compiled_text + "FP32 and PyTorch comparisons PASS", flush=True)
            destination.write_text(json.dumps(report, indent=2) + "\n")
            del graphs, outputs
    print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
