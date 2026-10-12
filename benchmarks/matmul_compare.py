"""Compare PyTorch and Triton float16 matmul with paired CUDA graph timings."""

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

from fastinfer.kernels.matmul import reference_matmul
from fastinfer.kernels.triton_matmul import triton_matmul


REPO_ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def capture_graph(
    operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    graph_ops: int,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_ops):
            output = operation(a, b)
    return graph, output


def summarize(samples_us: list[float]) -> dict:
    ordered = sorted(samples_us)
    return {
        "latency_us": {
            "median": statistics.median(ordered),
            "p95": ordered[(95 * len(ordered) - 1) // 100],
            "min": ordered[0],
            "max": ordered[-1],
        },
        "samples_us": samples_us,
    }


def accuracy_against_fp32(output: torch.Tensor, reference: torch.Tensor) -> dict:
    difference = (output.float() - reference).abs()
    allowed = 1e-2 + 1e-2 * reference.abs()
    return {
        "mismatched_elements": int((difference > allowed).sum().item()),
        "total_elements": reference.numel(),
        "max_abs_error": difference.max().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=positive_int, default=1)
    parser.add_argument("--k", type=positive_int, default=256)
    parser.add_argument("--n", type=positive_int, default=256)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--repetitions", type=positive_int, default=30)
    parser.add_argument("--graph-ops", type=positive_int, default=100)
    sweep = parser.add_mutually_exclusive_group()
    sweep.add_argument(
        "--block-k-sweep", action="store_true",
        help="compare the existing BLOCK_K=32 kernel with BLOCK_K=64 on the same inputs",
    )
    sweep.add_argument(
        "--block-n-sweep", action="store_true",
        help="compare BLOCK_N=32 and 64 while keeping BLOCK_K=64",
    )
    sweep.add_argument(
        "--block-m-sweep", action="store_true",
        help="compare BLOCK_M=16 and 32 while keeping BLOCK_N=32, BLOCK_K=32",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")

    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float16)
    operations = {"pytorch": reference_matmul}
    if args.block_k_sweep:
        operations["triton_bk32"] = lambda x, y: triton_matmul(x, y, block_k=32)
        operations["triton_bk64"] = lambda x, y: triton_matmul(x, y, block_k=64)
        benchmark_name = "pytorch_vs_triton_block_k"
        measurement_order = "rotating_pytorch_triton_bk32_bk64"
        triton_tile = {"block_m": 16, "block_n": 32, "block_k": [32, 64]}
        sweep_suffix = "_bk32vs64"
    elif args.block_n_sweep:
        operations["triton_bn32"] = lambda x, y: triton_matmul(
            x, y, block_k=64, block_n=32
        )
        operations["triton_bn64"] = lambda x, y: triton_matmul(
            x, y, block_k=64, block_n=64
        )
        benchmark_name = "pytorch_vs_triton_block_n"
        measurement_order = "rotating_pytorch_triton_bn32_bn64"
        triton_tile = {"block_m": 16, "block_n": [32, 64], "block_k": 64}
        sweep_suffix = "_bn32vs64_bk64"
    elif args.block_m_sweep:
        operations["triton_bm16"] = lambda x, y: triton_matmul(
            x, y, block_m=16, block_n=32, block_k=32
        )
        operations["triton_bm32"] = lambda x, y: triton_matmul(
            x, y, block_m=32, block_n=32, block_k=32
        )
        benchmark_name = "pytorch_vs_triton_block_m"
        measurement_order = "rotating_pytorch_triton_bm16_bm32"
        triton_tile = {"block_m": [16, 32], "block_n": 32, "block_k": 32}
        sweep_suffix = "_bm16vs32_bn32_bk32"
    else:
        operations["triton"] = triton_matmul
        benchmark_name = "pytorch_vs_triton_matmul"
        measurement_order = "alternating_pytorch_triton_pairs"
        triton_tile = {"block_m": 16, "block_n": 32, "block_k": 32}
        sweep_suffix = ""

    with torch.inference_mode():
        # Compare every float16 output to the same higher-precision result.
        expected_fp32 = reference_matmul(a.float(), b.float())
        accuracy = {}
        for name, operation in operations.items():
            actual = operation(a, b)
            accuracy[name] = accuracy_against_fp32(actual, expected_fp32)
            if name != "pytorch":
                try:
                    torch.testing.assert_close(
                        actual.float(), expected_fp32, rtol=1e-2, atol=1e-2,
                    )
                except AssertionError as error:
                    raise AssertionError(f"{name} failed the FP32 check") from error
        del expected_fp32, actual
        torch.cuda.synchronize()

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(args.warmup):
                for operation in operations.values():
                    operation(a, b)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graphs = {}
        captured_outputs = {}
        for name, operation in operations.items():
            graphs[name], captured_outputs[name] = capture_graph(
                operation, a, b, args.graph_ops
            )

        for _ in range(args.warmup):
            for graph in graphs.values():
                graph.replay()
        torch.cuda.synchronize()

        samples = {name: [] for name in operations}
        for repetition in range(args.repetitions):
            # Rotate measurement order so no implementation always runs first.
            names = list(operations)
            shift = repetition % len(names)
            order = names[shift:] + names[:shift]
            for name in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000 / args.graph_ops)

        # Keep captured output storage alive through all graph replays.
        _ = captured_outputs

    environment = json.loads(
        subprocess.check_output(
            [sys.executable, str(REPO_ROOT / "scripts" / "environment_report.py")],
            text=True,
            cwd=REPO_ROOT,
        )
    )
    result = {
        "benchmark": benchmark_name,
        "environment": environment,
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "dtype": "float16",
        "correctness": {
            "reference": "pytorch_float32",
            "rtol": 1e-2,
            "atol": 1e-2,
            "triton_checked": True,
            "accuracy_by_implementation": accuracy,
        },
        "timing_mode": "cuda_graph_replay_amortized",
        "warmup": args.warmup,
        "graph_replay_warmup": args.warmup,
        "repetitions": args.repetitions,
        "operations_per_timed_sample": args.graph_ops,
        "measurement_order": measurement_order,
        "triton_tile": triton_tile,
        "implementations": {name: summarize(values) for name, values in samples.items()},
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        REPO_ROOT
        / "results"
        / "raw"
        / f"matmul_compare_{args.m}x{args.k}x{args.n}_float16_graph{args.graph_ops}{sweep_suffix}_{stamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    for name, values in result["implementations"].items():
        print(f"{name}: {values['latency_us']['median']:.2f} us median per matmul")
        print(
            f"  FP32 mismatches: {accuracy[name]['mismatched_elements']}"
            f"/{accuracy[name]['total_elements']}"
        )
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
