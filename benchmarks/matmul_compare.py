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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=positive_int, default=1)
    parser.add_argument("--k", type=positive_int, default=256)
    parser.add_argument("--n", type=positive_int, default=256)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--repetitions", type=positive_int, default=30)
    parser.add_argument("--graph-ops", type=positive_int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")

    torch.manual_seed(0)
    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float16)
    operations = {"pytorch": reference_matmul, "triton": triton_matmul}

    with torch.inference_mode():
        # Compiles the Triton specialization and checks this exact workload.
        torch.testing.assert_close(
            triton_matmul(a, b), reference_matmul(a, b), rtol=1e-2, atol=1e-2
        )
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
            # Reverse each pair so a slow clock drift does not always favor one implementation.
            order = ("pytorch", "triton") if repetition % 2 == 0 else ("triton", "pytorch")
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
        "benchmark": "pytorch_vs_triton_matmul",
        "environment": environment,
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "dtype": "float16",
        "correctness": {"reference": "pytorch", "rtol": 1e-2, "atol": 1e-2},
        "timing_mode": "cuda_graph_replay_amortized",
        "warmup": args.warmup,
        "graph_replay_warmup": args.warmup,
        "repetitions": args.repetitions,
        "operations_per_timed_sample": args.graph_ops,
        "measurement_order": "alternating_pytorch_triton_pairs",
        "implementations": {name: summarize(values) for name, values in samples.items()},
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        REPO_ROOT
        / "results"
        / "raw"
        / f"matmul_compare_{args.m}x{args.k}x{args.n}_float16_graph{args.graph_ops}_{stamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    for name, values in result["implementations"].items():
        print(f"{name}: {values['latency_us']['median']:.2f} us median per matmul")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
