import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from fastinfer.kernels.matmul import reference_matmul


REPO_ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def time_eager(a: torch.Tensor, b: torch.Tensor, warmup: int, repetitions: int) -> list[float]:
    """Time one Python-submitted matmul per CUDA-event interval."""
    with torch.inference_mode():
        for _ in range(warmup):
            reference_matmul(a, b)
        torch.cuda.synchronize()

        samples_us = []
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            reference_matmul(a, b)
            end.record()
            end.synchronize()
            samples_us.append(start.elapsed_time(end) * 1000)
    return samples_us


def time_graph(
    a: torch.Tensor, b: torch.Tensor, warmup: int, repetitions: int, graph_ops: int
) -> list[float]:
    """Time repeated matmuls per replay, reporting amortized time per operation."""
    with torch.inference_mode():
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup):
                reference_matmul(a, b)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(graph_ops):
                captured_output = reference_matmul(a, b)

        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()

        samples_us = []
        for _ in range(repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples_us.append(start.elapsed_time(end) * 1000 / graph_ops)

        # The captured output must stay alive while the graph is replayed.
        _ = captured_output
    return samples_us


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=positive_int, default=1)
    parser.add_argument("--k", type=positive_int, default=256)
    parser.add_argument("--n", type=positive_int, default=256)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--repetitions", type=positive_int, default=30)
    parser.add_argument("--timing-mode", choices=("eager", "graph"), default="eager")
    parser.add_argument("--graph-ops", type=positive_int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required; run this benchmark on HPC")

    torch.manual_seed(0)
    a = torch.randn((args.m, args.k), device="cuda", dtype=DTYPES[args.dtype])
    b = torch.randn((args.k, args.n), device="cuda", dtype=DTYPES[args.dtype])

    if args.timing_mode == "graph":
        samples_us = time_graph(a, b, args.warmup, args.repetitions, args.graph_ops)
    else:
        samples_us = time_eager(a, b, args.warmup, args.repetitions)

    environment = json.loads(
        subprocess.check_output(
            [sys.executable, str(REPO_ROOT / "scripts" / "environment_report.py")],
            text=True,
            cwd=REPO_ROOT,
        )
    )
    ordered = sorted(samples_us)
    result = {
        "benchmark": "pytorch_matmul_baseline",
        "environment": environment,
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "dtype": args.dtype,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "timing_mode": (
            "cuda_graph_replay_amortized" if args.timing_mode == "graph" else "eager_cuda_events"
        ),
        "operations_per_timed_sample": args.graph_ops if args.timing_mode == "graph" else 1,
        "graph_replay_warmup": args.warmup if args.timing_mode == "graph" else 0,
        "latency_us": {
            "median": statistics.median(ordered),
            "p95": ordered[(95 * len(ordered) - 1) // 100],
            "min": ordered[0],
            "max": ordered[-1],
        },
        "samples_us": samples_us,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    method_suffix = f"_graph{args.graph_ops}" if args.timing_mode == "graph" else ""
    output = args.output or (
        REPO_ROOT
        / "results"
        / "raw"
        / f"matmul_baseline_{args.m}x{args.k}x{args.n}_{args.dtype}{method_suffix}_{stamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Median: {result['latency_us']['median']:.2f} us")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
