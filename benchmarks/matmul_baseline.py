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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=positive_int, default=1)
    parser.add_argument("--k", type=positive_int, default=256)
    parser.add_argument("--n", type=positive_int, default=256)
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--repetitions", type=positive_int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required; run this benchmark on HPC")

    torch.manual_seed(0)
    a = torch.randn((args.m, args.k), device="cuda", dtype=DTYPES[args.dtype])
    b = torch.randn((args.k, args.n), device="cuda", dtype=DTYPES[args.dtype])

    with torch.inference_mode():
        for _ in range(args.warmup):
            reference_matmul(a, b)
        torch.cuda.synchronize()

        samples_us = []
        for _ in range(args.repetitions):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            reference_matmul(a, b)
            end.record()
            end.synchronize()
            samples_us.append(start.elapsed_time(end) * 1000)

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
        "latency_us": {
            "median": statistics.median(ordered),
            "p95": ordered[(95 * len(ordered) - 1) // 100],
            "min": ordered[0],
            "max": ordered[-1],
        },
        "samples_us": samples_us,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        REPO_ROOT
        / "results"
        / "raw"
        / f"matmul_baseline_{args.m}x{args.k}x{args.n}_{args.dtype}_{stamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Median: {result['latency_us']['median']:.2f} us")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
