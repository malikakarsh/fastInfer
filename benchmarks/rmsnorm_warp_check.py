"""Controlled RMSNorm warp comparison; optionally confirm four versus eight."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((1, 1536), (16, 1536), (128, 1536))
CONFIRM_SEEDS = (3, 4, 5)


def paired_four_eight(samples):
    """Positive savings mean eight warps was faster within the same round."""
    if not samples["warps4"] or len(samples["warps4"]) != len(samples["warps8"]):
        raise ValueError("nonempty paired sample lists of equal length required")
    differences = [four - eight for four, eight in zip(samples["warps4"], samples["warps8"])]
    return {
        "saving_us_samples": differences,
        "median_saving_us": statistics.median(differences),
        "eight_faster_rounds": sum(value > 0 for value in differences),
        "tied_rounds": sum(value == 0 for value in differences),
        "rounds": len(differences),
    }


def measure_case(x, weight, warps_values, offset):
    from fastinfer.kernels.rmsnorm import reference_rmsnorm
    from fastinfer.kernels.triton_rmsnorm import _rmsnorm
    from matmul_compare import capture_graph, summarize

    tokens, hidden = x.shape
    expected = reference_rmsnorm(x, weight)
    oracle = reference_rmsnorm(x.float(), weight.float())
    assert torch.isfinite(oracle).all().item(), "nonfinite FP32 reference"
    case = {"shape": list(x.shape), "configurations": {}}
    operations = {}
    for warps in warps_values:
        def operation(a, w, warps=warps):
            output = torch.empty_like(a)
            _rmsnorm[(a.shape[0],)](a, w, output, 1536, 1e-6, 2048,
                                   num_warps=warps, enable_fp_fusion=False)
            return output
        name = f"warps{warps}"
        operations[name] = operation
        output = torch.empty_like(x)
        handle = _rmsnorm[(tokens,)](x, weight, output, hidden, 1e-6, 2048,
                                    num_warps=warps, enable_fp_fusion=False)
        torch.testing.assert_close(output, expected, rtol=.002, atol=.002)
        torch.testing.assert_close(output.float(), oracle, rtol=.002, atol=.002)
        error = (output.float() - oracle).abs()
        case["configurations"][name] = {
            "accuracy_passed": True,
            "fp32_mismatches": int((error > .002 + .002 * oracle.abs()).sum().item()),
            "max_abs_fp32_error": error.max().item(),
            "resources": {"programs": tokens, "threads_per_program": 32 * warps,
                          "registers_per_thread": handle.n_regs, "spills": handle.n_spills,
                          "shared_bytes": handle.metadata.shared},
        }
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            for operation in operations.values():
                operation(x, weight)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graphs, outputs = {}, {}
    names = list(operations)
    start_offset = offset % len(names)
    for name in names[start_offset:] + names[:start_offset]:
        graphs[name], outputs[name] = capture_graph(operations[name], x, weight, 100)
    for _ in range(10):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in names}
    for repetition in range(30):
        start_offset = (offset + repetition) % len(names)
        for name in names[start_offset:] + names[:start_offset]:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 100)
    for name in names:
        case["configurations"][name]["timing"] = summarize(samples[name])
    case["four_vs_eight_paired"] = paired_four_eight(samples)
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-four-eight", action="store_true",
                        help="compare four/eight warps at 1,16,128 tokens across new seeds 3,4,5")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"rmsnorm_warps_{'confirmation_' if args.confirm_four_eight else ''}{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    warps = (4, 8) if args.confirm_four_eight else (4, 8, 16)
    report = {
        "benchmark": "manual_rmsnorm_controlled_warp_check",
        "confirmation_mode": args.confirm_four_eight,
        "dtype": "float16", "eps": 1e-6, "block": 2048,
        "num_warps": list(warps), "enable_fp_fusion": False,
        "timing": "cuda_graph_replay_amortized", "operations_per_sample": 100,
        "warmup": 10, "graph_warmup": 10, "repetitions": 30,
        "order": "rotating_warp_variants", "rtol": .002, "atol": .002,
        "scope": "Only num_warps changes in the existing manual kernel. "
                 "Confirmation mode uses three new seeds and reverses shape order in the middle pass. "
                 "Not occupancy, whole-model timing or all-shape tuning. Four-warps default unchanged.",
        "kernel_sha256": hashlib.sha256((ROOT / "kernels/triton_rmsnorm.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "cases": [],
    }
    try:
        with torch.inference_mode():
            if args.confirm_four_eight:
                for index, seed in enumerate(CONFIRM_SEEDS):
                    torch.manual_seed(seed)
                    shapes = SHAPES[::-1] if index % 2 else SHAPES
                    for shape_index, shape in enumerate(shapes):
                        x = torch.randn(shape, device="cuda", dtype=torch.float16)
                        weight = torch.randn(shape[-1], device="cuda", dtype=torch.float16)
                        case = measure_case(x, weight, warps, index + shape_index)
                        case["seed"] = seed
                        report["cases"].append(case)
                        four = case["configurations"]["warps4"]["timing"]["latency_us"]["median"]
                        eight = case["configurations"]["warps8"]["timing"]["latency_us"]["median"]
                        paired = case["four_vs_eight_paired"]
                        print(f"Seed {seed}, tokens={shape[0]}: 4 warps {four:.3f} us | "
                              f"8 warps {eight:.3f} us | paired saving {paired['median_saving_us']:+.3f} us | "
                              f"8 faster {paired['eight_faster_rounds']}/30 | accuracy PASS", flush=True)
                        destination.write_text(json.dumps(report, indent=2) + "\n")
            else:
                torch.manual_seed(2)
                for shape in SHAPES:
                    x = torch.randn(shape, device="cuda", dtype=torch.float16)
                    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float16)
                case = measure_case(x, weight, warps, 0)
                case["seed"] = 2
                report["cases"].append(case)
                report["shape"], report["seed"] = [128, 1536], 2
                report["configurations"] = case["configurations"]
                for name, row in case["configurations"].items():
                    resources = row["resources"]
                    print(f"{name}: {row['timing']['latency_us']['median']:.3f} us | "
                          f"registers/thread {resources['registers_per_thread']} | spills {resources['spills']} | "
                          f"shared bytes {resources['shared_bytes']} | accuracy PASS", flush=True)
                destination.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        destination.write_text(json.dumps(report, indent=2) + "\n")
        raise
    finally:
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
