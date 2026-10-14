"""Compare one/two/four/eight softmax warps with accuracy and resource reports."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from softmax_compare import native_triton_comparison

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((1, 4), (1, 65), (128, 128), (16, 1024), (128, 1536), (1, 8192))
WARPS = (1, 2, 4, 8)
CONFIRM_CASES = (((1, 65), 1), ((128, 128), 1), ((128, 1536), 2))


def timing_orders(names):
    """Thirty rounds: alternating forward/reverse cycles balance all positions."""
    names = tuple(names)
    orders = []
    for index in range(30):
        base = names if (index // len(names)) % 2 == 0 else names[::-1]
        offset = index % len(names)
        orders.append(base[offset:] + base[:offset])
    return orders


def measure_case(x, warps_to_test=WARPS):
    import triton
    from fastinfer.kernels.softmax import reference_softmax
    from fastinfer.kernels.triton_softmax import softmax, _softmax
    from matmul_compare import capture_graph, summarize

    native = torch.softmax(x, dim=-1)
    reference = reference_softmax(x)
    oracle = torch.softmax(x.double(), dim=-1).float()
    operations = {"pytorch_native": lambda a, unused: torch.softmax(a, dim=-1)}
    case = {"shape": list(x.shape), "variants": {}}
    for warps in warps_to_test:
        name = f"warps{warps}"
        output = torch.empty_like(x)
        handle = _softmax[(x.shape[0],)](
            x, output, x.shape[1], triton.next_power_of_2(x.shape[1]),
            num_warps=warps, enable_fp_fusion=False,
        )
        for expected in (native, reference):
            torch.testing.assert_close(output, expected, rtol=.002, atol=2e-6)
        torch.testing.assert_close(output.float(), oracle, rtol=.002, atol=2e-6)
        if not torch.isfinite(output).all().item() or not (output >= 0).all().item():
            raise RuntimeError(f"Nonfinite or negative probabilities from {name}")
        torch.testing.assert_close(output.float().sum(dim=-1),
                                   torch.ones(x.shape[0], device=x.device), rtol=0, atol=5e-4)
        normal_output = softmax(x, num_warps=warps)
        torch.testing.assert_close(normal_output, output, rtol=0, atol=0)
        error = (output.float() - oracle).abs()
        case["variants"][name] = {
            "accuracy_passed": True,
            "oracle_mismatches": int((error > 2e-6 + .002 * oracle.abs()).sum().item()),
            "max_abs_oracle_error": error.max().item(),
            "resources": {"programs": x.shape[0], "block": triton.next_power_of_2(x.shape[1]),
                          "num_warps": warps, "threads_per_program": 32 * warps,
                          "registers_per_thread": handle.n_regs, "spills": handle.n_spills,
                          "shared_bytes": handle.metadata.shared},
        }
        operations[name] = lambda a, unused, warps=warps: softmax(a, num_warps=warps)

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
    samples = {name: [] for name in operations}
    for order in timing_orders(operations):
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 100)
    case["implementations"] = {name: summarize(values) for name, values in samples.items()}
    for name, row in case["variants"].items():
        row["native_vs_variant"] = native_triton_comparison(samples["pytorch_native"], samples[name])
        row["four_warps_vs_variant"] = native_triton_comparison(samples["warps4"], samples[name])
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true",
                        help="Compare frozen candidates against four warps at seeds 4, 5, 6")
    args = parser.parse_args()
    seeds = (4, 5, 6) if args.confirm else (3,)
    trials = []
    for index, seed in enumerate(seeds):
        cases = CONFIRM_CASES if args.confirm else tuple((shape, None) for shape in SHAPES)
        if index % 2:
            cases = cases[::-1]
        trials.extend((seed, shape, candidate) for shape, candidate in cases)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"softmax_warps_{'confirmation_' if args.confirm else ''}{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "controlled_softmax_warp_comparison", "dtype": "float16",
        "mode": "confirmation" if args.confirm else "selection",
        "seeds": list(seeds),
        "trials": [{"seed": seed, "shape": list(shape), "candidate_warps": candidate}
                   for seed, shape, candidate in trials],
        "warps": [1, 2, 4] if args.confirm else list(WARPS),
        "rtol": .002, "atol": 2e-6, "enable_fp_fusion": False,
        "timing_mode": "cuda_graph_replay_amortized", "operations_per_sample": 100,
        "warmup": 10, "graph_warmup": 10, "repetitions": 30,
        "order": "alternating_forward_reverse_rotations_of_three_providers" if args.confirm
                 else "alternating_forward_reverse_rotations_of_five_providers",
        "scope": "Only num_warps changes among manual variants on identical inputs per case. "
                 "Native baseline included; no compiler change or full-model claim. "
                 "Confirmation freezes shape-specific candidates at new seeds 4/5/6; "
                 "selection uses seed 3. Defaults remain four warps. "
                 "Spills is Triton3.1 local-byte metadata divided by four, not runtime traffic.",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("kernels/triton_softmax.py", "benchmarks/softmax_warp_check.py",
                                       "benchmarks/softmax_compare.py", "benchmarks/matmul_compare.py")},
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "cases": [],
    }
    try:
        with torch.inference_mode():
            torch.manual_seed(3)
            for seed, shape, candidate in trials:
                if args.confirm:
                    torch.manual_seed(seed)
                print(f"\nSeed={seed}, rows={shape[0]}, columns={shape[1]}", flush=True)
                x = torch.randn(shape, device="cuda", dtype=torch.float16)
                warps = (4, candidate) if candidate is not None else WARPS
                case = measure_case(x, warps)
                case.update(seed=seed, candidate_warps=candidate,
                            measurement_orders=timing_orders(
                                ("pytorch_native", *(f"warps{w}" for w in warps))))
                report["cases"].append(case)
                native_us = case["implementations"]["pytorch_native"]["latency_us"]["median"]
                print(f"  Native PyTorch: {native_us:.3f} us", flush=True)
                for name, row in case["variants"].items():
                    resources = row["resources"]
                    median = case["implementations"][name]["latency_us"]["median"]
                    paired = row["four_warps_vs_variant"]
                    print(f"  {name}: {median:.3f} us | registers/thread "
                          f"{resources['registers_per_thread']} | spills {resources['spills']} | "
                          f"shared {resources['shared_bytes']} bytes | "
                          f"paired saving versus four {paired['paired_median_saving_us']:+.3f} us | "
                          f"faster than four {paired['triton_faster_rounds']}/30 | accuracy PASS", flush=True)
                destination.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
