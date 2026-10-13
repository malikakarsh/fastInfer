"""Change only the manual RMSNorm warp count at the measured 128x1536 shape."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
WARPS = (4, 8, 16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    from fastinfer.kernels.rmsnorm import reference_rmsnorm
    from fastinfer.kernels.triton_rmsnorm import _rmsnorm
    from matmul_compare import capture_graph, summarize

    torch.manual_seed(2)
    for shape in ((1, 1536), (16, 1536), (128, 1536)):
        x = torch.randn(shape, device="cuda", dtype=torch.float16)
        weight = torch.randn(shape[-1], device="cuda", dtype=torch.float16)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"rmsnorm_warps_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "manual_rmsnorm_controlled_warp_check", "shape": [128, 1536],
        "dtype": "float16", "seed": 2, "eps": 1e-6, "block": 2048,
        "num_warps": list(WARPS), "enable_fp_fusion": False,
        "timing": "cuda_graph_replay_amortized", "operations_per_sample": 100,
        "warmup": 10, "graph_warmup": 10, "repetitions": 30,
        "order": "rotating_warp_variants", "rtol": .002, "atol": .002,
        "scope": "Only num_warps changes in the existing manual kernel. "
                 "This does not isolate all differences from compiled PyTorch, "
                 "measure occupancy or alter the production wrapper's four-warps default.",
        "kernel_sha256": hashlib.sha256((ROOT / "kernels/triton_rmsnorm.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "configurations": {},
    }
    operations = {}
    with torch.inference_mode():
        expected = reference_rmsnorm(x, weight)
        oracle = reference_rmsnorm(x.float(), weight.float())
        assert torch.isfinite(oracle).all().item(), "nonfinite FP32 reference"
        for warps in WARPS:
            def operation(a, w, warps=warps):
                output = torch.empty_like(a)
                _rmsnorm[(128,)](a, w, output, 1536, 1e-6, 2048,
                                num_warps=warps, enable_fp_fusion=False)
                return output
            name = f"warps{warps}"
            operations[name] = operation
            output = torch.empty_like(x)
            handle = _rmsnorm[(128,)](x, weight, output, 1536, 1e-6, 2048,
                                     num_warps=warps, enable_fp_fusion=False)
            torch.testing.assert_close(output, expected, rtol=.002, atol=.002)
            torch.testing.assert_close(output.float(), oracle, rtol=.002, atol=.002)
            error = (output.float() - oracle).abs()
            report["configurations"][name] = {
                "accuracy_passed": True,
                "fp32_mismatches": int((error > .002 + .002 * oracle.abs()).sum().item()),
                "max_abs_fp32_error": error.max().item(),
                "resources": {"programs": 128, "threads_per_program": 32 * warps,
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
        for name, operation in operations.items():
            graphs[name], outputs[name] = capture_graph(operation, x, weight, 100)
        for _ in range(10):
            for graph in graphs.values():
                graph.replay()
        torch.cuda.synchronize()
        samples = {name: [] for name in names}
        for repetition in range(30):
            offset = repetition % len(names)
            for name in names[offset:] + names[:offset]:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[name].replay()
                end.record()
                end.synchronize()
                samples[name].append(start.elapsed_time(end) * 1000 / 100)
        for name in names:
            row = report["configurations"][name]
            row["timing"] = summarize(samples[name])
            latency = row["timing"]["latency_us"]["median"]
            resources = row["resources"]
            print(f"{name}: {latency:.3f} us | registers/thread {resources['registers_per_thread']} | "
                  f"spills {resources['spills']} | shared bytes {resources['shared_bytes']} | accuracy PASS",
                  flush=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
