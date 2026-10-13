"""Paired CUDA-graph comparison of fused RMSNorm and the Qwen-style reference."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from fastinfer.kernels.rmsnorm import reference_rmsnorm
from fastinfer.kernels.triton_rmsnorm import rmsnorm
from matmul_compare import capture_graph, summarize

ROOT = Path(__file__).resolve().parents[1]
SHAPES = ((1, 1536), (16, 1536), (128, 1536), (1, 65),
          (1, 1024), (1, 4096), (1, 8192))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(2)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"rmsnorm_first_comparison_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "qwen_style_pytorch_vs_fused_rmsnorm", "dtype": "float16",
        "seed": 2, "eps": 1e-6, "rtol": .002, "atol": .002,
        "kernel": {"programs": "one per token", "num_warps": 4, "enable_fp_fusion": False},
        "timing": "cuda_graph_replay_amortized", "operations_per_sample": 100,
        "repetitions": 30, "warmup": 10, "graph_warmup": 10,
        "order": "alternating_pytorch_triton", "cases": [],
        "scope": "Synthetic isolated RMSNorm; PyTorch reference matches Qwen arithmetic "
                 "order. Not native BF16, torch.compile or full model performance.",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("kernels/rmsnorm.py", "kernels/triton_rmsnorm.py",
                                       "benchmarks/rmsnorm_compare.py")},
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
    }
    operations = {"pytorch": reference_rmsnorm, "triton": rmsnorm}
    names = list(operations)
    with torch.inference_mode():
        for index, shape in enumerate(SHAPES):
            x = torch.randn(shape, device="cuda", dtype=torch.float16)
            weight = torch.randn(shape[-1], device="cuda", dtype=torch.float16)
            expected = reference_rmsnorm(x, weight)
            actual = rmsnorm(x, weight)
            oracle = reference_rmsnorm(x.float(), weight.float())
            assert torch.isfinite(oracle).all().item(), "nonfinite FP32 reference"
            accuracy = {}
            for name, value in (("pytorch", expected), ("triton", actual)):
                error = (value.float() - oracle).abs()
                accuracy[name] = {
                    "fp32_mismatches": int(((error > .002 + .002 * oracle.abs())
                                            | ~torch.isfinite(value)).sum().item()),
                    "max_abs_fp32_error": error.max().item(),
                }
                torch.testing.assert_close(value.float(), oracle, rtol=.002, atol=.002)
            torch.testing.assert_close(actual, expected, rtol=.002, atol=.002)
            case = {"shape": list(shape), "accuracy": accuracy,
                    "max_abs_pytorch_triton_difference": (actual.float()-expected.float()).abs().max().item()}
            report["cases"].append(case)
            del actual, expected, oracle
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(10):
                    for operation in operations.values():
                        operation(x, weight)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graphs, outputs = {}, {}
            for name in (names if index % 2 == 0 else names[::-1]):
                graphs[name], outputs[name] = capture_graph(operations[name], x, weight, 100)
            for _ in range(10):
                for graph in graphs.values():
                    graph.replay()
            torch.cuda.synchronize()
            samples = {name: [] for name in names}
            for repetition in range(30):
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
            print(f"Tokens={shape[0]}, hidden={shape[1]} | PyTorch {pt:.2f} us | "
                  f"Triton {tt:.2f} us | PT/Triton {pt/tt:.3f}x | "
                  "FP32 and PyTorch comparisons PASS", flush=True)
            destination.write_text(json.dumps(report, indent=2) + "\n")
            del graphs, outputs
    print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
