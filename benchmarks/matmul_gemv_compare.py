"""Compare specialized GEMV, tiled matmul and PyTorch on stored linear weights."""

from functools import partial
import hashlib
import json
import runpy
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from fastinfer.kernels.triton_matmul import matmul_kernel_resources, triton_matmul


from fastinfer.kernels.triton_gemv import GEMV_CONFIGS, gemv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(block_m=64, block_n=64, block_k=64, num_warps=4)


def run_round(seed, round_index, helpers):
    torch.manual_seed(seed)
    a = torch.randn((1, 1536), device="cuda", dtype=torch.float16)
    b = torch.randn((1536, 8960), device="cuda", dtype=torch.float16)
    # Same values, but storage corresponds to contiguous linear weights W[N,K].
    bt = b.t().contiguous().t()
    assert torch.equal(b, bt)
    expected = a.float() @ b.float()
    weight = bt.t()
    operations = {
        "pytorch": lambda x, w: x @ w.t(),
        "triton_tiled": lambda x, w: triton_matmul(x, w.t(), **CONFIG),
        **{name: partial(gemv, **config) for name, config in GEMV_CONFIGS.items()},
    }
    accuracy = {}
    resources = {"triton_tiled": matmul_kernel_resources(a, bt, **CONFIG)}
    inputs = {name: weight for name in operations}
    for name, op in operations.items():
        result = op(a, inputs[name])
        accuracy[name] = helpers["accuracy_against_fp32"](result, expected)
        if name != "pytorch":
            torch.testing.assert_close(result.float(), expected, rtol=.01, atol=.01)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            for name, op in operations.items():
                op(a, inputs[name])
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graphs, outputs = {}, {}
    names = list(operations)
    # Vary graph construction order across repeats too.
    shift = round_index % len(names)
    for name in names[shift:] + names[:shift]:
        graphs[name], outputs[name] = helpers["capture_graph"](
            operations[name], a, inputs[name], 100)
    for _ in range(10):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in names}
    for repetition in range(30):
        shift = (repetition + round_index) % len(names)
        for name in names[shift:] + names[:shift]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 100)
    result = {
        "seed": seed, "round": round_index + 1,
        "strides": {"a": list(a.stride()), "stored_weight": list(weight.stride())},
        "accuracy": accuracy, "kernel_resources": resources,
        "implementations": {name: helpers["summarize"](values)
                            for name, values in samples.items()},
    }
    pt = result["implementations"]["pytorch"]["latency_us"]["median"]
    for name, timing in result["implementations"].items():
        us = timing["latency_us"]["median"]
        errors = accuracy[name]["mismatched_elements"]
        print(f"Round {round_index + 1}, seed {seed}, {name}: {us:.2f} us | "
              f"PT/implementation {pt / us:.3f}x | mismatches {errors}", flush=True)
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("highest")
    previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    helpers = runpy.run_path(str(ROOT / "benchmarks/matmul_compare.py"))
    output = ROOT / "results/raw" / (
        "mlp_expand_gemv_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "single_row_gemv_candidates",
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")],
            cwd=ROOT, text=True)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shape": {"m": 1, "k": 1536, "n": 8960}, "dtype": "float16",
        "tile": CONFIG, "gemv_configs": GEMV_CONFIGS, "device": torch.cuda.current_device(),
        "allow_fp16_reduced_precision_reduction": True,
        "timing_mode": "cuda_graph_replay_amortized", "graph_ops": 100,
        "warmup": 10, "graph_warmup": 10, "repetitions": 30,
        "measurement_order": "rotating_five_implementations",
        "correctness": {"reference": "pytorch_float32", "rtol": .01, "atol": .01},
        "scope": "Synthetic FP16 matmul; transpose/copy preparation excluded from timing. "
                 "No bias, actual model weights, or end-to-end inference.",
        "kernel_sha256": hashlib.sha256((ROOT / "kernels/triton_gemv.py").read_bytes()).hexdigest(),
        "selection_note": "Three candidates on target shape; repeat the selected winner separately.",
        "rounds": [],
    }
    try:
        with torch.inference_mode():
            for index, seed in enumerate((0, 0, 1)):
                report["rounds"].append(run_round(seed, index, helpers))
                output.write_text(json.dumps(report, indent=2) + "\n")
                torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
