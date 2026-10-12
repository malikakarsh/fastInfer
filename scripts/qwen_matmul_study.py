"""Run a fixed-tile, synthetic FP16 study at Qwen2.5-1.5B dimensions."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECTIONS = {
    "query_or_output": (1536, 1536),
    "key_or_value": (1536, 256),
    "mlp_gate_or_up": (1536, 8960),
    "mlp_down": (8960, 1536),
}
CONFIG = {"block_m": 64, "block_n": 64, "block_k": 64, "num_warps": 4}
SOURCE = "https://huggingface.co/Qwen/Qwen2.5-1.5B/blob/main/config.json"

# Reuse the existing checked, rotating graph benchmark, including its raw samples.
# Each child has fresh CUDA state. No edits to the kernel or benchmark defaults.
CHILD = """
import json, runpy, sys, torch
import fastinfer.kernels.triton_matmul as kernel
settings = json.loads(sys.argv[1])
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = settings['allow']
kernel.MATMUL_SWEEP_CONFIGS.clear()
kernel.MATMUL_SWEEP_CONFIGS['triton_m64_n64_k64_w4'] = settings['config']
sys.argv = settings['argv']
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    repo = args.repo.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = repo / "results" / "raw" / f"qwen_fixed_{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "study.json").write_text(json.dumps({
        "model": "Qwen/Qwen2.5-1.5B", "config_source": SOURCE,
        "projections_kn": PROJECTIONS, "rows": [1, 16, 128], "tile": CONFIG,
        "scope": "Synthetic standard-normal FP16 inputs, contiguous A and B; "
        "dimensions only, not model weights, BF16, transposed linear weights, "
        "bias, fused operations, or end-to-end inference.",
        "method": "Separate paired PyTorch/Triton runs for each precision setting; "
        "compare within each file. Fixed tile, no candidate selection.",
    }, indent=2) + "\n")
    summaries = []
    for projection, (k, n) in PROJECTIONS.items():
        for m in (1, 16, 128):
            # Alternate which precision setting is tested first across shapes.
            for allow in ((True, False) if m != 16 else (False, True)):
                label = "reduced_allowed" if allow else "reduced_disabled"
                output = destination / f"{projection}_m{m}_{label}.json"
                print(f"\n{projection}: ({m}, {k}) @ ({k}, {n}), {label}", flush=True)
                settings = {"allow": allow, "config": CONFIG, "argv": [
                    "benchmarks/matmul_compare.py", "--m", str(m),
                    "--k", str(k), "--n", str(n), "--config-sweep",
                    "--output", str(output),
                ]}
                subprocess.run([args.python, "-c", CHILD, json.dumps(settings)],
                               cwd=repo, check=True)
                result = json.loads(output.read_text())
                timings = result["implementations"]
                pt = timings["pytorch"]["latency_us"]["median"]
                tt = timings["triton_m64_n64_k64_w4"]["latency_us"]["median"]
                summaries.append({"projection": projection, "m": m, "k": k, "n": n,
                    "allow_reduced_precision": allow, "pytorch_us": pt, "triton_us": tt,
                    "pytorch_over_triton": pt / tt,
                    "accuracy": result["correctness"]["accuracy_by_implementation"],
                    "raw_file": output.name})
                (destination / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print("\nProjection M reduced_allowed PyTorch_us Triton_us PT/Triton")
    for row in summaries:
        print(f"{row['projection']} {row['m']} {row['allow_reduced_precision']} "
              f"{row['pytorch_us']:.2f} {row['triton_us']:.2f} "
              f"{row['pytorch_over_triton']:.3f}")
    print(f"Results: {destination}")


if __name__ == "__main__":
    main()
