"""Report compiled GEMV resources; this script does not measure latency."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=positive_int, default=1536)
    parser.add_argument("--n", type=positive_int, default=8960)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.k > 16384:
        parser.error("K must be at most 16384, matching the GEMV wrapper")

    import torch
    import triton

    from fastinfer.kernels.triton_gemv import GEMV_CONFIGS, _gemv

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    device = torch.cuda.current_device()
    pad_k = triton.next_power_of_2(args.k)
    variants = []

    with torch.inference_mode():
        a = torch.randn((1, args.k), device="cuda", dtype=torch.float16)
        weight = torch.randn((args.n, args.k), device="cuda", dtype=torch.float16)
        expected = a.float() @ weight.float().t()

        for name, config in GEMV_CONFIGS.items():
            rows = config["block_rows"]
            grid = (triton.cdiv(args.n, rows),)
            output = torch.empty((1, args.n), device=a.device, dtype=a.dtype)

            # Same kernel and launch arguments as gemv(); retain its compiled handle.
            compiled = _gemv[grid](
                a,
                weight,
                output,
                args.n,
                args.k,
                rows,
                pad_k,
                num_warps=config["num_warps"],
            )
            torch.cuda.synchronize()
            torch.testing.assert_close(output.float(), expected, rtol=.01, atol=.01)
            difference = (output.float() - expected).abs()

            variants.append({
                "name": name,
                "block_rows": rows,
                "pad_k": pad_k,
                "programs": grid[0],
                "warps_per_program": compiled.metadata.num_warps,
                "threads_per_program": 32 * compiled.metadata.num_warps,
                "registers_per_thread": compiled.n_regs,
                "spills": compiled.n_spills,
                "shared_memory_bytes_per_program": compiled.metadata.shared,
                "correctness": {
                    "reference": "pytorch_float32",
                    "rtol": .01,
                    "atol": .01,
                    "passed": True,
                    "max_abs_error": difference.max().item(),
                },
            })

    environment = json.loads(subprocess.check_output(
        [sys.executable, str(ROOT / "scripts/environment_report.py")],
        cwd=ROOT,
        text=True,
    ))
    report = {
        "diagnostic": "gemv_compiled_resources",
        "environment": environment,
        "device_index": device,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "sm_count": torch.cuda.get_device_properties(device).multi_processor_count,
        "shape": {"m": 1, "k": args.k, "n": args.n},
        "dtype": "float16",
        "strides": {"a": list(a.stride()), "weight": list(weight.stride())},
        "kernel_sha256": hashlib.sha256(
            (ROOT / "kernels/triton_gemv.py").read_bytes()
        ).hexdigest(),
        "interpretation": (
            "Resource metadata from the compiled kernel, not measured occupancy or latency. "
            "spills is Triton's n_spills field, not a count of runtime memory transactions. "
            "Zero spills does not establish full occupancy or identify a bottleneck."
        ),
        "variants": variants,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = args.output or ROOT / "results/raw" / f"gemv_resources_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")

    print(f"GPU: {report['gpu']} | SMs: {report['sm_count']} | K={args.k} | PAD_K={pad_k}")
    print("Rows/program | Programs | Threads/program | Registers/thread | Spills | Shared bytes | FP32 check")
    for item in variants:
        print(
            f"{item['block_rows']:12} | {item['programs']:8} | "
            f"{item['threads_per_program']:15} | {item['registers_per_thread']:16} | "
            f"{item['spills']:6} | {item['shared_memory_bytes_per_program']:12} | PASS"
        )
    print(report["interpretation"])
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
