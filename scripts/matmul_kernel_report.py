"""Report resources and PTX matrix instructions for measured Triton matmul tiles."""

import argparse
import json

import torch
import triton

from fastinfer.kernels.triton_matmul import matmul_kernel_resources


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=positive_int, default=128)
    parser.add_argument("--k", type=positive_int, default=1024)
    parser.add_argument("--n", type=positive_int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")

    a = torch.zeros((args.m, args.k), device="cuda", dtype=torch.float16)
    b = torch.zeros((args.k, args.n), device="cuda", dtype=torch.float16)
    variants = []

    for block_m, block_n, block_k in ((16, 32, 32), (32, 32, 32), (32, 64, 32)):
        resources = matmul_kernel_resources(
            a, b, block_m=block_m, block_n=block_n, block_k=block_k,
        )
        variants.append({
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            **resources,
        })

    torch.cuda.synchronize()
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "variants": variants,
    }, indent=2))


if __name__ == "__main__":
    main()
