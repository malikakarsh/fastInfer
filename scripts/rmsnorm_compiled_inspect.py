"""Save the PyTorch 2.5.1-generated RMSNorm code for one measured shape."""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def extract_kernels(source):
    """Read generated definitions and wrapper launches without executing source."""
    tree = ast.parse(source)
    definitions, launches = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if (node.func.attr == "triton" and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[1], ast.Constant)
                and isinstance(node.args[1].value, str)):
            definitions.append({"name": node.args[0].value, "source": node.args[1].value})
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "call":
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "run"):
                    launches.append(ast.get_source_segment(source, call))
    return {"kernel_definitions": definitions, "wrapper_launches": launches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=1536)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    if torch.__version__.split("+")[0] != "2.5.1":
        parser.error("this inspection uses private PyTorch 2.5.1 helpers; use the pinned environment")
    from torch._inductor import config
    from torch._inductor.utils import run_and_get_code
    from fastinfer.kernels.rmsnorm import reference_rmsnorm
    from fastinfer.kernels.triton_rmsnorm import rmsnorm, _rmsnorm
    import triton

    shape = (args.tokens, args.hidden)
    shapes = ((1, 1536), (16, 1536), (128, 1536), (1, 65),
              (1, 1024), (1, 4096), (1, 8192))
    if shape not in shapes:
        parser.error("select one of the seven benchmark shapes")
    # Recreate seed-2 inputs in benchmark order.
    torch.manual_seed(2)
    for current in shapes:
        x = torch.randn(current, device="cuda", dtype=torch.float16)
        weight = torch.randn(current[-1], device="cuda", dtype=torch.float16)
        if current == shape:
            break
    compiled = torch.compile(reference_rmsnorm, backend="inductor",
                             fullgraph=True, dynamic=False,
                             options={"triton.cudagraphs": False})
    print(f"Inspecting tokens={args.tokens}, hidden={args.hidden}...", flush=True)
    with torch.inference_mode(), config.patch({"force_disable_caches": True}):
        actual, sources = run_and_get_code(compiled, x, weight)
        expected = reference_rmsnorm(x, weight)
        manual = rmsnorm(x, weight)
        torch.testing.assert_close(actual, expected, rtol=.002, atol=.002)
        torch.testing.assert_close(manual, expected, rtol=.002, atol=.002)
        oracle = reference_rmsnorm(x.float(), weight.float())
        torch.testing.assert_close(actual.float(), oracle, rtol=.002, atol=.002)
        torch.testing.assert_close(manual.float(), oracle, rtol=.002, atol=.002)
        output = torch.empty_like(x)
        handle = _rmsnorm[(args.tokens,)](
            x, weight, output, args.hidden, 1e-6, triton.next_power_of_2(args.hidden),
            num_warps=4, enable_fp_fusion=False,
        )
        torch.cuda.synchronize()
    if not sources:
        raise RuntimeError("generated source unavailable; no code-layout conclusion can be drawn")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"rmsnorm_generated_{stamp}"
    destination.mkdir(parents=True)
    summary = {
        "shape": list(shape), "torch_version": torch.__version__, "seed": 2,
        "accuracy_passed": True,
        "compiled_settings": {"backend": "inductor", "fullgraph": True,
                              "dynamic": False, "internal_cuda_graphs": False,
                              "force_disable_caches_during_inspection": True},
        "manual_kernel_resources": {"programs": args.tokens, "num_warps": 4,
                                    "block": triton.next_power_of_2(args.hidden),
                                    "registers_per_thread": handle.n_regs,
                                    "spills": handle.n_spills,
                                    "shared_bytes": handle.metadata.shared},
        "scope": "Code inspection, not timing. Generated decorator configurations may "
                 "be candidate settings rather than the selected runtime setting. "
                 "Kernel layout differences do not establish a causal performance explanation.",
        "sources": [],
    }
    for index, source in enumerate(sources):
        path = destination / f"compiled_{index}.py"
        path.write_text(source)
        extracted = extract_kernels(source)
        summary["sources"].append({"path": str(path),
                                   "sha256": hashlib.sha256(source.encode()).hexdigest(),
                                   **extracted})
        print(f"\nGenerated Triton definitions: {len(extracted['kernel_definitions'])}", flush=True)
        for kernel in extracted["kernel_definitions"]:
            print(f"\nKernel: {kernel['name']}\n{kernel['source']}", flush=True)
        print("Wrapper launches:", flush=True)
        for launch in extracted["wrapper_launches"]:
            print(launch, flush=True)
    print(f"\nManual kernel resources: {summary['manual_kernel_resources']}", flush=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
