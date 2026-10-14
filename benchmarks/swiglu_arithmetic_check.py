"""Replay a saved Qwen pair to separate exponential and division hypotheses."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import torch
import torch.nn.functional as F

from qwen_decode_compare import ROOT
from qwen_linear_check import require_finite
from qwen_rmsnorm_stages import difference_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    if not args.snapshot.is_file():
        parser.error(f"Snapshot missing: {args.snapshot}. Re-run qwen_swiglu_stages.py if the session lost it.")
    from fastinfer.kernels.swiglu import reference_swiglu
    from fastinfer.kernels.triton_swiglu import swiglu
    from fastinfer.kernels.swiglu_arithmetic_candidate import VARIANTS, arithmetic_stages

    # This snapshot contains tensors and basic metadata; no model download/execution.
    data = torch.load(args.snapshot, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or not all(isinstance(data.get(k), torch.Tensor)
                                           for k in ("gate", "up", "captured_product")):
        raise ValueError("Expected the gate/up/captured_product diagnosis snapshot")
    if any(tuple(data[k].shape) != (1, 8960) or data[k].dtype != torch.float16
           for k in ("gate", "up", "captured_product")):
        raise ValueError("Expected saved FP16 single-token branches of shape1x8960")
    gate, up, captured = (data[k].to("cuda").contiguous()
                          for k in ("gate", "up", "captured_product"))
    for key, value in (("gate", gate), ("up", up), ("captured", captured)):
        require_finite(value, key)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"swiglu_arithmetic_{stamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "saved_qwen_swiglu_arithmetic_accuracy", "snapshot": str(args.snapshot),
        "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "module": data.get("module"), "prompt": data.get("prompt"),
        "scope": "Replay a fixed saved original-model pair; no model loading or timing. "
                 "Exponent implementation and source arithmetic varied independently. "
                 "FP16 intermediate rounding, geometry and fusion=False retained. "
                 "Plain direct division can be rewritten by compiler; inspect saved PTX. "
                 "div_rn is an explicit round-to-nearest division control. "
                 "Baseline instrumented product must exactly match production. "
                 "Exact matches here do not establish broader/full-model agreement or speed.",
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("kernels/swiglu_arithmetic_candidate.py", "kernels/triton_swiglu.py",
                                       "benchmarks/swiglu_arithmetic_check.py", "benchmarks/qwen_rmsnorm_stages.py")},
        "variants": {},
    }
    try:
        with torch.inference_mode():
            native = reference_swiglu(gate, up)
            if not torch.equal(native, captured):
                raise RuntimeError("Eager replay differs from saved original product; do not attribute candidate changes")
            reference = {"silu_fp32": F.silu(gate.float()), "silu_fp16": F.silu(gate),
                         "product_fp16": native}
            production = swiglu(gate, up)
            report["production_vs_eager"] = difference_report(production, native)
            if "production_triton" in data:
                report["production_matches_saved_triton"] = torch.equal(
                    production, data["production_triton"].to("cuda"))
                if not report["production_matches_saved_triton"]:
                    raise RuntimeError("Production replay changed since snapshot; investigate before attribution")
            print(f"Replaying {data.get('module')} | shape {list(gate.shape)}", flush=True)
            for variant in VARIANTS:
                stages, handle = arithmetic_stages(gate, up, variant=variant)
                if variant == "fast_exp_reciprocal" and not torch.equal(stages["product_fp16"], production):
                    raise RuntimeError("Instrumented baseline does not exactly match production")
                comparisons = {name: difference_report(value, reference[name]) for name, value in stages.items()}
                torch.testing.assert_close(stages["product_fp16"], native, rtol=.002, atol=2e-4)
                ptx = handle.asm.get("ptx")
                if not ptx:
                    raise RuntimeError("Compiled PTX unavailable")
                ptx_path = destination.with_name(destination.stem + f"_{variant}.ptx")
                ptx_path.write_text(ptx)
                code = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.DOTALL)
                operations = [line.strip() for line in code.splitlines()
                              if re.search(r"\b(?:div|rcp|ex2)\.", line)]
                report["variants"][variant] = {
                    "stages_vs_eager": comparisons, "within_original_tolerance": True,
                    "product_exactly_matches_eager": torch.equal(stages["product_fp16"], native),
                    "resources": {"registers_per_thread": handle.n_regs, "spills": handle.n_spills,
                                  "shared_bytes": handle.metadata.shared},
                    "ptx": str(ptx_path), "division_reciprocal_exp_instructions": operations,
                }
                print(f"\n{variant}:", flush=True)
                for name, diff in comparisons.items():
                    print(f"  {name}: {diff['different_elements']}/{diff['elements']} different | "
                          f"max difference {diff['max_abs_difference']:.9g}", flush=True)
            report["baseline_cross_check_passed"] = True
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
