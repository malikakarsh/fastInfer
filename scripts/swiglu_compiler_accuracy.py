"""Standalone PyTorch rounding reproducer; no fastinfer or custom Triton required.

Run on the original GPU/software stack first. A numerical difference is evidence,
not by itself proof of a compiler bug. FP64 references are numerical oracles,
not proofs of correctly rounded transcendental functions.
"""

import argparse
import json
from pathlib import Path
import platform

import torch
import torch.nn.functional as F


def cast_roundtrip(x, up):
    return x.to(torch.float16).to(torch.float32)


def cast_then_multiply(x, up):
    return x.to(torch.float16).to(torch.float32) * up


def swiglu(gate, up):
    return F.silu(gate) * up


def fp64_references(gate, up):
    # Work on CPU to provide a separate high-precision reference path.
    x, u = gate.detach().cpu().double(), up.detach().cpu().double()
    activation = x * torch.sigmoid(x)
    return {
        "fp64_final_rounding": (activation * u).to(gate.dtype),
        "fp64_staged_rounding": (
            activation.to(gate.dtype).double() * u
        ).to(gate.dtype),
    }


def compare(actual, expected, rtol=0.002, atol=0.0002):
    a, b = actual.detach().cpu(), expected.detach().cpu()
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"contract_passed": False, "shape": list(a.shape),
                "dtype": str(a.dtype), "expected_shape": list(b.shape),
                "expected_dtype": str(b.dtype)}
    a, b = a.double().reshape(-1), b.double().reshape(-1)
    finite = torch.isfinite(a) & torch.isfinite(b)
    errors = (a - b).abs()
    close = finite & (errors <= atol + rtol * b.abs())
    different = (a != b).nonzero().flatten()
    examples = [{"index": i, "actual": float(a[i]), "expected": float(b[i])}
                for i in different[:8].tolist()]
    return {
        "contract_passed": True,
        "elements": a.numel(),
        "exact_value_mismatches": different.numel(),
        "outside_tolerance": int((~close).sum()),
        "nonfinite_pairs": int((~finite).sum()),
        "max_abs_error": float(errors.max()) if finite.all() else None,
        "examples": examples,
    }


def exact_match(comparison):
    return (comparison["contract_passed"]
            and comparison["exact_value_mismatches"] == 0
            and comparison["nonfinite_pairs"] == 0)


def make_cases(device):
    # Exactly representable FP32 inputs straddle FP16 midpoints around +/-1.
    # Power-of-two multipliers make the following FP32 multiplication exact.
    offsets = torch.arange(-17, 18, dtype=torch.float32)
    positive = 1 + offsets / 4096
    x = torch.cat((positive, -positive)).to(device)
    up = torch.tensor([0.5, 2.0], device=device).repeat(x.numel() // 2)
    yield "explicit_cast_roundtrip", cast_roundtrip, x, up, True
    yield "explicit_cast_multiply", cast_then_multiply, x, up, True

    for seed in (21, 22, 23):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for shape in ((7,), (1, 8960), (16, 8960)):
            gate = (2 * torch.randn(shape, generator=generator)).half().to(device)
            up = torch.randn(shape, generator=generator).half().to(device)
            yield f"swiglu_seed{seed}_{'x'.join(map(str, shape))}", swiglu, gate, up, False
    # All finite half-precision gate values, with a bounded multiplier.
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    gate = bits.view(torch.float16)
    gate = gate[torch.isfinite(gate)].contiguous().to(device)
    yield "swiglu_finite_fp16_gates", swiglu, gate, torch.full_like(gate, 0.75), False


def run_case(name, fn, x, up, explicit, backends, rtol, atol):
    eager = fn(x, up)
    references = {"eager": eager}
    if explicit:
        # Eager must itself satisfy the intended cast contract before judging compile.
        expected = x.detach().cpu().half().float()
        if fn is cast_then_multiply:
            expected = expected * up.detach().cpu()
        references["explicit_rounding_reference"] = expected
    else:
        references.update(fp64_references(x, up))
    report = {"case": name, "shape": list(x.shape), "input_dtype": str(x.dtype),
              "explicit_rounding_case": explicit,
              "eager_references": {key: compare(eager, value, rtol, atol)
                                   for key, value in references.items() if key != "eager"},
              "backends": {}}
    for backend in backends:
        try:
            torch.compiler.reset()
            compiled = torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)
            actual = compiled(x, up)
            repeat = compiled(x, up)
            if x.is_cuda:
                torch.cuda.synchronize(x.device)
            comparisons = {key: compare(actual, value, rtol, atol)
                           for key, value in references.items()}
            repeat_check = compare(repeat, actual, rtol, atol)
            if explicit:
                reference_ok = exact_match(report["eager_references"]["explicit_rounding_reference"])
                if not reference_ok:
                    status = "reference_check_failed"
                elif not exact_match(comparisons["explicit_rounding_reference"]):
                    status = "potential_cast_semantics_bug"
                else:
                    status = "explicit_rounding_preserved"
            elif any(not c["contract_passed"] or c.get("nonfinite_pairs", 0)
                     for c in comparisons.values()):
                status = "investigate_contract_or_nonfinite_output"
            elif comparisons["eager"]["outside_tolerance"]:
                status = "investigate_tolerance_failure"
            elif not exact_match(comparisons["eager"]):
                status = "numerical_drift_within_tolerance"
            else:
                status = "matches_eager"
            if not exact_match(repeat_check):
                status = "investigate_unstable_repeat"
            report["backends"][backend] = {
                "status": status, "comparisons": comparisons, "repeat": repeat_check}
        except Exception as exc:
            # A backend unavailable on this installation is not a numerical bug.
            report["backends"][backend] = {
                "status": "execution_error", "error": f"{type(exc).__name__}: {exc}"}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--backends", nargs="+", default=["eager", "aot_eager", "inductor"])
    parser.add_argument("--output", type=Path, default=Path("results/raw/swiglu_compiler_accuracy.json"))
    parser.add_argument("--rtol", type=float, default=0.002)
    parser.add_argument("--atol", type=float, default=0.0002)
    parser.add_argument("--emulate-precision-casts", action="store_true",
                        help="enable Inductor's eager intermediate-rounding policy")
    parser.add_argument("--strict", action="store_true",
                        help="exit 1 for suspected cast bugs, tolerance/contract/repeat failures, or execution errors")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu only for a separate CPU investigation")
    if args.rtol < 0 or args.atol < 0:
        parser.error("tolerances must be nonnegative")
    from torch._inductor import config
    if args.emulate_precision_casts:
        config.emulate_precision_casts = True
    report = {"environment": {
        "torch": torch.__version__, "torch_git": torch.version.git_version,
        "cuda_build": torch.version.cuda, "python": platform.python_version(),
        "platform": platform.platform(), "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "compute_capability": torch.cuda.get_device_capability() if args.device == "cuda" else None,
    }, "compile_settings": {"fullgraph": True, "dynamic": False,
                            "emulate_precision_casts": config.emulate_precision_casts},
        "rtol": args.rtol, "atol": args.atol,
        "interpretation": "Cast failures are potential bugs requiring compiler-policy review. "
        "SwiGLU differences alone do not prove a bug. FP64 references are approximations. "
        "Exact comparisons here mean equal finite values, not signed-zero bit identity.",
        "cases": []}
    try:
        import triton
        report["environment"]["triton"] = triton.__version__
    except ImportError:
        report["environment"]["triton"] = None
    with torch.inference_mode():
        for case in make_cases(args.device):
            result = run_case(*case, args.backends, args.rtol, args.atol)
            report["cases"].append(result)
            print(result["case"] + ": " + ", ".join(
                f"{backend}={entry['status']}" for backend, entry in result["backends"].items()), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")
    acceptable = {"explicit_rounding_preserved", "matches_eager", "numerical_drift_within_tolerance"}
    if args.strict and any(entry["status"] not in acceptable
                           for case in report["cases"] for entry in case["backends"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
