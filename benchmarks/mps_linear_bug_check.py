"""Operator-only experiments for the captured layer-0 Qwen query projection.

No full model, generation, Transformers, draft, KV cache, or timing. Reconstructs
the known layer-0 inputs from three small checkpoint tensor reads plus embedding
rows, verifies the captured output fingerprint, then exports a standalone fixture.
Fresh workers test exact arithmetic and real-input accuracy under MPS settings.
Passing these tests does not prove that the backend has no bugs.
"""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gamma_n(count, unit_roundoff):
    product = count * unit_roundoff
    if product >= 1:
        raise ValueError("standard gamma_n bound is not informative when n*u >= 1")
    return product / (1 - product)


def error_metrics(actual, expected):
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    if a.shape != b.shape:
        raise ValueError("comparison shapes differ")
    finite = torch.isfinite(a) & torch.isfinite(b)
    if not bool(finite.all()):
        return {"finite": False, "nonfinite_pairs": int((~finite).sum())}
    error = (a - b).abs()
    return {"finite": True, "different_values": int((a != b).sum()),
            "max_abs": float(error.max()), "mean_abs": float(error.mean()),
            "rms_error": float(error.square().mean().sqrt())}


def exact_result(actual, expected):
    metrics = error_metrics(actual, expected)
    metrics["passed"] = metrics["finite"] and metrics["different_values"] == 0
    if not metrics["passed"]:
        a, b = actual.detach().cpu(), expected.detach().cpu()
        coordinates = (a != b).nonzero()[:8].tolist()
        metrics["examples"] = [{"coordinates": index, "actual": json_number(a[tuple(index)]),
                                "expected": float(b[tuple(index)])} for index in coordinates]
    return metrics


def json_number(value):
    value = float(value)
    return value if math.isfinite(value) else str(value)


def exact_dot(x, w, bias=0.):
    return sum((Fraction.from_float(float(a)) * Fraction.from_float(float(b))
                for a, b in zip(x, w)), Fraction.from_float(float(bias)))


def round_fraction(value, dtype=torch.float16):
    """Nearest-even rounding using exact rational distances, not FP64 distances."""
    initial = torch.tensor(float(value), dtype=dtype)
    if not torch.isfinite(initial):
        raise ValueError("exact reference outside the finite output range")
    candidates = [initial,
                  torch.nextafter(initial, torch.tensor(float("-inf"), dtype=dtype)),
                  torch.nextafter(initial, torch.tensor(float("inf"), dtype=dtype))]
    integer_dtype = torch.int16 if dtype == torch.float16 else torch.int32
    finite = [candidate for candidate in candidates if torch.isfinite(candidate)]
    chosen = min(finite, key=lambda candidate: (
        abs(Fraction.from_float(float(candidate)) - value),
        int(candidate.view(integer_dtype)) & 1))
    return float(chosen)


def rational_checks(fixture, full, single):
    source_changed = fixture["captured_full"] != fixture["captured_single"]
    changed = source_changed | (full != single)
    records = []
    for row, column in changed.nonzero()[:16].tolist():
        reference = exact_dot(fixture["inputs"][row].tolist(), fixture["weight"][column].tolist(),
                              float(fixture["bias"][column]))
        nearest = round_fraction(reference)
        a, b = float(full[row, column]), float(single[row, column])
        records.append({"row": row, "column": column,
                        "exact_reference_numerator": str(reference.numerator),
                        "exact_reference_denominator": str(reference.denominator),
                        "exact_reference_approx": float(reference), "nearest_even_fp16": nearest,
                        "full": a, "single": b, "full_matches_nearest": a == nearest,
                        "single_matches_nearest": b == nearest,
                        "note": "Exact dot plus bias, then one rounding. Linear need not promise this scheme."})
    return records


def validate_fixture(fixture):
    x, w, b = fixture["inputs"], fixture["weight"], fixture["bias"]
    if (x.ndim != 2 or w.ndim != 2 or b.ndim != 1 or x.shape[1] != w.shape[1]
            or b.shape[0] != w.shape[0] or x.dtype != torch.float16
            or w.dtype != x.dtype or b.dtype != x.dtype):
        raise ValueError("fixture must contain compatible FP16 inputs, weight and bias")
    if not all(bool(torch.isfinite(value).all()) for value in (x, w, b)):
        raise ValueError("fixture tensors must be finite")


def operation(name, x, weight, bias):
    # x is [rows, K]; return [rows, N] for every public operation spelling.
    if name == "linear_3d":
        return F.linear(x.unsqueeze(0), weight, bias).squeeze(0)
    if name == "linear_2d":
        return F.linear(x, weight, bias)
    if name == "addmm":
        additive = bias if bias is not None else x.new_zeros(weight.shape[0])
        return torch.addmm(additive, x, weight.t())
    if name == "mm_then_bias":
        result = torch.mm(x, weight.t())
        return result if bias is None else result + bias
    if name == "fp32_linear_then_cast":
        return F.linear(x.float(), weight.float(), None if bias is None else bias.float()).to(x.dtype)
    raise ValueError(f"unknown operation {name}")


OPERATIONS = ("linear_3d", "linear_2d", "addmm", "mm_then_bias", "fp32_linear_then_cast")


@torch.inference_mode()
def evaluate(name, x, weight, bias, serial=False):
    if serial:
        output = torch.cat([operation(name, row.unsqueeze(0), weight, bias) for row in x], dim=0)
    else:
        output = operation(name, x, weight, bias)
    return output.detach().cpu().clone()


def reconstruction_matches(full, single, expected):
    changed = int((full != single).sum())
    if changed != expected["full_vs_single"]["different_values"]:
        return False
    if float((full.double() - single.double()).abs().max()) != expected["full_vs_single"]["max_abs"]:
        return False
    details = expected.get("first_differing_values", [])
    if not details:
        return False
    return all(float(full[tuple(item["coordinates"])]) == item["block_value"]
               and float(single[tuple(item["coordinates"])]) == item["single_value"] for item in details)


def read_checkpoint_tensor(snapshot, name, row_ids=None):
    # Memory-map only the requested tensor/embedding rows, not the 7B model.
    from safetensors import safe_open
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        file_name = json.loads(index_path.read_text())["weight_map"][name]
    else:
        file_name = "model.safetensors"
    with safe_open(snapshot / file_name, framework="pt", device="cpu") as handle:
        if row_ids is None:
            return handle.get_tensor(name).clone()
        sliced = handle.get_slice(name)
        return torch.cat([sliced[index:index + 1] for index in row_ids], dim=0).clone()


@torch.inference_mode()
def build_fixture(layer_report, device, destination):
    saved = json.loads(layer_report.read_text())
    if saved.get("selected_probe_module") != "model.layers.0.self_attn.q_proj":
        raise ValueError("automatic reconstruction supports only the captured layer-0 q_proj")
    if not saved.get("controls_passed") or not saved.get("operator_probe", {}).get("shape_difference_reproduced"):
        raise ValueError("source layer report must have passed controls and reproduced the operator difference")
    snapshot = Path(saved["target_snapshot"])
    config = json.loads((snapshot / "config.json").read_text())
    if config["model_type"] != "qwen2":
        raise ValueError("fixture reconstruction requires the original Qwen2 checkpoint")
    dtype = getattr(torch, saved["environment"]["dtype"].split(".")[-1])
    if dtype != torch.float16:
        raise ValueError("this suite targets the captured FP16 case; use its case148 layer report")
    tokens = saved["verification_input_ids"]
    embeddings = read_checkpoint_tensor(snapshot, "model.embed_tokens.weight", tokens).to(dtype).unsqueeze(0).to(device)
    norm_weight = read_checkpoint_tensor(snapshot, "model.layers.0.input_layernorm.weight").to(dtype).to(device)
    # Same explicit arithmetic as installed Qwen2RMSNorm; no model construction.
    values = embeddings.float()
    variance = values.pow(2).mean(-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + config["rms_norm_eps"])
    inputs = norm_weight * normalized.to(dtype)
    weight_cpu = read_checkpoint_tensor(snapshot, "model.layers.0.self_attn.q_proj.weight").to(dtype)
    bias_cpu = read_checkpoint_tensor(snapshot, "model.layers.0.self_attn.q_proj.bias").to(dtype)
    weight, bias = weight_cpu.to(device), bias_cpu.to(device)
    full = F.linear(inputs, weight, bias).cpu()
    single = torch.cat([F.linear(inputs[:, i:i + 1].contiguous(), weight, bias).cpu()
                        for i in range(inputs.shape[1])], dim=1)
    if not reconstruction_matches(full, single, saved["operator_probe"]):
        raise RuntimeError("reconstructed operator does not match the captured output fingerprint; "
                           "refusing to present it as the original case")
    fixture = {"inputs": inputs[0].cpu().clone(), "weight": weight_cpu, "bias": bias_cpu,
               "captured_full": full[0], "captured_single": single[0],
               "provenance": {"layer_report": str(layer_report.resolve()), "layer_report_sha256": sha256(layer_report),
                              "target_snapshot": str(snapshot), "module": saved["selected_probe_module"],
                              "fingerprint_matches_captured_report": True,
                              "note": "Layer-0 projection input reconstructed from embedding rows and normalization. "
                                      "Captured output mismatch count, maximum, and listed coordinates match."}}
    torch.save(fixture, destination)
    return fixture


def fp32_model_bound(dot64, reference64, absolute_sum64, dtype, bias_present, reduction_length):
    """Conservative CONDITIONAL bound, not a PyTorch accuracy guarantee.

    Assumes conventional rounded FP32 products/reduction, no overflow, and at
    most one output-type rounding before bias plus a final output-type rounding.
    Includes a conservative FP64 reference error. Passing does not prove a bug
    absent; exceeding this requires investigating the actual backend policy.
    """
    count = 2 * reduction_length + 4
    g64 = gamma_n(count, 2.0 ** -53)
    sum_upper = absolute_sum64 / (1 - g64)
    reference_error = g64 * sum_upper
    accumulation_error = gamma_n(count, 2.0 ** -24) * sum_upper
    uout = torch.finfo(dtype).eps / 2
    half_smallest_subnormal = torch.finfo(dtype).tiny * torch.finfo(dtype).eps / 2
    error = accumulation_error
    if bias_present:
        error = error + uout * (dot64.abs() + reference_error + error) + half_smallest_subnormal
    error = error + uout * (reference64.abs() + reference_error + error) + half_smallest_subnormal
    return error + reference_error


def check_bound(actual, reference, bound):
    actual64 = actual.detach().cpu().double()
    finite = torch.isfinite(actual64)
    error = (actual64 - reference).abs()
    violations = (~finite) | (error > bound)
    return {"violating_values": int(violations.sum()), "finite": bool(finite.all()),
            "max_error_over_bound": float((error[finite] / bound[finite].clamp_min(1e-300)).max()) if finite.any() else None,
            "classification": "conditional FP32-accumulation model only; not a published PyTorch contract"}


def integer_case(rows, inputs, outputs):
    # At most 32 +/-1 products per row and |bias|<=2. Every product and every
    # partial sum is exactly representable in FP16 regardless of reduction order.
    x = torch.zeros(rows, inputs, dtype=torch.float16)
    columns = torch.arange(inputs)
    w = ((torch.arange(outputs)[:, None] + columns[None, :]) % 3 - 1).to(torch.float16)
    bias = (torch.arange(outputs) % 5 - 2).to(torch.float16)
    expected = []
    count = min(32, inputs)
    for row in range(rows):
        indices = (torch.arange(count) * max(1, inputs // count) + row) % inputs
        signs = (torch.arange(count) % 2 * 2 - 1).to(torch.float16)
        x[row, indices] = signs
        value = (w[:, indices].long() * signs.long()).sum(-1) + bias.long()
        expected.append(value)
    return x, w, bias, torch.stack(expected).to(torch.float16)


@torch.inference_mode()
def exact_tests(fixture, device, repeats):
    k, n = fixture["weight"].shape[1], fixture["weight"].shape[0]
    cases = []
    for rows in (1, 2, 4, 8):
        x, w, bias, expected = integer_case(rows, k, n)
        for name in OPERATIONS:
            xg, wg, bg = x.to(device), w.to(device), bias.to(device)
            for serial in (False, True):
                for repeat in range(repeats):
                    output = evaluate(name, xg, wg, bg, serial)
                    result = exact_result(output, expected)
                    cases.append({"case": "sparse_small_integer", "rows": rows, "operation": name,
                                  "serial": serial, "repeat": repeat, **result})
    # Normal real weights, with subnormals explicitly zeroed for these hard
    # identity checks. This avoids confusing a backend's denormal policy with
    # an exact-arithmetic violation. Actual real-input accuracy tests keep them.
    identity_weight = fixture["weight"].clone()
    identity_weight[identity_weight.abs() < torch.finfo(identity_weight.dtype).tiny] = 0
    identity_bias = fixture["bias"].clone()
    identity_bias[identity_bias.abs() < torch.finfo(identity_bias.dtype).tiny] = 0
    indices = [0, k // 3, k // 2, k - 1]
    x = torch.zeros(4, k, dtype=fixture["weight"].dtype)
    x[torch.arange(4), torch.tensor(indices)] = 1
    expected = identity_weight[:, indices].t().contiguous()
    zero = torch.zeros_like(x)
    expected_bias = identity_bias.expand(4, n).clone()
    for name in OPERATIONS:
        for label, values, bias, reference in (
            ("one_hot_normal_weights_no_bias", x, None, expected),
            ("zero_input_normal_bias", zero, identity_bias, expected_bias)):
            for serial in (False, True):
                for repeat in range(repeats):
                    output = evaluate(name, values.to(device), identity_weight.to(device),
                                      None if bias is None else bias.to(device), serial)
                    cases.append({"case": label, "rows": 4, "operation": name,
                                  "serial": serial, "repeat": repeat, **exact_result(output, reference)})
    return cases


@torch.inference_mode()
def real_tests(fixture, device, repeats):
    x64 = fixture["inputs"].double()
    w64 = fixture["weight"].double()
    bias64 = fixture["bias"].double()
    dot64 = F.linear(x64, w64)
    absolute_sum = F.linear(x64.abs(), w64.abs())
    records = []
    for dtype in (torch.float16, torch.float32):
        x, w = fixture["inputs"].to(device=device, dtype=dtype), fixture["weight"].to(device=device, dtype=dtype)
        for with_bias in (False, True):
            b = fixture["bias"].to(device=device, dtype=dtype) if with_bias else None
            reference = dot64 + bias64 if with_bias else dot64
            sum_abs = absolute_sum + bias64.abs() if with_bias else absolute_sum
            bound = fp32_model_bound(dot64, reference, sum_abs, dtype, with_bias, x.shape[1])
            for name in OPERATIONS:
                fulls = [evaluate(name, x, w, b) for _ in range(repeats)]
                singles = [evaluate(name, x, w, b, serial=True) for _ in range(repeats)]
                full, single = fulls[0], singles[0]
                record = {"dtype": str(dtype), "bias": with_bias, "operation": name,
                          "full_vs_single": error_metrics(full, single),
                          "full_vs_cpu_fp64": error_metrics(full, reference),
                          "single_vs_cpu_fp64": error_metrics(single, reference),
                          "full_vs_rounded_cpu_fp64": error_metrics(full, reference.to(dtype)),
                          "single_vs_rounded_cpu_fp64": error_metrics(single, reference.to(dtype)),
                          "full_conditional_bound": check_bound(full, reference, bound),
                          "single_conditional_bound": check_bound(single, reference, bound),
                          "full_repeat_stable": all(torch.equal(v, full) for v in fulls[1:]),
                          "single_repeat_stable": all(torch.equal(v, single) for v in singles[1:]),
                          "max_sum_absolute_terms": float(sum_abs.max())}
                # Compare the captured real case only for its matching FP16 biased linear form.
                if dtype == fixture["inputs"].dtype and with_bias and name == "linear_3d":
                    record["full_matches_fixture_capture"] = torch.equal(full, fixture["captured_full"])
                    record["single_matches_fixture_capture"] = torch.equal(single, fixture["captured_single"])
                    record["exact_rational_dot_checks"] = rational_checks(fixture, full, single)
                records.append(record)
    # Bias convention controls distinguish fused final rounding from an allowed
    # separately rounded product and bias addition; neither is automatically a bug.
    reference = dot64 + bias64
    separate = (dot64.half().double() + bias64).half()
    return records, {"fp64_final_rounding_vs_separate_half_bias": error_metrics(reference.half(), separate)}


def worker(args):
    fixture = torch.load(args.fixture, map_location="cpu", weights_only=True)
    validate_fixture(fixture)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this interpreter")
    report = {"status": "running", "environment": {
        "torch": torch.__version__, "torch_git": torch.version.git_version,
        "python": platform.python_version(), "platform": platform.platform(), "device": str(device),
        "mps_flags": {key: os.environ.get(key) for key in ("PYTORCH_MPS_FAST_MATH", "PYTORCH_MPS_PREFER_METAL")}},
        "fixture_sha256": sha256(args.fixture), "source_sha256": sha256(__file__),
        "fixture_subnormal_values": {name: int(((fixture[name] != 0) &
            (fixture[name].abs() < torch.finfo(fixture[name].dtype).tiny)).sum())
            for name in ("inputs", "weight", "bias")},
        "configuration": args.config_label,
        "interpretation": "Wrong exact-arithmetic outputs are strong correctness evidence. "
        "FP64-rounding disagreement alone is not a bug criterion. The FP32 accumulation "
        "bound is conditional on its stated arithmetic model, not a PyTorch guarantee. "
        "MPS flags request settings; they do not prove which internal kernel dispatched."}

    def save():
        args.worker_output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        report["exact_tests"] = exact_tests(fixture, device, args.repeats)
        failures = [case for case in report["exact_tests"] if not case["passed"]]
        print(f"{args.config_label}: exact arithmetic {len(report['exact_tests']) - len(failures)}/{len(report['exact_tests'])} passed", flush=True)
        save()
        report["real_tests"], report["bias_rounding_control"] = real_tests(fixture, device, args.repeats)
        for case in report["real_tests"]:
            if case["dtype"] == "torch.float16":
                print(f"  {args.config_label} {case['operation']} bias={case['bias']}: "
                      f"shape differences={case['full_vs_single'].get('different_values')}, "
                      f"rounded-reference differences full/single="
                      f"{case['full_vs_rounded_cpu_fp64'].get('different_values')}/"
                      f"{case['single_vs_rounded_cpu_fp64'].get('different_values')}", flush=True)
        bound_failures = [case for case in report["real_tests"] if
            case["full_conditional_bound"]["violating_values"] or case["single_conditional_bound"]["violating_values"]]
        unstable = [case for case in report["real_tests"] if not case["full_repeat_stable"] or not case["single_repeat_stable"]]
        report["summary"] = {"exact_arithmetic_failures": len(failures),
                             "conditional_bound_failures": len(bound_failures), "unstable_real_cases": len(unstable),
                             "finding": "exact_arithmetic_correctness_failure" if failures else
                                        "conditional_precision_model_exceeded" if bound_failures else
                                        "unstable_operation_results" if unstable else "no_correctness_violation_demonstrated"}
        capture = next(case for case in report["real_tests"] if case["dtype"] == "torch.float16"
                       and case["bias"] and case["operation"] == "linear_3d")
        report["summary"]["original_operator_outputs_reproduced"] = (
            capture["full_matches_fixture_capture"] and capture["single_matches_fixture_capture"])
        report["status"] = "complete"
        print(f"{args.config_label}: {json.dumps(report['summary'])}", flush=True)
    except BaseException as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--layer-report", type=Path)
    sources.add_argument("--fixture", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--config-label", default="inherited", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two repeats are required")
    if args.worker:
        worker(args)
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output_dir or ROOT / "results/raw" / f"mps_linear_bug_check_{stamp}").resolve()
    if output.exists():
        parser.error("output directory exists; choose a new path")
    output.mkdir(parents=True)
    summary = {"status": "running", "source_sha256": sha256(__file__), "workers": [],
               "scope": "Operator-only diagnostics. No full model or generation. "
                        "Exact arithmetic can expose a clear defect; passing cannot certify the whole backend."}
    try:
        if args.layer_report:
            fixture_path = output / "q_proj_fixture.pt"
            print("Reconstructing and fingerprint-checking the captured first-layer projection only", flush=True)
            build_fixture(args.layer_report, torch.device(args.device), fixture_path)
            summary["captured_output_fingerprint_matches"] = True
        else:
            fixture_path = args.fixture.resolve()
        summary["fixture"] = str(fixture_path)
        summary["fixture_sha256"] = sha256(fixture_path)
        configurations = [("inherited", {})]
        if args.device == "mps":
            configurations += [("fast_math_off", {"PYTORCH_MPS_FAST_MATH": "0"}),
                               ("prefer_metal_fast_math_off", {"PYTORCH_MPS_FAST_MATH": "0", "PYTORCH_MPS_PREFER_METAL": "1"})]
        for label, overrides in configurations:
            environment = os.environ.copy()
            environment.update(overrides)
            destination = output / f"{label}.json"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--fixture", str(fixture_path),
                       "--device", args.device, "--repeats", str(args.repeats), "--worker-output", str(destination), "--config-label", label]
            completed = subprocess.run(command, env=environment, check=False)
            entry = {"configuration": label, "returncode": completed.returncode, "report": str(destination)}
            if destination.exists():
                details = json.loads(destination.read_text())
                entry["summary"] = details.get("summary")
                entry["status"] = details["status"]
            summary["workers"].append(entry)
            (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summary["status"] = "complete" if all(entry["returncode"] == 0 for entry in summary["workers"]) else "worker_failed"
        if summary["status"] == "complete":
            findings = [entry["summary"]["finding"] for entry in summary["workers"]]
            summary["finding"] = next((name for name in (
                "exact_arithmetic_correctness_failure", "conditional_precision_model_exceeded",
                "unstable_operation_results") if name in findings), "no_correctness_violation_demonstrated")
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as exc:
        summary["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved operator reports and standalone fixture in {output}", flush=True)
    if summary["status"] != "complete":
        raise SystemExit("An operator worker failed; inspect summary.json and its report.")


if __name__ == "__main__":
    main()
