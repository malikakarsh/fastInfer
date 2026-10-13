"""Compare a reduction-order candidate on saved real inputs and new random inputs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch

from qwen_rmsnorm_stages import ROOT, difference_report, pytorch_stages


def paired_savings(baseline, candidate):
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("nonempty paired samples of equal length required")
    differences = [a - b for a, b in zip(baseline, candidate)]
    return {"samples_us": differences, "median_us": statistics.median(differences),
            "candidate_faster_rounds": sum(value > 0 for value in differences),
            "rounds": len(differences)}


def check_case(x, weight, eps):
    from matmul_compare import capture_graph, summarize
    from fastinfer.kernels.rmsnorm import reference_rmsnorm
    from fastinfer.kernels.triton_rmsnorm import rmsnorm
    from fastinfer.kernels.rmsnorm_diagnostic import triton_stages
    from fastinfer.kernels.rmsnorm_mean_candidate import rmsnorm_candidate

    expected = pytorch_stages(x, weight, eps=eps)
    oracle = reference_rmsnorm(x.float(), weight.float(), eps=eps)
    baseline = rmsnorm(x, weight, eps=eps)
    baseline_mean = triton_stages(x, weight, eps=eps)["mean_square"]
    candidate, candidate_mean = rmsnorm_candidate(x, weight, eps=eps, return_mean=True)
    uninstrumented = rmsnorm_candidate(x, weight, eps=eps)
    # Timing uses the uninstrumented candidate; enforce agreement with inspected output.
    torch.testing.assert_close(candidate, uninstrumented, rtol=0, atol=0)
    for output in (baseline, candidate):
        torch.testing.assert_close(output, expected["weighted_output_fp16"], rtol=.002, atol=.002)
        torch.testing.assert_close(output.float(), oracle, rtol=.002, atol=.002)
    row = {
        "eps": eps, "accuracy_tolerance_passed": True,
        "candidate_diagnostic_matches_uninstrumented": True,
        "baseline_vs_pytorch": difference_report(baseline, expected["weighted_output_fp16"]),
        "candidate_vs_pytorch": difference_report(candidate, expected["weighted_output_fp16"]),
        "baseline_mean_vs_pytorch": difference_report(baseline_mean, expected["mean_square"]),
        "candidate_mean_vs_pytorch": difference_report(candidate_mean, expected["mean_square"]),
    }
    operations = {
        "baseline": lambda a, w: rmsnorm(a, w, eps=eps),
        "candidate": lambda a, w: rmsnorm_candidate(a, w, eps=eps),
    }
    # Compile and warm up outside capture; retain graph output tensors for replay.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            for operation in operations.values():
                operation(x, weight)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graphs, outputs = {}, {}
    for name, operation in operations.items():
        graphs[name], outputs[name] = capture_graph(operation, x, weight, 100)
    for _ in range(10):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in operations}
    for round_index in range(30):
        order = ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / 100)
    row["timings"] = {name: summarize(values) for name, values in samples.items()}
    row["paired_baseline_minus_candidate"] = paired_savings(samples["baseline"], samples["candidate"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-report", type=Path,
                        help="stage diagnosis JSON; defaults to latest local full stage report")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    if torch.__version__.split("+")[0] != "2.5.1":
        parser.error("This candidate investigates the pinned PyTorch 2.5.1 mean implementation")
    if args.input_report is None:
        reports = sorted((ROOT / "results/raw").glob("qwen_rmsnorm_stages_*.json"))
        reports = [path for path in reports if not path.name.endswith("_console_summary.json")]
        if not reports:
            parser.error("Run benchmarks/qwen_rmsnorm_stages.py first to save real inputs")
        args.input_report = reports[-1]
    source = json.loads(args.input_report.read_text())
    if source.get("failure") or len(source.get("cases", [])) != 3:
        parser.error("Need a complete three-prompt stage report")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = ROOT / "results/raw" / f"rmsnorm_mean_candidate_{stamp}.json"
    report = {
        "benchmark": "experimental_rmsnorm_reduction_order", "input_report": str(args.input_report),
        "scope": "Three saved original-model inputs plus five new synthetic seeds; single token "
                 "width1536 only. Candidate models a PyTorch2.5.1 source reduction path, "
                 "not a guarantee of equality. No model replacement or generation claim.",
        "rtol": .002, "atol": .002, "timing_mode": "cuda_graph_replay_amortized",
        "operations_per_sample": 100, "warmup": 10, "graph_warmup": 10,
        "repetitions": 30, "order": "alternating_baseline_candidate",
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("kernels/triton_rmsnorm.py", "kernels/rmsnorm_mean_candidate.py",
                                       "benchmarks/rmsnorm_mean_candidate_check.py")},
        "environment": json.loads(subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/environment_report.py")], cwd=ROOT, text=True)),
        "cases": [],
    }
    try:
        with torch.inference_mode():
            inputs = []
            for case in source["cases"]:
                path = args.input_report.parent / Path(case["tensor_snapshot"]).name
                snapshot = torch.load(path, map_location="cpu", weights_only=True)
                inputs.append((case["selected_module"], snapshot["input"].cuda(),
                               snapshot["weight"].cuda(), snapshot["eps"], case["prompt"]))
            for seed in (6, 7, 8, 9, 10):
                torch.manual_seed(seed)
                inputs.append((f"synthetic_seed{seed}",
                               torch.randn(1, 1, 1536, device="cuda", dtype=torch.float16),
                               torch.randn(1536, device="cuda", dtype=torch.float16), 1e-6, None))
            for label, x, weight, eps, prompt in inputs:
                case = {"label": label, "prompt": prompt, **check_case(x, weight, eps)}
                report["cases"].append(case)
                baseline = case["timings"]["baseline"]["latency_us"]["median"]
                candidate = case["timings"]["candidate"]["latency_us"]["median"]
                print(f"{label}: output differences baseline/candidate "
                      f"{case['baseline_vs_pytorch']['different_elements']}/"
                      f"{case['candidate_vs_pytorch']['different_elements']} | "
                      f"mean matches PyTorch {case['candidate_mean_vs_pytorch']['different_elements'] == 0} | "
                      f"{baseline:.3f}/{candidate:.3f} us | paired saving "
                      f"{case['paired_baseline_minus_candidate']['median_us']:+.3f} us | accuracy PASS",
                      flush=True)
                destination.write_text(json.dumps(report, indent=2) + "\n")
    except Exception as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved: {destination}", flush=True)


if __name__ == "__main__":
    main()
