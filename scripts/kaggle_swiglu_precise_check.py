"""Run the precise SwiGLU candidate through accuracy before optional timing.

Run with notebook Python. Setup repairs the editable install in the isolated
GPU interpreter. Every subprocess must succeed before the next stage starts.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/kaggle/working/fastinfer-t4-py312/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true", help="time only after model accuracy passes")
    args = parser.parse_args()

    def run(python, script, *arguments):
        subprocess.run([python, script, *arguments], cwd=ROOT, check=True)

    # Runs all SwiGLU tests, including fresh seeds and every finite FP16 gate value.
    run(sys.executable, "scripts/kaggle_qwen_swiglu_check.py")
    run(PYTHON, "benchmarks/qwen_swiglu_check.py",
        "--kernel", "precise_candidate", "--accuracy-only")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    accuracy_report = ROOT / "results/raw" / f"qwen_swiglu_decode_precise_candidate_accuracy_{stamp}.json"
    run(PYTHON, "benchmarks/qwen_swiglu_decode_compare.py",
        "--kernel", "precise_candidate", "--diagnose", "--require-accuracy",
        "--steps", "16", "--report", str(accuracy_report))
    report = json.loads(accuracy_report.read_text())
    exact = sum(case["candidates"]["triton_swiglu"]["accuracy"]["exact_logits"]
                for case in report["cases"])
    print(f"Candidate passed strict model checks; exact logits in {exact}/{len(report['cases'])} prompts.",
          flush=True)
    if args.measure:
        # Includes compiled as an isolated performance comparator, with its local
        # tolerance and exact-difference counts reported. This does not declare
        # it eligible for model deployment or whole-model timing.
        run(PYTHON, "benchmarks/qwen_swiglu_check.py", "--kernel", "precise_candidate")
        run(PYTHON, "benchmarks/qwen_swiglu_decode_compare.py",
            "--kernel", "precise_candidate", "--require-accuracy",
            "--steps", "16", "--repetitions", "30")
    run(PYTHON, "scripts/archive_swiglu_evidence.py")


if __name__ == "__main__":
    main()
