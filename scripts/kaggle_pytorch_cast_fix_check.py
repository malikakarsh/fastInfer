"""Compare Kaggle's PyTorch with isolated PyTorch 2.13.0+cu126 on CUDA.

Installs only into a separate venv. Tests both default and precision-emulation
policies in fresh processes; retains failures and reports for all four runs.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
TARGET = "2.13.0+cu126"
INDEX = "https://download.pytorch.org/whl/cu126"


def run(command, **kwargs):
    print("Running: " + " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), cwd=ROOT, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=Path("/kaggle/working/pytorch_cast_fix_env"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/pytorch_cast_fix_results"))
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("run this CUDA experiment in Kaggle's Linux GPU environment")
    environment = args.venv.resolve()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for label in ("baseline", "fixed"):
        for policy in ("default", "emulated"):
            if (destination / f"{label}_{policy}.json").exists():
                parser.error(f"reports exist in {destination}; choose a new --output-dir")
    # No system-site-packages: Torch, Triton, and CUDA Python dependencies must
    # resolve from the isolated environment rather than Kaggle's base packages.
    venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
    python = environment / "bin/python"
    run([python, "-m", "pip", "install", "--no-cache-dir", "--index-url", INDEX,
         f"torch=={TARGET}"], check=True)
    run([python, "-c", "import torch; "
         f"assert torch.__version__ == '{TARGET}', torch.__version__; "
         "assert torch.cuda.is_available(), 'CUDA unavailable in isolated environment'; "
         "print('Isolated build:', torch.__version__, torch.cuda.get_device_name()); "
         "print('Compiled GPU architectures:', torch.cuda.get_arch_list()); "
         "print('CUDA smoke test:', (torch.ones(1, device='cuda') + 1).item())"], check=True)

    summary = {"target": TARGET, "index_url": INDEX,
               "note": "Version comparison changes Torch, Triton and CUDA dependencies; "
                       "it is not a single-commit causal experiment.", "runs": []}
    for label, interpreter in (("baseline", Path(sys.executable)), ("fixed", python)):
        audit = destination / f"{label}_source_audit.json"
        run([interpreter, "scripts/check_pytorch_cast_fix.py", "--offline", "--output", audit], check=True)
        for emulate in (False, True):
            name = f"{label}_{'emulated' if emulate else 'default'}"
            report_path = destination / f"{name}.json"
            if report_path.exists():
                parser.error(f"report exists: {report_path}; choose a new --output-dir for another run")
            command = [interpreter, "scripts/swiglu_compiler_accuracy.py", "--device", "cuda",
                       "--strict", "--output", report_path]
            if emulate:
                command.append("--emulate-precision-casts")
            process_env = os.environ.copy()
            process_env.pop("PYTHONPATH", None)
            process_env.pop("PYTHONHOME", None)
            # Separate graph/code caches for each build and numerical policy.
            process_env["TORCHINDUCTOR_CACHE_DIR"] = str(destination / f"cache_{name}")
            process_env["TRITON_CACHE_DIR"] = str(destination / f"triton_cache_{name}")
            process_env["PYTHONUNBUFFERED"] = "1"
            result = run(command, check=False, env=process_env)
            entry = {"name": name, "returncode": result.returncode, "report": str(report_path)}
            if report_path.exists():
                data = json.loads(report_path.read_text())
                entry["environment"] = data["environment"]
                entry["compile_settings"] = data["compile_settings"]
                casts = [case for case in data["cases"] if case["explicit_rounding_case"]]
                entry["inductor_cast_statuses"] = {
                    case["case"]: case["backends"]["inductor"]["status"] for case in casts}
                entry["explicit_casts_passed"] = len(casts) == 2 and all(
                    status == "explicit_rounding_preserved"
                    for status in entry["inductor_cast_statuses"].values())
            else:
                entry["error"] = "No JSON report; inspect subprocess output."
            summary["runs"].append(entry)
            (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Reports saved in {destination}")
    fixed_emulated = summary["runs"][-1]
    if not fixed_emulated.get("explicit_casts_passed"):
        raise SystemExit("Fixed build still failed explicit casts or could not complete; inspect saved reports.")
    print("Fixed build preserves both explicit cast cases with precision emulation enabled.")
    if fixed_emulated["returncode"]:
        raise SystemExit("Explicit casts passed, but other strict checks failed; inspect fixed_emulated.json.")


if __name__ == "__main__":
    main()
