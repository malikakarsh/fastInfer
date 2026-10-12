import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(*args: str) -> str | None:
    if not shutil.which(args[0]):
        return None
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> None:
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "git_commit": command_output("git", "rev-parse", "HEAD"),
        "torch_version": package_version("torch"),
        "triton_version": package_version("triton"),
        "transformers_version": package_version("transformers"),
        "cuda_version": None,
        "gpu_count": 0,
        "gpus": [],
        "nvidia_driver": command_output(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ),
        "scheduler": (
            "SLURM" if shutil.which("squeue") else "PBS" if shutil.which("qstat") else None
        ),
    }

    if report["torch_version"] is not None:
        import torch

        report["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            report["gpu_count"] = torch.cuda.device_count()
            report["gpus"] = [
                {
                    "name": torch.cuda.get_device_properties(i).name,
                    "compute_capability": list(torch.cuda.get_device_capability(i)),
                    "memory_gib": round(
                        torch.cuda.get_device_properties(i).total_memory / 1024**3, 2
                    ),
                }
                for i in range(report["gpu_count"])
            ]

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
