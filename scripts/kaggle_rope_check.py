"""Repair the existing isolated environment and run only RoPE reference tests.

Run with notebook Python. No model weights, GPU kernel or timing is involved.
"""

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = "/kaggle/working/fastinfer-t4-py312/bin/python"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    args = parser.parse_args()
    if not Path(args.python).is_file():
        parser.error(f"Existing isolated interpreter missing: {args.python}")
    if importlib.util.find_spec("uv") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    subprocess.run(
        [sys.executable, "-m", "uv", "pip", "install", "--python", args.python,
         "--link-mode", "copy", "--no-deps", "numpy==1.26.4", "-e", str(ROOT)],
        cwd=ROOT, check=True,
    )
    subprocess.run(
        [args.python, "-m", "unittest", "discover", "-s", "tests",
         "-p", "test_rope_reference.py", "-v"],
        cwd=ROOT, check=True,
    )


if __name__ == "__main__":
    main()
