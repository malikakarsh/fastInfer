"""Prepare the existing pinned Kaggle interpreter, then check GPU SwiGLU.

Run this script with the notebook's Python. It installs only NumPy and the
editable project into the existing isolated interpreter, without resolving
or replacing its PyTorch/Triton dependencies.
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
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="existing pinned GPU interpreter")
    args = parser.parse_args()
    if not Path(args.python).is_file():
        parser.error(f"Pinned interpreter missing: {args.python}. Run the T4 environment setup first.")

    if importlib.util.find_spec("uv") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    subprocess.run(
        [sys.executable, "-m", "uv", "pip", "install", "--python", args.python,
         "--link-mode", "copy", "--no-deps", "numpy==1.26.4", "-e", str(ROOT)],
        cwd=ROOT, check=True,
    )
    subprocess.run(
        [args.python, "-c", "\n".join([
            "import numpy, torch, triton, fastinfer",
            "from fastinfer.kernels.triton_swiglu import swiglu",
            "assert torch.__version__.split('+')[0] == '2.5.1', torch.__version__",
            "assert triton.__version__ == '3.1.0', triton.__version__",
            "assert torch.cuda.is_available(), 'Select a GPU runtime before running the tests'",
            "print('Import OK:', fastinfer.__file__)",
            "print('GPU:', torch.cuda.get_device_name(0))",
        ])],
        cwd=ROOT, check=True,
    )
    subprocess.run(
        [args.python, "-m", "unittest", "discover", "-s", "tests",
         "-p", "*swiglu*.py", "-v"],
        cwd=ROOT, check=True,
    )


if __name__ == "__main__":
    main()
