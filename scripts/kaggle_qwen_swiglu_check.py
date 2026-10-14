"""Prepare the pinned kernel environment and Qwen dependencies for capture."""

from pathlib import Path
import subprocess
import sys
import tempfile

from kaggle_swiglu_check import main as prepare_kernels, DEFAULT_PYTHON, ROOT


def main():
    # This also runs the kernel and capture-hook tests (no model download).
    prepare_kernels()
    with tempfile.TemporaryDirectory(prefix="fastinfer-qwen-setup-") as directory:
        constraints = Path(directory) / "constraints.txt"
        constraints.write_text("torch==2.5.1+cu121\ntriton==3.1.0\nnumpy==1.26.4\n")
        subprocess.run(
            [sys.executable, "-m", "uv", "pip", "install", "--python", DEFAULT_PYTHON,
             "--link-mode", "copy", "--constraint", str(constraints),
             "transformers==4.46.3", "accelerate==1.1.1", "safetensors==0.4.5"],
            cwd=ROOT, check=True,
        )
    subprocess.run(
        [DEFAULT_PYTHON, "-c", "\n".join([
            "import torch, triton, transformers, accelerate, safetensors",
            "assert torch.__version__ == '2.5.1+cu121', torch.__version__",
            "assert triton.__version__ == '3.1.0', triton.__version__",
            "assert transformers.__version__ == '4.46.3', transformers.__version__",
            "print('Pinned Qwen dependencies ready')",
        ])], cwd=ROOT, check=True,
    )


if __name__ == "__main__":
    main()
