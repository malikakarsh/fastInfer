"""Check environment creation without ensurepip or any package downloads."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location(
    "kaggle_cast_fix_runner",
    Path(__file__).resolve().parents[1] / "scripts/kaggle_pytorch_cast_fix_check.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class BootstrapTest(unittest.TestCase):
    def test_partial_environment_needs_no_pip_and_targets_uv_install(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = Path(directory) / "partial_env"
            environment.mkdir()  # Simulate the directory left by failed setup.
            with patch.object(runner.shutil, "which", return_value="/test/uv"), \
                    patch.object(runner, "run") as run:
                python = runner.prepare_environment(environment)
            command = run.call_args.args[0]
            self.assertEqual(command[:3], ["/test/uv", "pip", "install"])
            self.assertEqual(command[command.index("--python") + 1], python)
            self.assertIn(f"torch=={runner.TARGET}", command)
            subprocess.run([str(python), "-c", "import importlib.util, sys; "
                            "assert sys.prefix != sys.base_prefix; "
                            "assert importlib.util.find_spec('pip') is None"], check=True)
            self.assertIn("include-system-site-packages = false",
                          (environment / "pyvenv.cfg").read_text())


if __name__ == "__main__":
    unittest.main()
