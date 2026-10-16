"""The notebook workflow must stop before timing after an accuracy failure."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


class PreciseWorkflowTest(unittest.TestCase):
    def run_workflow(self, failure=None):
        script = Path(__file__).resolve().parents[1] / "scripts/kaggle_swiglu_precise_check.py"
        spec = importlib.util.spec_from_file_location("precise_workflow", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calls = []

        def run(command, **kwargs):
            self.assertTrue(kwargs["check"])
            calls.append(command)
            if failure and command[1].endswith(failure) and "--diagnose" in command:
                raise subprocess.CalledProcessError(1, command)
            if "--report" in command:
                destination = Path(command[command.index("--report") + 1])
                destination.parent.mkdir(parents=True)
                destination.write_text(json.dumps({"cases": [
                    {"candidates": {"triton_swiglu": {"accuracy": {"exact_logits": True}}}}
                ]}))

        with tempfile.TemporaryDirectory() as directory:
            module.ROOT = Path(directory)
            with patch("sys.argv", [str(script), "--measure"]), patch.object(module.subprocess, "run", run):
                if failure:
                    with self.assertRaises(subprocess.CalledProcessError):
                        module.main()
                else:
                    module.main()
        return calls

    def test_failed_model_accuracy_stops_before_timing(self):
        calls = self.run_workflow("qwen_swiglu_decode_compare.py")
        self.assertEqual(len(calls), 3)
        self.assertIn("--accuracy-only", calls[1])
        self.assertIn("--require-accuracy", calls[-1])

    def test_performance_follows_model_accuracy(self):
        calls = self.run_workflow()
        self.assertIn("--diagnose", calls[2])
        self.assertNotIn("--accuracy-only", calls[3])
        self.assertIn("--require-accuracy", calls[4])
        self.assertNotIn("--diagnose", calls[4])
        self.assertTrue(calls[-1][1].endswith("archive_swiglu_evidence.py"))


if __name__ == "__main__":
    unittest.main()
