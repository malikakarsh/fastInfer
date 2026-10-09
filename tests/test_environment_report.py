"""Smoke test for the benchmark environment inventory."""

import json
import subprocess
import sys
import unittest
from pathlib import Path


class EnvironmentReportTest(unittest.TestCase):
    def test_report_runs_and_identifies_python(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "environment_report.py"
        output = subprocess.check_output([sys.executable, str(script)], text=True)
        report = json.loads(output)

        self.assertEqual(report["python"], sys.version.split()[0])
        self.assertIn("timestamp_utc", report)
        self.assertIn("gpu_count", report)
        self.assertIn("gpus", report)


if __name__ == "__main__":
    unittest.main()
