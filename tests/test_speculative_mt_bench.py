"""Verify category coverage and exclusion of incorrect outputs from timing summaries."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location(
    "speculative_mt_bench", Path(__file__).resolve().parents[1] / "benchmarks/speculative_mt_bench.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class MtBenchTest(unittest.TestCase):
    def test_first_turn_and_balanced_pilot(self):
        rows = [{"question_id": index, "category": category,
                 "prompt": [f"first-{index}", "followup"]}
                for index, category in enumerate(("writing", "writing", "math", "math"))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            selected = benchmark.load_prompts(path, limit=2)
            self.assertEqual([row["category"] for row in selected], ["writing", "math"])
            self.assertEqual([row["prompt"] for row in selected], ["first-0", "first-2"])
            path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]))
            with self.assertRaises(ValueError):
                benchmark.load_prompts(path)

    def test_summary_during_partial_run_and_failed_case(self):
        good = {"id": 1, "correctness_passed": True, "samples": [
            {"baseline": {"total_s": 4.}, "speculative": {"total_s": 2.}}]}
        bad = {"id": 2, "correctness_passed": False, "samples": [
            {"baseline": {"total_s": 100.}, "speculative": {"total_s": .01}}]}
        summary = benchmark.summarize([good, bad])
        self.assertEqual(summary["aggregate_speedup"], 2.)
        self.assertEqual(summary["timed_cases"], 1)
        self.assertEqual(summary["failed_cases"], [2])


if __name__ == "__main__":
    unittest.main()
