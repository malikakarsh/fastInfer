"""Full-model timing must require both candidates' accuracy and actual dispatch."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
try:
    from qwen_swiglu_decode_compare import (
        timing_allowed, eligible_precise_providers, eligible_wrapper_providers,
    )
finally:
    sys.path.pop(0)


class SwigluTimingGateTest(unittest.TestCase):
    def test_wrapper_control_requires_both_exact_and_exercised_paths(self):
        row = {"accuracy": {"passed": True, "exact_logits": True},
               "local_checks": {"all_passed": True, "calls_checked": 6},
               "dispatch": {"stage_calls": 6, "fallback_calls": 0}}
        case = {"baseline_repeat_accuracy": {"passed": True},
                "candidates": {"wrapped_eager": deepcopy(row), "triton_swiglu": deepcopy(row)}}
        self.assertEqual(eligible_wrapper_providers(case, diagnose=False, expected_calls=6),
                         ["eager", "wrapped_eager", "triton_swiglu"])
        for name in case["candidates"]:
            for group, key, value in (("accuracy", "passed", False),
                                      ("accuracy", "exact_logits", False),
                                      ("local_checks", "all_passed", False),
                                      ("local_checks", "calls_checked", 5),
                                      ("dispatch", "stage_calls", 5),
                                      ("dispatch", "fallback_calls", 1)):
                with self.subTest(name=name, group=group, key=key):
                    broken = deepcopy(case)
                    broken["candidates"][name][group][key] = value
                    self.assertEqual(eligible_wrapper_providers(
                        broken, diagnose=False, expected_calls=6), [])
            broken = deepcopy(case)
            del broken["candidates"][name]
            self.assertEqual(eligible_wrapper_providers(broken, diagnose=False, expected_calls=6), [])
        self.assertEqual(eligible_wrapper_providers(case, diagnose=True, expected_calls=6), [])
        case["baseline_repeat_accuracy"]["passed"] = False
        self.assertEqual(eligible_wrapper_providers(case, diagnose=False, expected_calls=6), [])

    def test_precise_gate_checks_candidates_independently(self):
        row = {"accuracy": {"passed": True},
               "local_checks": {"all_passed": True, "calls_checked": 6},
               "dispatch": {"stage_calls": 6, "fallback_calls": 0}}
        case = {"baseline_repeat_accuracy": {"passed": True},
                "candidates": {"compiled_swiglu": deepcopy(row), "triton_swiglu": deepcopy(row)}}
        self.assertEqual(eligible_precise_providers(case, diagnose=False, expected_calls=6),
                         ["eager", "compiled_swiglu", "triton_swiglu"])
        case["candidates"]["compiled_swiglu"]["accuracy"]["passed"] = False
        self.assertEqual(eligible_precise_providers(case, diagnose=False, expected_calls=6),
                         ["eager", "triton_swiglu"])
        for group, key, value in (("accuracy", "passed", False),
                                  ("local_checks", "all_passed", False),
                                  ("local_checks", "calls_checked", 5),
                                  ("dispatch", "stage_calls", 5),
                                  ("dispatch", "fallback_calls", 1)):
            broken = deepcopy(case)
            broken["candidates"]["triton_swiglu"][group][key] = value
            self.assertEqual(eligible_precise_providers(broken, diagnose=False, expected_calls=6), [])
        self.assertEqual(eligible_precise_providers(case, diagnose=True, expected_calls=6), [])
        broken = deepcopy(case)
        broken["baseline_repeat_accuracy"]["passed"] = False
        self.assertEqual(eligible_precise_providers(broken, diagnose=False, expected_calls=6), [])
        broken = deepcopy(case)
        del broken["candidates"]["triton_swiglu"]
        self.assertEqual(eligible_precise_providers(broken, diagnose=False, expected_calls=6), [])

    def test_rejects_failed_missing_and_incomplete_validation(self):
        row = {"accuracy": {"passed": True},
               "local_checks": {"all_passed": True, "calls_checked": 6},
               "dispatch": {"stage_calls": 6, "fallback_calls": 0}}
        case = {"baseline_repeat_accuracy": {"passed": True},
                "candidates": {"compiled_swiglu": deepcopy(row), "triton_swiglu": deepcopy(row)}}
        self.assertTrue(timing_allowed(case, diagnose=False, expected_calls=6))
        self.assertFalse(timing_allowed(case, diagnose=True, expected_calls=6))
        broken = deepcopy(case);broken["baseline_repeat_accuracy"]["passed"] = False
        self.assertFalse(timing_allowed(broken, diagnose=False, expected_calls=6))
        for name in case["candidates"]:
            for group, key, value in (("accuracy", "passed", False),
                                      ("local_checks", "all_passed", False),
                                      ("local_checks", "calls_checked", 5),
                                      ("dispatch", "stage_calls", 5),
                                      ("dispatch", "fallback_calls", 1)):
                with self.subTest(name=name, group=group, key=key):
                    broken = deepcopy(case);broken["candidates"][name][group][key] = value
                    self.assertFalse(timing_allowed(broken, diagnose=False, expected_calls=6))
        missing = deepcopy(case);del missing["candidates"]["compiled_swiglu"]
        self.assertFalse(timing_allowed(missing, diagnose=False, expected_calls=6))


if __name__ == "__main__":
    unittest.main()
