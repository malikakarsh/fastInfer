"""Reference checks on CPU and strict explicit-cast regression checks on CUDA."""

import importlib.util
from pathlib import Path
import unittest

import torch


spec = importlib.util.spec_from_file_location(
    "swiglu_compiler_accuracy",
    Path(__file__).resolve().parents[1] / "scripts/swiglu_compiler_accuracy.py",
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


class RoundingReferencesTest(unittest.TestCase):
    def test_explicit_cast_witness_changes_values(self):
        for _, fn, x, up, explicit in diagnostic.make_cases("cpu"):
            if not explicit:
                break
            expected = x.half().float()
            self.assertTrue((expected != x).any().item())
            if fn is diagnostic.cast_then_multiply:
                expected = expected * up
            torch.testing.assert_close(fn(x, up), expected, rtol=0, atol=0)

    def test_staged_and_final_rounding_are_distinct(self):
        gate = torch.tensor([1., 2., -1., .5], dtype=torch.float16)
        up = torch.full_like(gate, 1.5)
        refs = diagnostic.fp64_references(gate, up)
        self.assertTrue((refs["fp64_staged_rounding"] != refs["fp64_final_rounding"]).any().item())
        activation = (gate.double() * torch.sigmoid(gate.double())).half()
        expected = (activation.double() * up.double()).half()
        torch.testing.assert_close(refs["fp64_staged_rounding"], expected, rtol=0, atol=0)

    def test_nonfinite_outputs_do_not_pass(self):
        value = torch.tensor([float("nan"), float("inf")])
        comparison = diagnostic.compare(value, value)
        self.assertEqual(comparison["nonfinite_pairs"], 2)
        self.assertFalse(diagnostic.exact_match(comparison))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA compiler regression requires a GPU")
class CudaCompilerRoundingTest(unittest.TestCase):
    def test_explicit_rounding_contract(self):
        with torch.inference_mode():
            for case in diagnostic.make_cases("cuda"):
                if not case[-1]:
                    break
                result = diagnostic.run_case(*case, ["inductor"], .002, .0002)
                with self.subTest(case=case[0]):
                    entry = result["backends"]["inductor"]
                    self.assertEqual(entry["status"], "explicit_rounding_preserved", entry)


if __name__ == "__main__":
    unittest.main()
