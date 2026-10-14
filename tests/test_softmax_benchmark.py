"""Paired timing interpretation must not be inferred from separate medians."""

from pathlib import Path
import sys
import unittest
from collections import Counter
from itertools import permutations
from unittest.mock import patch

import torch

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
try:
    import softmax_compare as benchmark
    from softmax_compare import native_triton_comparison
finally:
    sys.path.pop(0)


class SoftmaxBenchmarkTest(unittest.TestCase):
    def test_provider_order_balance_and_original_order_preserved(self):
        three = ("reference", "native", "triton")
        original = list(permutations(three))
        self.assertEqual(benchmark.measurement_orders(three), original * 5)
        four = ("reference", "native", "compiled", "triton")
        orders = benchmark.measurement_orders(four)
        self.assertEqual(len(orders), 30)
        for order in orders:
            self.assertEqual(set(order), set(four))
        for position in range(4):
            counts = Counter(order[position] for order in orders)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_compiled_reference_captures_the_calculation_and_matches_examples(self):
        from fastinfer.kernels.softmax import reference_softmax

        real_compile = torch.compile
        def cpu_compile(function, **settings):
            settings["backend"] = "eager"  # Exercise Dynamo fullgraph capture without a GPU.
            settings.pop("options")
            return real_compile(function, **settings)
        with patch.object(torch, "compile", side_effect=cpu_compile) as compile_call:
            compiled = benchmark.compile_reference()
            for dtype in (torch.float16, torch.float32):
                x = torch.tensor([[-1002., -1001., -1000.], [60000., 60000., 60000.]], dtype=dtype)
                torch.testing.assert_close(compiled(x), reference_softmax(x), rtol=0, atol=0)
        self.assertIs(compile_call.call_args.args[0], reference_softmax)
        self.assertEqual(compile_call.call_args.kwargs["options"], {"triton.cudagraphs": False})
        self.assertTrue(compile_call.call_args.kwargs["fullgraph"])

    def test_separate_medians_do_not_replace_paired_differences(self):
        result = native_triton_comparison([1., 100., 101.], [2., 3., 102.])
        self.assertGreater(result["ratio_of_separate_medians"], 1)
        self.assertEqual(result["paired_median_saving_us"], -1)
        self.assertEqual(result["triton_faster_rounds"], 1)

    def test_ties_and_invalid_samples(self):
        result = native_triton_comparison([2., 3.], [2., 3.])
        self.assertEqual(result["tied_rounds"], 2)
        self.assertEqual(result["triton_faster_rounds"], 0)
        for native, triton in (([], []), ([1.], [1., 2.]), ([0.], [1.])):
            with self.assertRaises(ValueError):
                native_triton_comparison(native, triton)


if __name__ == "__main__":
    unittest.main()
