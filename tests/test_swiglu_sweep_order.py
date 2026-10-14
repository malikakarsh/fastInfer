"""Sweeping providers must not favor a fixed timing position."""

from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
try:
    from swiglu_compare import sweep_orders
finally:
    sys.path.pop(0)


class SwigluSweepOrderTest(unittest.TestCase):
    def test_selection_and_confirmation_positions_are_balanced(self):
        # Seven variants plus two baselines in selection; four plus two in confirmation.
        for count in (8, 9, 6):
            with self.subTest(provider_count=count):
                names = tuple(f"provider{i}" for i in range(count))
                orders = sweep_orders(names)
                self.assertEqual(len(orders), 30)
                for order in orders:
                    self.assertEqual(set(order), set(names))
                for position in range(count):
                    counts = Counter(order[position] for order in orders)
                    self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)


if __name__ == "__main__":
    unittest.main()
