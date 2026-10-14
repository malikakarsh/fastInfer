"""The five-provider sweep must balance order rather than favor one variant."""

from collections import Counter
from pathlib import Path
import sys
import unittest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS))
try:
    from softmax_warp_check import timing_orders
finally:
    sys.path.pop(0)


class WarpOrderTest(unittest.TestCase):
    def test_every_provider_has_equal_position_counts(self):
        self.check_positions(("native", "w1", "w2", "w4", "w8"))
        self.check_positions(("native", "w4", "candidate"))

    def check_positions(self, names):
        orders = timing_orders(names)
        self.assertEqual(len(orders), 30)
        for order in orders:
            self.assertEqual(set(order), set(names))
        for position in range(len(names)):
            self.assertEqual(Counter(order[position] for order in orders),
                             Counter({name: 30 // len(names) for name in names}))


if __name__ == "__main__":
    unittest.main()
