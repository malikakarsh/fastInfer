"""Dispatch must stay within the measured shape/dtype/hardware/software scope."""

import unittest

from fastinfer.kernels.softmax_config import choose_softmax_warps


class SoftmaxConfigTest(unittest.TestCase):
    def test_confirmed_shapes_and_scope_fallback(self):
        context = dict(dtype="float16", gpu_name="Tesla T4",
                       torch_version="2.5.1+cu121", triton_version="3.1.0")
        for shape, expected in (((1, 65), 1), ((128, 128), 1),
                                ((128, 1536), 2), ((16, 128), 4), ((1, 8192), 4)):
            self.assertEqual(choose_softmax_warps(shape, **context), expected)
        for key, value in (("dtype", "float32"), ("gpu_name", "NVIDIA A100"),
                           ("torch_version", "2.6.0+cu124"), ("triton_version", "3.2.0")):
            with self.subTest(key=key):
                self.assertEqual(choose_softmax_warps((1, 65), **(context | {key: value})), 4)


if __name__ == "__main__":
    unittest.main()
