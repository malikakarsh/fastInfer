"""Automatic selection must remain within the measured scope."""

import unittest

from fastinfer.kernels.swiglu_config import choose_swiglu_config


class SwigluConfigTest(unittest.TestCase):
    def test_confirmed_shape_and_scope_fallback(self):
        context = dict(dtype="float16", gpu_name="Tesla T4",
                       torch_version="2.5.1+cu121", triton_version="3.1.0")
        self.assertEqual(choose_swiglu_config((16, 8960), **context), (512, 4))
        for shape in ((1, 8960), (128, 8960), (16, 8192), (1, 16, 8960)):
            self.assertEqual(choose_swiglu_config(shape, **context), (256, 4))
        for key, value in (("dtype", "float32"), ("gpu_name", "NVIDIA A100"),
                           ("torch_version", "2.6.0+cu124"), ("triton_version", "3.2.0")):
            with self.subTest(key=key):
                self.assertEqual(choose_swiglu_config((16, 8960),
                                 **(context | {key: value})), (256, 4))


if __name__ == "__main__":
    unittest.main()
