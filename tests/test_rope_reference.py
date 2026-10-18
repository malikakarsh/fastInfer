"""Small rotation examples and properties before writing a GPU kernel."""

import math
import unittest

import torch

from fastinfer.kernels.rope import reference_rope, rope_cos_sin


class RopeReferenceTest(unittest.TestCase):
    def test_known_quarter_turn_and_split_half_pairs(self):
        # Pair (5,1) and pair (2,3), arranged in Qwen's split-half layout.
        q = torch.tensor([[[[5., 2., 1., 3.]]]])
        k = 2 * q
        cos, sin = torch.zeros(1, 1, 4), torch.ones(1, 1, 4)
        actual_q, actual_k = reference_rope(q, k, cos, sin)
        expected = torch.tensor([[[[-1., -3., 5., 2.]]]])
        torch.testing.assert_close(actual_q, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_k, 2 * expected, rtol=0, atol=0)
        self.assertEqual(q.flatten().tolist(), [5., 2., 1., 3.])

    def test_position_zero_and_different_pair_rates(self):
        positions = torch.tensor([[0, 1, 2]])
        # dim=4, theta=100: rates are 1 and 0.1 radians/position.
        cos, sin = rope_cos_sin(positions, 4, theta=100)
        angles = torch.tensor([[[0., 0., 0., 0.], [1., .1, 1., .1], [2., .2, 2., .2]]])
        torch.testing.assert_close(cos, angles.cos())
        torch.testing.assert_close(sin, angles.sin())
        q, k = torch.randn(1, 3, 3, 4), torch.randn(1, 1, 3, 4)
        rotated_q, rotated_k = reference_rope(q, k, cos, sin)
        torch.testing.assert_close(rotated_q[:, :, 0], q[:, :, 0], rtol=0, atol=0)
        torch.testing.assert_close(rotated_k[:, :, 0], k[:, :, 0], rtol=0, atol=0)

    def test_matches_independent_scalar_rotation(self):
        torch.manual_seed(31)
        positions = torch.tensor([[0, 2, 7], [3, 1, 9]])
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                q = torch.randn(2, 3, 3, 8).to(dtype)
                k = torch.randn(2, 1, 3, 8).to(dtype)
                cos, sin = rope_cos_sin(positions, 8, theta=10000, dtype=dtype)
                rotated = reference_rope(q, k, cos, sin)
                for x, actual in zip((q, k), rotated):
                    expected = torch.empty_like(x)
                    for batch in range(2):
                        for head in range(x.shape[1]):
                            for token in range(3):
                                for pair in range(4):
                                    angle = int(positions[batch, token]) * 10000 ** (-2 * pair / 8)
                                    c = torch.tensor(math.cos(angle), dtype=dtype)
                                    s = torch.tensor(math.sin(angle), dtype=dtype)
                                    a, b = x[batch, head, token, pair], x[batch, head, token, pair + 4]
                                    expected[batch, head, token, pair] = a * c - b * s
                                    expected[batch, head, token, pair + 4] = b * c + a * s
                    torch.testing.assert_close(actual, expected,
                                               rtol=.002 if dtype == torch.float16 else 2e-6,
                                               atol=.002 if dtype == torch.float16 else 2e-6)

    def test_norm_and_shared_position_shift(self):
        torch.manual_seed(32)
        q, k = torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8)
        scores = []
        for shift in (0, 3):
            cq, sq = rope_cos_sin(torch.tensor([[2 + shift]]), 8)
            ck, sk = rope_cos_sin(torch.tensor([[5 + shift]]), 8)
            # Rotate Q and K at their respective positions, using the same rates.
            qr, _ = reference_rope(q, q, cq, sq)
            kr, _ = reference_rope(k, k, ck, sk)
            torch.testing.assert_close(qr.square().sum(-1), q.square().sum(-1), rtol=2e-6, atol=2e-6)
            scores.append((qr * kr).sum(-1))
        torch.testing.assert_close(scores[0], scores[1], rtol=2e-6, atol=2e-6)

    def test_input_contract(self):
        for positions, width in ((torch.zeros(2), 4), (torch.empty(1, 0, dtype=torch.int64), 4),
                                 (torch.zeros(1, 1, dtype=torch.int64), 3)):
            with self.assertRaises((ValueError, TypeError)):
                rope_cos_sin(positions, width)
        for theta in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                rope_cos_sin(torch.tensor([[0]]), 4, theta=theta)
        q = torch.ones(1, 2, 1, 4)
        cos, sin = rope_cos_sin(torch.tensor([[0]]), 4)
        for k, c, s in ((q[..., :3], cos, sin), (q, cos[:, :, :2], sin),
                        (q.half(), cos, sin)):
            with self.assertRaises(ValueError):
                reference_rope(q, k, c, s)
        with self.assertRaises(TypeError):
            rope_cos_sin(torch.tensor([[0.]]), 4)


if __name__ == "__main__":
    unittest.main()
