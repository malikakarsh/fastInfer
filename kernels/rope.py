"""Readable default RoPE reference with Qwen's split-half feature pairing.

Q/K layout: [batch, heads, tokens, head_dim]. Q and K may have different
head counts. This is fixed-frequency RoPE only, without scaling variants.
"""

import math

import torch

DTYPES = (torch.float16, torch.float32)


def rope_cos_sin(position_ids, head_dim, *, theta=10000.0, dtype=torch.float32):
    """Build [batch, tokens, head_dim] tables; angles are measured in radians.

position_ids is [batch, tokens], containing nonnegative integer positions.
The model's configured theta must be passed when using actual model inputs.
Frequencies/trigonometry use FP32; coefficients are then cast to dtype.
"""
    if position_ids.ndim != 2 or position_ids.numel() == 0:
        raise ValueError("position_ids must have nonempty [batch, tokens] shape")
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("integer position_ids required")
    if not isinstance(head_dim, int) or isinstance(head_dim, bool) or head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim must be a positive even integer")
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError("theta must be finite and positive")
    if dtype not in DTYPES:
        raise TypeError("FP16 or FP32 coefficients required")

    # One rate for each pair: theta ** (-2 * pair_index / head_dim).
    with torch.autocast(device_type=position_ids.device.type, enabled=False):
        pair_indices = torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32)
        inverse_frequencies = 1.0 / (theta ** (pair_indices / head_dim))
        angles = position_ids.float().unsqueeze(-1) * inverse_frequencies
        # Both members of each split-half pair receive the same angle.
        angles = torch.cat((angles, angles), dim=-1)
        cos, sin = angles.cos(), angles.sin()
    return cos.to(dtype), sin.to(dtype)


def reference_rope(q, k, cos, sin):
    """Rotate Q/K using precomputed tables, preserving input dtype arithmetic.

For a vector [a,b,c,d], pairs are (a,c) and (b,d), not adjacent features.
No V tensor is passed or changed. Finite floating inputs are expected.
Coefficient construction is separate so it can be measured separately later.
"""
    if q.ndim != 4 or k.ndim != 4 or q.numel() == 0 or k.numel() == 0:
        raise ValueError("Q/K require nonempty [batch, heads, tokens, head_dim] shapes")
    batch, _, tokens, width = q.shape
    if width % 2 or (k.shape[0], k.shape[2], k.shape[3]) != (batch, tokens, width):
        raise ValueError("Q/K must share batch, tokens and a positive even head_dim")
    if cos.shape != (batch, tokens, width) or sin.shape != cos.shape:
        raise ValueError("cos/sin must have [batch, tokens, head_dim] shape")
    if q.dtype not in DTYPES:
        raise TypeError("FP16 or FP32 Q/K required")
    if any(x.dtype != q.dtype or x.device != q.device for x in (k, cos, sin)):
        raise ValueError("Q/K/cos/sin must share dtype and device")

    # Insert the head dimension: each head uses the same positional tables.
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

    def rotate(x):
        left, right = x[..., :width // 2], x[..., width // 2:]
        rotated_half = torch.cat((-right, left), dim=-1)
        return x * cos + rotated_half * sin

    return rotate(q), rotate(k)
