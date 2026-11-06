"""Forward-only split-half RoPE rotation using precomputed cosine/sine tables."""

import torch
import triton
import triton.language as tl

from .rope import validate_rope_inputs


@triton.jit
def _rope(Q, K, COS, SIN, OUT_Q, OUT_K,
          WIDTH: tl.constexpr, TOKENS: tl.constexpr,
          Q_HEADS: tl.constexpr, K_HEADS: tl.constexpr,
          Q_ROWS: tl.constexpr, BLOCK: tl.constexpr):
    # Each program rotates one token vector from one Q or K head.
    row = tl.program_id(0)
    if row < Q_ROWS:
        batch = row // (Q_HEADS * TOKENS)
        token = row % TOKENS
        source = Q + row * WIDTH
        destination = OUT_Q + row * WIDTH
    else:
        k_row = row - Q_ROWS
        batch = k_row // (K_HEADS * TOKENS)
        token = k_row % TOKENS
        source = K + k_row * WIDTH
        destination = OUT_K + k_row * WIDTH

    features = tl.arange(0, BLOCK)
    valid = features < WIDTH
    first_half = features < WIDTH // 2
    partner_features = tl.where(first_half, features + WIDTH // 2, features - WIDTH // 2)
    value = tl.load(source + features, mask=valid, other=0)
    partner = tl.load(source + partner_features, mask=valid, other=0).to(tl.float32)
    table_offset = (batch * TOKENS + token) * WIDTH + features
    cos = tl.load(COS + table_offset, mask=valid, other=0).to(tl.float32)
    sin = tl.load(SIN + table_offset, mask=valid, other=0).to(tl.float32)
    rotated_half = tl.where(first_half, -partner, partner)

    # Eager reference stores each product in the input dtype before adding.
    # Keep those FP16 rounding boundaries; disable multiply-add fusion below.
    product_a = (value.to(tl.float32) * cos).to(value.dtype).to(tl.float32)
    product_b = (rotated_half * sin).to(value.dtype).to(tl.float32)
    tl.store(destination + features, product_a + product_b, mask=valid)


def triton_rope(q, k, cos, sin):
    """Rotate contiguous CUDA FP16/FP32 Q/K with even head widths 2..256.

    Q/K: [batch, heads, tokens, head_dim], possibly different head counts.
    cos/sin: [batch, tokens, head_dim], same device/dtype as Q/K.
    Tables come from rope_cos_sin outside this function. Inputs are unchanged.
    Finite inputs required; no backward pass or model integration provided.
    """
    validate_rope_inputs(q, k, cos, sin)
    if not q.is_cuda:
        raise ValueError("CUDA inputs required")
    if any(not x.is_contiguous() for x in (q, k, cos, sin)):
        raise ValueError("contiguous Q/K/cos/sin required")
    if q.shape[-1] > 256:
        raise ValueError("initial kernel supports even head_dim from 2 through 256")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, cos, sin)):
        raise RuntimeError("forward-only kernel; disable gradients")

    out_q, out_k = torch.empty_like(q), torch.empty_like(k)
    batch, q_heads, tokens, width = q.shape
    k_heads = k.shape[1]
    q_rows = batch * q_heads * tokens
    rows = q_rows + batch * k_heads * tokens
    _rope[(rows,)](
        q, k, cos, sin, out_q, out_k, width, tokens,
        q_heads, k_heads, q_rows, triton.next_power_of_2(width),
        num_warps=4, enable_fp_fusion=False,
    )
    return out_q, out_k
