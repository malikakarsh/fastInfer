"""Scoped single-token Qwen MLP replacement for elementwise-stage experiments."""

from contextlib import ExitStack, contextmanager

import torch


def eligible_swiglu_input(mlp, x):
    """Only dispatch the original SiLU, bias-free, FP16 single-token MLP path."""
    if (torch.is_grad_enabled() or mlp.training or tuple(x.shape) != (1, 1, 1536)
            or x.dtype != torch.float16 or not x.is_cuda or not x.is_contiguous()
            or not isinstance(mlp.act_fn, torch.nn.SiLU) or mlp.act_fn.inplace):
        return False
    for linear, shape in ((mlp.gate_proj, (8960, 1536)),
                          (mlp.up_proj, (8960, 1536)),
                          (mlp.down_proj, (1536, 8960))):
        if (linear.training or linear.bias is not None or tuple(linear.weight.shape) != shape
                or linear.weight.dtype != x.dtype or linear.weight.device != x.device
                or not linear.weight.is_contiguous()):
            return False
    return True


@contextmanager
def use_decode_swiglu(mlps, *, kernel, capture=None):
    """Replace only SiLU(gate)*up; restore every MLP on normal/error exit.

    Prefill must occur outside this context. Unsupported calls fall back.
    Intended for serial inference experiments, not concurrent serving.
    Optional capture receives(name,gate,up,product), only during validation.
    """
    counts = {"stage_calls": 0, "fallback_calls": 0,
              "modules": {name: {"stage_calls": 0, "fallback_calls": 0} for name in mlps}}

    def replacement(mlp, name, original):
        def forward(hidden_states):
            if not eligible_swiglu_input(mlp, hidden_states):
                counts["fallback_calls"] += 1
                counts["modules"][name]["fallback_calls"] += 1
                return original(hidden_states)
            gate = mlp.gate_proj(hidden_states)
            up = mlp.up_proj(hidden_states)
            product = kernel(gate.reshape(1, 8960), up.reshape(1, 8960))
            if capture is not None:
                capture(name, gate, up, product.reshape(1, 1, 8960))
            counts["stage_calls"] += 1
            counts["modules"][name]["stage_calls"] += 1
            return mlp.down_proj(product.reshape(1, 1, 8960))
        return forward

    def restore(module, had_override, override):
        if had_override:
            module.forward = override
        else:
            del module.forward

    with ExitStack() as cleanup:
        for name, mlp in mlps.items():
            original = mlp.forward
            cleanup.callback(restore, mlp, "forward" in mlp.__dict__, mlp.__dict__.get("forward"))
            mlp.forward = replacement(mlp, name, original)
        yield counts
