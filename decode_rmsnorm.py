"""Scoped inference-only replacement of Qwen's single-token normalization."""

from contextlib import ExitStack, contextmanager

import torch


def eligible_norm_input(module, x):
    return (
        not torch.is_grad_enabled()
        and not module.training
        and tuple(x.shape) == (1, 1, 1536)
        and tuple(module.weight.shape) == (1536,)
        and x.dtype == module.weight.dtype == torch.float16
        and x.is_cuda
        and x.device == module.weight.device
        and x.is_contiguous()
        and module.weight.is_contiguous()
    )


@contextmanager
def use_decode_rmsnorm(norms, *, kernel=None):
    """Temporarily replace a mapping of normalization modules during decode.

    Other inputs fall back. Restore all modules even on failure. Intended for
    serial experiments; no backward implementation or concurrent mutation.
    """
    if kernel is None:
        from fastinfer.kernels.triton_rmsnorm import rmsnorm
        kernel = rmsnorm
    counts = {"rmsnorm_calls": 0, "fallback_calls": 0,
              "modules": {name: {"rmsnorm_calls": 0, "fallback_calls": 0} for name in norms}}

    def replacement(module, name, original):
        def forward(hidden_states):
            if eligible_norm_input(module, hidden_states):
                counts["rmsnorm_calls"] += 1
                counts["modules"][name]["rmsnorm_calls"] += 1
                return kernel(hidden_states, module.weight, eps=module.variance_epsilon)
            counts["fallback_calls"] += 1
            counts["modules"][name]["fallback_calls"] += 1
            return original(hidden_states)
        return forward

    def restore(module, had_override, override):
        if had_override:
            module.forward = override
        else:
            del module.forward

    with ExitStack() as cleanup:
        for name, module in norms.items():
            original = module.forward
            had_override = "forward" in module.__dict__
            override = module.__dict__.get("forward")
            cleanup.callback(restore, module, had_override, override)
            module.forward = replacement(module, name, original)
        yield counts
