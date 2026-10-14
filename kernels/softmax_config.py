"""Warp choices confirmed for specific FP16 shapes on the pinned T4 stack."""


def choose_softmax_warps(shape, *, dtype, gpu_name, torch_version, triton_version):
    """Use four warps outside the measured hardware, software and shapes."""
    if (dtype != "float16" or gpu_name != "Tesla T4"
            or torch_version != "2.5.1+cu121" or triton_version != "3.1.0"):
        return 4
    return {(1, 65): 1, (128, 128): 1, (128, 1536): 2}.get(tuple(shape), 4)
