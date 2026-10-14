"""Scheduling choice confirmed for one FP16 shape on the pinned T4 stack."""


def choose_swiglu_config(shape, *, dtype, gpu_name, torch_version, triton_version):
    """Use 512/4 only within the measured scope, otherwise the original256/4."""
    if (tuple(shape) == (16, 8960) and dtype == "float16" and gpu_name == "Tesla T4"
            and torch_version == "2.5.1+cu121" and triton_version == "3.1.0"):
        return 512, 4
    return 256, 4
