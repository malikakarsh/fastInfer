"""Identify static matrix and scalar FMA instructions in generated NVIDIA PTX."""

import re
from collections import Counter


def summarize_ptx(ptx: str | None) -> dict:
    """Count instruction statements, excluding comments; these are not runtime counts."""
    if ptx is None:
        return {"available": False}
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", ptx, flags=re.DOTALL)
    opcodes = re.findall(
        r"^\s*(?:@!?%\w+\s+)?([a-z][\w:]*(?:\.[\w:]+)+)\s",
        code, flags=re.MULTILINE,
    )
    matrix = Counter(op for op in opcodes if op.startswith(
        ("mma.", "wmma.mma.", "wgmma.mma_async.", "tcgen05.mma.")
    ))
    fma = Counter(op for op in opcodes if op.startswith("fma."))
    return {
        "available": True,
        "matrix_instruction_statements": sum(matrix.values()),
        "matrix_opcodes": dict(matrix),
        "fma_instruction_statements": sum(fma.values()),
        "fma_opcodes": dict(fma),
    }
