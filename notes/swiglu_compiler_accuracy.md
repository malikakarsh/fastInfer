# Investigating eager/compiled numerical differences

Run from the `fastinfer` directory on the original GPU and pinned software stack:

```bash
python scripts/swiglu_compiler_accuracy.py --device cuda --strict
python -m unittest discover -s tests -p test_swiglu_compiler_accuracy.py -v
```

The standalone script needs PyTorch and a working CUDA Inductor installation;
it does not import our Triton kernels or download a model. Its default backends
are `eager`, `aot_eager`, and `inductor`, all through `torch.compile` with
`fullgraph=True` and `dynamic=False`. The `eager` backend is a graph-capture
control; `aot_eager` adds AOT transformations; Inductor adds code generation.
Each is compared against ordinary uncompiled eager execution.

The JSON report defaults to `results/raw/swiglu_compiler_accuracy.json` (a rerun
overwrites that file; use `--output PATH` to preserve separate runs). It records
versions, device, compiler settings, errors, mismatch counts, sample values,
repeat stability, and comparisons to both FP64 references.

The test cases cover:

- Explicit FP32 → FP16 → FP32 casts around representational boundaries. Inputs
  and power-of-two multipliers are chosen so expected cast behavior can be
  checked exactly, without an exponential or reduction clouding the result.
- The same explicit cast followed by multiplication, to expose fusion that
  removes an intermediate rounding boundary.
- SwiGLU at three fresh seeds and short/decode/prefill shapes.
- Every finite FP16 gate value, with multiplier 0.75, including subnormals and
  extreme negative values. This does not exhaust all gate/multiplier pairs.

For SwiGLU, the FP64 references start from the exact same FP16 inputs. One
rounds only the final product; the other also rounds the SiLU intermediate to
FP16. They are high-precision numerical references, not a proof of correctly
rounded sigmoid. The default tolerance (.002 relative, .0002 absolute) matches
our existing SwiGLU checks; it is our investigation threshold, not a documented
universal PyTorch guarantee. Exact comparisons mean equal finite values and
do not distinguish positive and negative zero.

Interpret results as follows:

| Status | Interpretation |
| --- | --- |
| `explicit_rounding_preserved` | The focused cast contract passed. |
| `potential_cast_semantics_bug` | Eager satisfies the explicit cast reference, but compilation differs. Reduce to this case and review the compiler's precision policy before filing. |
| `numerical_drift_within_tolerance` | SwiGLU differs from eager within our chosen tolerance; this alone does not establish a bug. |
| `investigate_tolerance_failure` | SwiGLU exceeded our tolerance; inspect both oracle comparisons to separate rounding effects from approximation error. |
| `investigate_contract_or_nonfinite_output` | Shape, dtype, or finite-output expectations failed. |
| `investigate_unstable_repeat` | Identical inputs produced different repeated outputs. |
| `reference_check_failed` | The eager control did not satisfy the constructed reference; investigate that before blaming compilation. |
| `execution_error` | Backend execution failed or is unavailable; the report contains the exception. This is not classified as a numerical bug. |
| `matches_eager` | No finite-value mismatch for that test. |

`--strict` exits nonzero for investigation statuses, reference failures, and
execution errors. It permits SwiGLU drift within tolerance. The CUDA unittest
enforces the explicit rounding cases; CPU tests verify the reference logic.
Passing this suite does not establish whole-model equivalence or absence of
compiler bugs. Our previous model-trajectory checks remain separate.

If only Inductor fails, that localizes the observation to its path; it does not
by itself prove which compiler pass is responsible. Retest on a current stable
PyTorch environment separately from the original pinned environment. Attach
this standalone script, command, and JSON to any issue. Generated source can
also be collected with `TORCH_LOGS=output_code` when supported by that version.
CPU investigation is available with `--device cpu`; it cannot validate CUDA
behavior.

PyTorch documents that mathematical equivalence does not guarantee bitwise
identity in its [numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
