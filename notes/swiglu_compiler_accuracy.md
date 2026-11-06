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

## Checking whether the upstream fix is installed

To compare Kaggle's installed build against a separate fixed build:

```bash
python scripts/kaggle_pytorch_cast_fix_check.py
```

The runner creates `/kaggle/working/pytorch_cast_fix_env`, installs the pinned
official `torch==2.13.0+cu126` wheel and its dependencies there, and runs both
builds with default and emulated precision policies in separate processes and
compiler caches. Kaggle Internet and a GPU are required. Downloads and the
environment require several GB of disk space. The notebook interpreter's
packages are unchanged; no kernel restart is required. A CUDA smoke test checks
that the isolated build works on the assigned GPU before diagnostics start.

Reports and `summary.json` are saved under
`/kaggle/working/pytorch_cast_fix_results`. Expected baseline strict failures
do not stop the four-run comparison. The final check requires explicit casts
to pass in `fixed_emulated`; other strict failures are reported separately.
Use a new `--output-dir` to repeat without overwriting earlier results.
The comparison changes Torch, Triton and CUDA dependencies together, so a pass
on the new build supports version-level resolution, not single-commit causality.
This runner does not run the full model or establish exact model logits.

The [official CUDA 12.6 wheel index](https://download.pytorch.org/whl/cu126/torch/)
lists the PyTorch 2.13.0 Linux x86_64 Python 3.13 wheel. CUDA 12.6 is used here
because the CUDA 12.8 index does not list that version.

```bash
python scripts/check_pytorch_cast_fix.py --output results/raw/pytorch_cast_fix_audit.json
```

This read-only audit saves the installed `to_dtype` and `_convert_element_type`
lowering functions and asks the GitHub compare API whether the original fix
commit is an ancestor of `torch.version.git_version`. Use `--offline` to skip
the API query. Network/API failures mean ancestry is unknown. Missing ancestry
does not exclude a cherry-picked equivalent fix, so the installed source and
runtime cast tests must also be considered.

For the reported build `70d99e998b4955e0049d13a98d77ae1b14db1f45`
(PyTorch 2.11.0), upstream source lacks the explicit-cast precision guard
visible in PyTorch 2.13.0. The 2.13.0 release notes list the cast fix as #180575.
This is consistent with an unfixed older build, rather than evidence of a new
regression. The audit fingerprints that later source pattern; it cannot identify
every equivalent implementation or prove runtime correctness by source alone.

Sources: [reported build's lowering](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/torch/_inductor/lowering.py),
[2.13.0 lowering](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/_inductor/lowering.py),
[upstream issue](https://github.com/pytorch/pytorch/issues/179561),
[2.13.0 release notes](https://github.com/pytorch/pytorch/releases/tag/v2.13.0).

PyTorch documents that mathematical equivalence does not guarantee bitwise
identity in its [numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
