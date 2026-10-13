# FastInfer

FastInfer is an experimental LLM inference runtime centered on custom Triton GPU kernels and reproducible performance measurements. Python handles orchestration and benchmarking, while PyTorch provides reference implementations.

## Implementation scope

- **Triton:** GPU kernels for matrix multiplication and transformer operations, compared against PyTorch baselines.
- **Python:** model execution, KV-cache management, batching, benchmark runners, and tests.
- **C++:** selected runtime components, such as the page allocator or scheduler data structures, after their Python implementations are validated.

## Implemented

- A PyTorch reference for two-dimensional matrix multiplication.
- A tiled float16 Triton matrix multiplication with K-block accumulation, validated against PyTorch on tested CUDA shapes.
- A specialized single-row FP16 matrix–vector kernel with FP32 reductions for contiguous stored linear weights.
- A readable PyTorch RMSNorm reference and an initial fused Triton implementation preserving Qwen's rounding order; all seven CPU/GPU checks passed on the tested setup, with a first isolated performance comparison recorded below.
- CPU correctness tests for the reference and environment reporter.
- CUDA correctness tests for the Triton kernel, including partial tiles and multiple K blocks.
- A CUDA-event benchmark for the PyTorch matmul baseline.
- A paired PyTorch-versus-Triton graph-replay benchmark runner.
- An environment reporter that records hardware and software versions.

Measured results cover the [PyTorch matmul baseline](notes/matmul_baseline_analysis.md) and an initial [paired PyTorch–Triton comparison](notes/matmul_triton_comparison.md).

## Specialized matrix–vector result

Implemented a specialized FP16 matrix–vector kernel achieving approximately 1.11–1.20× speedup over PyTorch 2.5.1 across five tested widths on a T4, using CUDA-graph timing and passing FP32-reference accuracy checks.

The frozen configuration processes one stored weight row per Triton program with four warps. Inputs and outputs are FP16; products and reductions use FP32. It computes `a @ weight.T` for `a[1, 1536]` and contiguous `weight[N, 1536]`. Confirmation covered `N = 256, 1536, 8192, 8960, 9216`, with three new random seeds (2, 3, 4) and no retuning. PyTorch's `allow_fp16_reduced_precision_reduction` remained enabled. Both implementations had zero mismatches under `rtol=0.01, atol=0.01` against an FP32 reference in all 15 paired comparisons.

| Output width N | PyTorch median range (µs) | GEMV median range (µs) | Paired speedup range |
| ---: | ---: | ---: | ---: |
| 256 | 4.10–4.13 | 3.41–3.44 | 1.201× |
| 1536 | 22.16–22.24 | 19.99–20.04 | 1.106–1.112× |
| 8192 | 107.58–107.80 | 91.17–91.42 | 1.177–1.181× |
| 8960 | 117.14–117.30 | 99.24–99.54 | 1.178–1.180× |
| 9216 | 120.10–120.51 | 102.04–102.26 | 1.177–1.179× |

Ranges span three round medians, not confidence intervals. Measurements used one Tesla T4, Python 3.12.15, PyTorch 2.5.1+cu121, and Triton 3.1.0, with 100 operations per graph replay and 30 alternating timing samples per implementation per round. These are synthetic FP16, single-row, fixed-input graph benchmarks; a second confirmation in a fresh GPU session also won all 15 paired comparisons with zero accuracy mismatches. They do not establish an eager-call, batched-matmul, or end-to-end model speedup. Widths 8192 and 9216 are neighboring synthetic checks; 256, 1536 and 8960 match selected Qwen2.5-1.5B projection dimensions. Those synthetic runs did not test actual model weights or native BF16 execution.

The [console-summary data](results/raw/gemv_confirmation_20261012T230940025682Z_console_summary.json) records the supplied rounded medians. The full raw-sample JSON remains on Kaggle and has not yet been archived here. The [fresh-session console summary](results/raw/gemv_confirmation_20261013T040026340595Z_console_summary.json) records speedups of 1.105–1.203× across the same five widths and seeds, using the frozen configuration. Original full raw samples from both sessions still need to be archived here. A subsequent real-weight projection check is recorded below; an exploratory full decode measurement is recorded below.

Reproduce with the [confirmation notebook](benchmarks/matmul_gemv_confirm.ipynb), or run `python benchmarks/matmul_gemv_compare.py --confirm` in the configured GPU environment. See the [benchmark instructions](benchmarks/README.md#frozen-gemv-confirmation).

## Real Qwen projection and decode integration

On three real decode inputs from Qwen2.5-1.5B, the first layer's `up_proj` took 116.56–116.60 µs in PyTorch versus 99.14–99.30 µs with our frozen GEMV (1.174–1.176×). Captured model outputs, PyTorch replay and GEMV all passed the FP32-reference tolerance. The model was loaded in FP16 with math SDPA attention; native BF16 execution was not tested. These are isolated CUDA-graph projection timings, not a whole-model speedup. The [console summary](results/raw/qwen_real_linear_20261013T042024726496Z_console_summary.json) preserves the supplied rounded measurements; full raw samples remain on Kaggle.

An opt-in [decode integration](decode_linear.py) now routes that one projection through GEMV for eligible single-token inputs and restores the original forward afterwards. The [full decode comparison](benchmarks/qwen_decode_compare.py) checks logits and greedy token sequences before measuring the complete cached decode loop. An exploratory T4 run completed: baseline repeats were identical, all per-step projection FP32 checks passed, and greedy tokens matched, but strict logits tolerance failed. Complete decode took roughly 33 ms per step; the custom version won only 11–12 of 30 paired rounds per prompt, with negative paired median savings. This establishes no consistent whole-model speedup. See the [rounded console summary](results/raw/qwen_decode_20261013T051943862663Z_console_summary.json). See the [methodology](benchmarks/README.md#one-projection-decode-integration).

## Fused RMSNorm first result

The initial FP16 fused RMSNorm comparison on the Kaggle T4 passed all seven tests and all in-run FP32/PyTorch accuracy checks. Across seven synthetic shapes, the kernel measured 5.441–10.481× speedup over the Qwen-style PyTorch reference using amortized CUDA-graph timing. At hidden width1536, PyTorch/Triton medians were32.23/3.54µs for one token,23.28/2.33µs for16tokens and32.70/3.12µs for128tokens. This is a first-run isolated comparison, not a model-level improvement or a comparison against compiled PyTorch. A separate same-seed repeat also won all seven shapes (5.399–10.878×), with all accuracy comparisons passing. Comparisons against compiled PyTorch and real-model integration remain pending. The [console summary](results/raw/rmsnorm_first_comparison_20261013T055701423727Z_console_summary.json) stores rounded supplied measurements; full timing samples remain on Kaggle.

## Setup and tests

Requires Python 3.10 or newer. PyTorch is the only required package; optional dependencies are defined in `pyproject.toml`.

```bash
python -m pip install -e '.[test]'
python -m unittest discover -s tests -v
```

## GPU benchmark

Run from the repository directory in an environment with a CUDA GPU and PyTorch:

```bash
python scripts/environment_report.py
PYTHONPATH=.. python benchmarks/matmul_baseline.py
```

The benchmark defaults to a float16 multiplication of a `1 x 256` matrix by a `256 x 256` matrix. It uses 10 warmup runs and 30 timed repetitions, then writes the latency samples and environment metadata to `results/raw/`. Use `--m`, `--k`, `--n`, `--dtype`, `--warmup`, and `--repetitions` to change the workload.

For small workloads, `--timing-mode graph --graph-ops 100` reports an amortized CUDA graph replay time. It is recorded separately from the default eager timing mode; see `benchmarks/README.md` for the measurement distinction.

To compare PyTorch and Triton on the same float16 inputs and graph timing method, run:

```bash
PYTHONPATH=.. python benchmarks/matmul_compare.py --m 1 --k 256 --n 256
```

The comparison checks correctness before timing and saves samples for both implementations in one JSON file. See `benchmarks/README.md` for the method and limits.

The environment report should be checked before installing optional GPU dependencies, so the existing PyTorch and CUDA setup remains compatible.

## Repository layout

| Path | Contents |
| --- | --- |
| `kernels/` | Reference operations and GPU kernels |
| `tests/` | Correctness checks |
| `benchmarks/` | Benchmark runners and methodology |
| `scripts/` | Environment reporting and utilities |
| `results/raw/` | Saved measurements with run metadata |
| `notes/` | Technical findings tied to implementations and results |
