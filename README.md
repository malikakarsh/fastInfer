# FastInfer

FastInfer is an experimental LLM inference runtime centered on custom Triton GPU kernels and reproducible performance measurements. Python handles orchestration and benchmarking, while PyTorch provides reference implementations.

## Implementation scope

- **Triton:** GPU kernels for matrix multiplication and transformer operations, compared against PyTorch baselines.
- **Python:** model execution, KV-cache management, batching, benchmark runners, and tests.
- **C++:** selected runtime components, such as the page allocator or scheduler data structures, after their Python implementations are validated.

## Implemented

- A PyTorch reference for two-dimensional matrix multiplication.
- CPU correctness tests for the reference and environment reporter.
- A CUDA-event benchmark for the PyTorch matmul baseline.
- An environment reporter that records hardware and software versions.

Current measured results cover the [PyTorch matmul baseline](notes/matmul_baseline_analysis.md). Triton kernels and serving components will be reported after their implementations and comparisons are validated.

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
