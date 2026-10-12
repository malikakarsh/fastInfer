# Benchmarks

```bash
PYTHONPATH=.. python benchmarks/matmul_baseline.py
```

The script saves JSON in `results/raw/`. It defaults to a small `1 x 256` by `256 x 256` float16 matmul, with 10 warmups and 30 timed repetitions. Use `--m`, `--k`, `--n`, and `--dtype` to change the workload.

The default `--timing-mode eager` preserves the original method: one Python-submitted matmul between CUDA events. For tiny workloads, that interval can include host submission gaps. To reduce this effect, run the separately labeled graph-replay method:

```bash
PYTHONPATH=.. python benchmarks/matmul_baseline.py --timing-mode graph --graph-ops 100
```

Graph mode captures 100 matmuls, times each replay, and divides replay time by 100. This reports amortized time per operation under graph replay, not eager end-to-end latency or isolated kernel duration. Compare results only when their timing modes match.

## PyTorch versus Triton

Run the paired comparison on a CUDA machine with Triton installed:

```bash
PYTHONPATH=.. python benchmarks/matmul_compare.py --m 1 --k 256 --n 256
```

The script checks the Triton result against a float32 PyTorch matmul of the same float16 inputs before timing. It also records each implementation's mismatch count and maximum absolute error against that reference. PyTorch's float16 result remains the timing baseline, even when it differs from the float32 reference. The benchmark captures 100 operations per implementation by default, warms both graphs, then alternates their measurement order across 30 paired repetitions. Both receive the same float16 inputs. The JSON file records every sample, timing parameters, environment, and correctness tolerance. These graph-replay measurements use repeated fixed inputs and do not represent eager-call latency. Compare the two implementations within this paired run; the older standalone PyTorch results used a different sample order.

To compare the existing K tile of 32 with a K tile of 64, run:

```bash
PYTHONPATH=.. python benchmarks/matmul_compare.py --m 16 --k 256 --n 256 --block-k-sweep
```

This mode validates both Triton variants against the float32 reference on the same inputs, captures a separate graph for each of the three implementations, and rotates their measurement order. Only `BLOCK_K` differs between the two Triton variants; `BLOCK_M=16` and `BLOCK_N=32` stay fixed. A separate result label and filename distinguish this three-way sweep from the original paired method.

To compare output-column tiles while holding `BLOCK_K=64` fixed, run:

```bash
PYTHONPATH=.. python benchmarks/matmul_compare.py --m 1 --k 256 --n 256 --block-n-sweep
```

This checks `BLOCK_N=32` and `BLOCK_N=64` against the float32 reference, then times both variants and PyTorch float16 with rotating measurement order. `BLOCK_M=16` stays fixed. Run the same command with `--m 16` and `--m 128` to compare the three row counts from the K-tile experiment.

To test whether a taller output tile helps the larger matrix, hold `BLOCK_N=32` and `BLOCK_K=32` fixed while comparing `BLOCK_M=16` and `BLOCK_M=32`:

```bash
PYTHONPATH=.. python benchmarks/matmul_compare.py --m 128 --k 1024 --n 1024 --block-m-sweep
```

The runner checks both Triton variants against a float32 result before timing. With this shape, the output grid changes from 256 programs to 128 programs. The existing 16-row tile remains the default. Only one sweep flag can be used per run.

To inspect the compiled resource use of those two tiles on the same GPU, run:

```bash
PYTHONPATH=.. python scripts/matmul_kernel_report.py --m 128 --k 1024 --n 1024
```

This prints registers per thread, spill count, shared memory per program, warp count, and grid size. It launches each kernel once to obtain its compiled metadata; it is a diagnostic, not a timing benchmark. Zero spills rules out register spilling as an explanation for a large slowdown, but these counts alone do not establish the kernel's bottleneck.

## Joint tile and thread configuration sweep

Run the GPU correctness tests first, then the focused sweep:

```bash
PYTHONPATH=.. python -m unittest discover -s tests -p 'test_triton_matmul.py' -v
PYTHONPATH=.. python benchmarks/matmul_compare.py --m 128 --k 1024 --n 1024 --config-sweep
```

In a Kaggle notebook, prefix each command with `!`. The sweep compares PyTorch and nine explicit configurations of the existing tiled kernel:

| BM | BN | BK | Warps | Purpose |
| ---: | ---: | ---: | ---: | --- |
| 16 | 32 | 32 | 4 | Original baseline |
| 32 | 32 | 32 | 4 | Best previous configuration at the target shape |
| 64 | 32 | 32 | 4 | Increase reuse of B across rows |
| 32 | 64 | 32 | 4 | Increase reuse of A across columns |
| 64 | 64 | 32 | 4 | Combine row and column reuse |
| 64 | 64 | 64 | 4 | Reduce K iterations for the larger output tile |
| 64 | 64 | 32 | 2 | Fewer threads per program |
| 64 | 64 | 32 | 8 | More threads per program |
| 64 | 128 | 32 | 4 | Further increase reuse of A |

A CUDA warp contains 32 threads. Tile sizes and thread counts interact; testing their combinations is part of the approach described in the [official Triton matmul tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html). These configurations are candidates, not guaranteed improvements on the T4.

The timing method remains CUDA graph replay with 100 operations per sample, 10 warmup replays, and 30 samples per implementation. Each repetition rotates through all ten implementations. This changes the amount of work between measurements relative to earlier three-way sweeps, so compare implementations within this run. Every Triton candidate must pass the float32 accuracy check before timing begins. Resource metadata is collected before timing and saved alongside the full samples. The JSON also records PyTorch's FP16 reduced-precision-reduction setting; its accuracy result remains visible even when it fails the recorded tolerance.

The runner prints the fastest measured candidate and the ratio of PyTorch median to that candidate's median. A ratio above 1 means the candidate was faster in that run. Repeat the sweep independently before treating its selected winner as reproducible. Then validate the candidate at larger K/N and adjacent shapes. No runtime dispatch or default tile changes are selected automatically.
