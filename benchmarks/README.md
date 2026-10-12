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

The script checks the Triton result against PyTorch on the benchmark inputs before timing. It captures 100 operations per implementation by default, warms both graphs, then alternates their measurement order across 30 paired repetitions. Both receive the same float16 inputs. The JSON file records every sample, timing parameters, environment, and correctness tolerance. These graph-replay measurements use repeated fixed inputs and do not represent eager-call latency. Compare the two implementations within this paired run; the older standalone PyTorch results used a different sample order.

To compare the existing K tile of 32 with a K tile of 64, run:

```bash
PYTHONPATH=.. python benchmarks/matmul_compare.py --m 16 --k 256 --n 256 --block-k-sweep
```

This mode validates both Triton variants against PyTorch on the same inputs, captures a separate graph for each of the three implementations, and rotates their measurement order. Only `BLOCK_K` differs between the two Triton variants; `BLOCK_M=16` and `BLOCK_N=32` stay fixed. A separate result label and filename distinguish this three-way sweep from the original paired method.
