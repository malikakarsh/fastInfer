# PyTorch matmul baseline on Tesla T4

## Environment and methodology

- Hardware: Kaggle, two Tesla T4 GPUs with 14.56 GiB each; every benchmark used the default CUDA device only.
- Software: Python 3.13.15, PyTorch 2.11.0+cu128, Triton 3.6.0, CUDA 12.8.
- Data type: float16. Each run used 10 warmups and 30 timed samples.
- **Eager mode:** one Python-submitted matmul between CUDA events, followed by synchronization.
- **Graph mode:** 100 matmuls captured with fixed input tensors; each replay is timed and divided by 100. It also uses 10 pre-capture warmups and 10 warmup replays.

## Measurements

| Shape: A by B | Timing mode | Median per matmul | p95 | Raw data |
| --- | --- | ---: | ---: | --- |
| `1x256` by `256x256` | Eager | 36.27 µs | 92.16 µs | [JSON](../results/raw/matmul_baseline_1x256x256_float16_20261009T132439Z.json) |
| `1x256` by `256x256` | Graph replay | 4.87 µs | 4.94 µs | [JSON](../results/raw/matmul_baseline_1x256x256_float16_graph100_20261009T140551Z.json) |
| `128x1024` by `1024x1024` | Eager | 49.98 µs | 85.70 µs | [JSON](../results/raw/matmul_baseline_128x1024x1024_float16_20261010T163321Z.json) |
| `128x1024` by `1024x1024` | Graph replay | 34.07 µs | 34.28 µs | [JSON](../results/raw/matmul_baseline_128x1024x1024_float16_graph100_20261010T163326Z.json) |

The larger shape performs about 268 million floating-point operations under the `2 × M × K × N` convention. Its graph-replay median corresponds to approximately 7.88 TFLOP/s for this repeated-input benchmark.

## Analysis

For the one-row workload, eager timing is much higher and less stable than graph timing. The larger workload also shows a gap; its smaller difference is consistent with GPU work occupying a larger share of the measured interval. This pattern suggests fixed host submission or scheduling costs have greater impact on tiny operations.

The two larger-shape runs were collected five seconds apart on the same software and hardware. The graph samples nevertheless shift from roughly 32 µs in the first ten samples to roughly 34 µs thereafter. The cause of that drift is not established.

Eager and graph timings are different execution modes, so their numerical ratios are **not kernel speedups**. Graph replay repeatedly uses the same input tensors, which can change cache behavior. The small and large shapes also differ in all three dimensions; these runs do not isolate the effect of row count or establish whether either workload is compute-bound.

## Next comparison

Implement a simple tiled Triton matmul, verify its output against the PyTorch reference, then compare each implementation under matching shapes, dtypes, and timing modes. Keep the PyTorch baseline and raw samples for every comparison.
