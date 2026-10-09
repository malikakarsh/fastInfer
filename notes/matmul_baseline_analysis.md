# PyTorch matmul baseline on Tesla T4

## Benchmark configuration

- Source: `results/raw/matmul_baseline_1x256x256_float16_20261009T132439Z.json`
- Commit: `aa6e8e5c92792024fce5efd78e97fdec76130fab`
- Hardware: Kaggle, two Tesla T4 GPUs with 14.56 GiB each; the benchmark uses the default CUDA device only.
- Software: Python 3.13.15, PyTorch 2.11.0+cu128, Triton 3.6.0, CUDA 12.8.
- Operation: PyTorch float16 matmul, `1 x 256` multiplied by `256 x 256`.
- Method: 10 warmups, then 30 individual CUDA-event measurements, synchronizing after each call.

## Results

Median latency was **36.27 µs**; p95 was **92.16 µs**. The minimum was **33.70 µs** and the maximum was **212.99 µs**. The first timed sample was the maximum despite warmup.

## Interpretation

This tiny, one-row workload is sensitive to launch and scheduling effects. The current event interval starts before Python dispatches matmul, so it may include time when the GPU waits for the host to submit work. The result must not be treated as the isolated GPU kernel duration or used for a speedup claim.

## Measurement limits

The shape requires about 131,072 floating-point operations under the usual `2 × M × K × N` count, and the 30 samples vary substantially. Host launch delay is one possible source. GPU sharing, clock changes, and PyTorch/cuBLAS behavior are also plausible. These data do not distinguish among them.

## Follow-up measurement

A separately labeled CUDA graph replay timing method is available. The next measurements will repeat this shape under graph replay and then cover a larger, prefill-like shape. Graph replay changes the execution mode, so its latency cannot be interpreted as an eager-call speedup.
