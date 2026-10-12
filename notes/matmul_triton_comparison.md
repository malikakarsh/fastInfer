# PyTorch and Triton matmul comparison on Tesla T4

## Measurement

The first paired comparisons multiply float16 `Mx256` matrices by `256x256` matrices on one Kaggle Tesla T4, with M varied across 1, 16, and 128. All three runs used commit `117c4c58f9783729cb4333fb9b0db1871c72a1b8`, PyTorch 2.11.0+cu128, Triton 3.6.0, and CUDA 12.8. Each run checked the Triton output against PyTorch with `rtol=0.01` and `atol=0.01` before timing.

Each implementation was captured as 100 repeated operations on fixed inputs. After 10 pre-capture warmups and 10 graph replay warmups, 30 timed graph replays were recorded per implementation. Measurement order alternated between PyTorch and Triton within each run. Times below are replay durations divided by 100; they are amortized per-operation graph timings, not eager-call latency. Raw samples and environments were reconstructed from the full JSON outputs shared after the Kaggle runs.

| M | PyTorch median / p95 | Triton median / p95 | Within-run comparison | Raw data |
| ---: | ---: | ---: | --- | --- |
| 1 | 5.94 / 6.00 µs | 15.35 / 15.40 µs | Triton 2.59× slower | [JSON](../results/raw/matmul_compare_1x256x256_float16_graph100_20261012T185206Z_shared.json) |
| 16 | 19.70 / 19.73 µs | 16.04 / 16.08 µs | Triton 1.23× faster | [JSON](../results/raw/matmul_compare_16x256x256_float16_graph100_20261012T185645Z_shared.json) |
| 128 | 7.83 / 7.97 µs | 22.93 / 23.02 µs | Triton 2.93× slower | [JSON](../results/raw/matmul_compare_128x256x256_float16_graph100_20261012T185651Z_shared.json) |
| 16 repeat | 19.68 / 19.76 µs | 16.02 / 16.08 µs | Triton 1.23× faster | [JSON](../results/raw/matmul_compare_16x256x256_float16_graph100_20261012T190105Z_shared.json) |

Samples within each implementation were tightly grouped. The M=16 result is a measured win under this specific graph method and workload; it does not establish a general Triton advantage.

## Interpretation and next experiment

At M=1, the Triton kernel launches `ceil(256/32) = 8` output-tile programs. Each program is configured for a 16-row output tile although only one row is valid, and it iterates over eight 32-element K blocks. These design choices may contribute to the latency gap, but timing alone does not identify the bottleneck.

M=16 fills the output tile's row dimension and Triton is faster in that paired run. At M=128, however, Triton is slower again even though M is also a multiple of 16. Output-row padding alone therefore cannot explain the complete pattern. PyTorch's M=16 median is also unexpectedly higher than its M=128 median. Separate runs may differ in GPU state; tight samples within a run do not rule this out.

The repeated M=16 run returned nearly the same medians after the M=128 run. Its final three samples dropped to roughly 13.05 µs for PyTorch and 10.5 µs for Triton, with one transitional Triton sample at 14.30 µs. Both implementations changed at about the same point, suggesting some shared execution-state change, but the cause is unknown. The median still represents the dominant timing regime in that repeat.

## PyTorch dispatch diagnostic

A short PyTorch profiler capture used five graph-replayed matmuls per shape. At M=16 it showed five launches of a `turing_fp16_s1688gemm...` kernel and five launches of `cublasLt::splitKreduce_kernel...`. At M=128 it showed five launches of a `cutlass::Kernel2<cutlass_75_wmma_tensorop_f16...` kernel and no separate reduction kernel. This is direct evidence that PyTorch selected different GPU implementations for the two shapes. The split-K reduction is a plausible contributor to M=16 latency: partial products along K require a second kernel to combine them. It does not by itself account for every timing difference.

The profiler reported about 16.49 µs for the M=16 matmul kernel plus 4.44 µs for its reduction kernel, versus about 11.34 µs for the M=128 kernel. Profiling adds overhead and used five nodes per replay, so these values are diagnostic only; the paired benchmark medians above remain the performance measurements.

The older standalone PyTorch graph result for the same shape had a different median and was collected in a different run with a different sampling order. Use the PyTorch value from this paired run for the comparison above.

Short PyTorch-profiler captures of the Triton kernel, both under graph replay and eager execution, displayed only CPU activity on this Kaggle setup. Their `cudaGraphLaunch` and `cuLaunchKernelEx` CPU times are not Triton kernel durations. The paired CUDA-event measurements above remain the available timing evidence.

The next controlled experiment changes only `BLOCK_K` from 32 to 64. At K=256 this reduces the number of K-loop passes from eight to four, but doubles the input block size per pass and may increase resource use. The three-way sweep runner checks both variants on identical inputs and rotates their measurement order. Results are pending; no improvement is assumed.
