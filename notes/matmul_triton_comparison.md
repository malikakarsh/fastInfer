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

The next controlled experiment changes only `BLOCK_K` from 32 to 64. At K=256 this reduces the number of K-loop passes from eight to four, but doubles the input block size per pass and may increase resource use. The three-way sweep runner checks both variants on identical inputs and rotates their measurement order.

## K-block-size experiment

The first [three-way sweep](../results/raw/matmul_compare_16x256x256_float16_graph100_bk32vs64_20261012T191809Z_shared.json) used the same float16 `16x256` by `256x256` input matrices for PyTorch, Triton `BLOCK_K=32`, and Triton `BLOCK_K=64`. The runner checked both Triton outputs against PyTorch with `rtol=0.01` and `atol=0.01` before timing. The environment was one Kaggle Tesla T4, PyTorch 2.11.0+cu128, Triton 3.6.0, CUDA 12.8, at commit `af8c05a286349d5c57610dd96ad723bd49d78c2d`.

| Implementation | Median | p95 |
| --- | ---: | ---: |
| PyTorch | 19.81 µs | 19.91 µs |
| Triton, `BLOCK_K=32` | 16.07 µs | 16.14 µs |
| Triton, `BLOCK_K=64` | 13.80 µs | 13.89 µs |

At this shape, `BLOCK_K=64` had **14.1% lower median latency** than `BLOCK_K=32`, or a 1.16× ratio. The final seven samples of all three implementations abruptly entered a faster timing regime: medians across those seven samples were about 10.57, 8.54, and 7.33 µs respectively. The earlier regime had medians about 19.82, 16.07, and 13.81 µs. The cause of this shared shift is unknown. The Triton configuration ratio stayed near 1.16× in both regimes, which supports the within-run ranking while limiting any interpretation of absolute latency.

The same sweep was then run with [M=1](../results/raw/matmul_compare_1x256x256_float16_graph100_bk32vs64_20261012T192204Z_shared.json) and [M=128](../results/raw/matmul_compare_128x256x256_float16_graph100_bk32vs64_20261012T192210Z_shared.json), keeping K=N=256 and the same GPU and software versions. Each run checked both Triton configurations against PyTorch before timing. Medians were:

| M | PyTorch | Triton `BLOCK_K=32` | Triton `BLOCK_K=64` | BK64 reduction versus BK32 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 5.97 µs | 15.39 µs | 13.07 µs | 15.1% |
| 16 | 19.81 µs | 16.07 µs | 13.80 µs | 14.1% |
| 128 | 7.89 µs | 22.99 µs | 22.34 µs | 2.8% |

At M=1, the 30 samples for each implementation stayed close to their respective medians. At M=128, the last two samples for all three implementations dropped sharply (PyTorch about 4.69 µs, BK32 about 13.74 µs, BK64 about 13.35 µs), and BK64 had one transitional sample at 21.72 µs. The median therefore describes the first 28-sample regime; the cause of the shared change is unknown. BK64 retained a small lead even in the final two samples, but the M=128 improvement is modest and should not be interpreted as a universal 2.8% gain.

For this tested K=N=256 workload, BK64 was faster at all three row counts, most clearly at M=1 and M=16. PyTorch remained substantially faster at M=1 and M=128. BK64 cuts the K loop from eight passes to four; that is a plausible explanation for its lower Triton latency, but the timings alone do not isolate loop overhead from other changes in generated code or resource use. The current kernel retains BK32 as its default while this result is used to guide the next tuning step.

## Next experiment: output-column tile width

With N=256, `BLOCK_N=32` launches eight column tiles per 16-row output band; `BLOCK_N=64` launches four. A wider tile lets each loaded A block contribute to more output columns within one program. It also enlarges the B block and output accumulator, which may raise resource use. The controlled sweep holds `BLOCK_M=16` and `BLOCK_K=64` fixed, checks both variants against PyTorch, and measures M=1, 16, and 128 with the same graph method. No BLOCK_N performance result is available yet.
