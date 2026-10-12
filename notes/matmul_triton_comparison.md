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

With N=256, `BLOCK_N=32` launches eight column tiles per 16-row output band; `BLOCK_N=64` launches four. A wider tile lets each loaded A block contribute to more output columns within one program. It also enlarges the B block and output accumulator, which may raise resource use. The controlled sweep held `BLOCK_M=16` and `BLOCK_K=64` fixed, checked both variants against PyTorch, and measured M=1, 16, and 128 with the same graph method.

| M | PyTorch median | Triton BN32 median | Triton BN64 median | BN64 versus BN32 | Raw result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 6.00 µs | 13.03 µs | 23.77 µs | 1.83× slower | [JSON](../results/raw/matmul_compare_1x256x256_float16_graph100_bn32vs64_bk64_20261012T193939Z_shared.json) |
| 16 | 19.73 µs | 13.80 µs | 25.36 µs | 1.84× slower | [JSON](../results/raw/matmul_compare_16x256x256_float16_graph100_bn32vs64_bk64_20261012T193947Z_shared.json) |
| 128 | 11.09 µs | 33.32 µs | 34.14 µs | 1.02× slower | [JSON](../results/raw/matmul_compare_128x256x256_float16_graph100_bn32vs64_bk64_20261012T193955Z_shared.json) |

These runs used the same Kaggle Tesla T4 environment at commit `19b92d1de71ffcb53d9033b6f1f6a0fd2772d05d`. Each run's correctness check passed at `rtol=0.01`, `atol=0.01` before timing. All three runs showed an abrupt shared shift to lower latency near their end: around sample 23 at M=1, sample 21 at M=16, and sample 20 at M=128. The median represents the earlier, longer regime. BN64 was slower than BN32 both before and after each shift, including the small gap at M=128. The shift's cause is unknown; absolute medians from these runs should not be compared directly with the earlier BK sweep.

Widening the output-column tile did not help any of these three shapes. Fewer output programs and more A reuse were insufficient to offset the costs of the wider program on this setup. The measurements do not isolate whether the cost came from register pressure, lower occupancy, generated instruction count, or another factor. Keep `BLOCK_N=32` for this kernel and test a different design for decode-like M=1 rather than widening N again.

## Larger K and N: accuracy reference correction

A later BK32/BK64 sweep at K=N=1024 completed M=1 and M=16 on the same T4. Screenshot summaries were PyTorch/BK32/BK64 medians of 6.99/30.18/27.53 µs at M=1 and 11.60/34.16/31.41 µs at M=16. Raw JSON for these runs remains on Kaggle and has not been copied into this repository. The M=128 run stopped at its pre-timing correctness check; it has no latency result. These larger-shape results show that the M=16 Triton advantage at K=N=256 does not carry over to K=N=1024 in these runs.

The stopped run used PyTorch float16 output as its reference. It reported 85/131,072 Triton elements outside `rtol=0.01`, `atol=0.01`. A follow-up on the same seeded input compared both implementations to a PyTorch float32 matmul: PyTorch float16 had 86 mismatches, while both Triton BK32 and BK64 had zero. Thus this failure does not demonstrate a Triton indexing error; it shows that the two float16 implementations disagree and the Triton outputs satisfy the stated tolerance against the higher-precision reference. PyTorch float16 remains a meaningful speed baseline, but it is not the appropriate sole accuracy oracle for this case.

The benchmark now checks Triton against the float32 result and records mismatch counts and maximum absolute errors for every implementation, including PyTorch float16. Its correctness metadata names the reference explicitly. New GPU runs are needed to validate this benchmark change and obtain M=128 timings at K=N=1024. Older JSON files retain their original PyTorch float16 correctness metadata.

The corrected benchmark's [M=1, K=N=1024 run](../results/raw/matmul_compare_1x1024x1024_float16_graph100_bk32vs64_20261012T195516Z_shared.json) passed the float32 check for all three implementations (zero mismatches out of 1,024 at the recorded tolerance). Medians were PyTorch 10.09 µs, BK32 43.96 µs, and BK64 39.79 µs. All three shifted to a faster regime around repetition 20; these medians describe the earlier regime. BK64 remained faster than BK32 but about 3.95× slower than PyTorch within this run.

The corrected [M=16 run](../results/raw/matmul_compare_16x1024x1024_float16_graph100_bk32vs64_20261012T195523Z_shared.json) also passed the float32 check for all three implementations (zero mismatches out of 16,384). Medians were PyTorch 11.25 µs, BK32 33.06 µs, and BK64 30.29 µs. The first sample was high for every implementation, especially the Triton variants; their later samples settled near 33.0 and 30.0 µs. BK64 was 2.69× slower than PyTorch by the run medians.

The corrected [M=128 run](../results/raw/matmul_compare_128x1024x1024_float16_graph100_bk32vs64_20261012T195532Z_shared.json) completed with stable samples. PyTorch float16 had 86 mismatches out of 131,072 against the float32 result at the recorded tolerance; both Triton variants had zero. Medians were PyTorch 20.37 µs, BK32 180.09 µs, and BK64 180.20 µs. BK64 and BK32 were effectively tied here, and each took about 8.85× as long as PyTorch. The larger absolute errors allowed by the relative tolerance can still be present among values counted as matching; zero mismatches does not mean exact equality.

| M, K=N=1024 | PyTorch float16 | Triton BK32 | Triton BK64 | BK64 versus PyTorch |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 10.09 µs | 43.96 µs | 39.79 µs | 3.95× slower |
| 16 | 11.25 µs | 33.06 µs | 30.29 µs | 2.69× slower |
| 128 | 20.37 µs | 180.09 µs | 180.20 µs | 8.85× slower |

All three medians are from separate paired runs, each comparing the implementations on identical inputs with the same graph method. The M=1 run shifted to a faster timing regime near repetition 20, so its absolute median should not be compared directly with a separate run's median. The within-run outcome is clear: this kernel was slower than PyTorch at all three tested M values for K=N=1024. This contrasts with the M=16, K=N=256 result, where Triton BK64 was faster. The larger dimensions leave the output tile fixed at `BLOCK_M=16`, `BLOCK_N=32`; the measurements do not identify the kernel's exact bottleneck.

This completes the initial matmul comparison across six shapes and three Triton tile configurations. The implementation, tests, and raw results remain available as a baseline. The K=N=1024 results show that the current kernel is not yet a competitive matmul implementation for these shapes. Further tuning is warranted before selecting it for the inference runtime.

## Next experiment: output-row tile height

The M=128,K=N=1024 workload produces 8 row tiles by 32 column tiles, or 256 GPU programs, with the current `BLOCK_M=16`, `BLOCK_N=32` output tile. Raising only `BLOCK_M` to 32 changes that to 4 by 32, or 128 programs. Each program can use a loaded B block across twice as many output rows, but must maintain twice as many output accumulators. The paired sweep holds `BLOCK_N=32` and `BLOCK_K=32` fixed, checks both variants against the same float32 reference, and records their timings alongside PyTorch float16. This tests one output-tile change; it does not assume fewer programs will be faster. The existing 16-row tile remains the default until measured evidence supports changing it.

The [paired BM16/BM32 run](../results/raw/matmul_compare_128x1024x1024_float16_graph100_bm16vs32_bn32_bk32_20261012T201016Z_shared.json) on a Kaggle Tesla T4 at commit `39008314a1a2a3d3d4a54cdeaa8cc7cf90fa262d` produced these per-matmul CUDA-graph medians:

| Implementation | Median | p95 | Mismatches versus FP32 |
| --- | ---: | ---: | ---: |
| PyTorch float16 | 21.03 µs | 21.44 µs | 86 / 131,072 |
| Triton BM16, BN32, BK32 | 185.50 µs | 188.56 µs | 0 / 131,072 |
| Triton BM32, BN32, BK32 | 147.26 µs | 149.77 µs | 0 / 131,072 |

BM32 reduced this Triton kernel's median latency by 20.6% (1.26× speedup) relative to BM16 in the same run. It remained 7.00× slower than the PyTorch float16 implementation. The 30 samples per implementation stayed in a narrow band; the result is not explained by the abrupt timing shifts seen in some earlier runs. Both Triton variants met the recorded `rtol=0.01`, `atol=0.01` check against the float32 reference. These timings establish a tile-size effect, not its precise hardware cause: fewer programs and greater within-program B reuse are plausible benefits, while register use, occupancy, generated instructions, and memory behavior have not yet been measured. Keep BM16 as the existing default until a competitive configuration is established and checked across shapes.

The [configuration sweep](../benchmarks/README.md#joint-tile-and-thread-configuration-sweep) was prepared to compare both measured baselines with larger output tiles and multiple warp counts. It records compiled resource counts together with timings to evaluate the reuse/resource tradeoff. Its results and the resulting change in direction are recorded below.

## Joint configuration sweep and compiler compatibility

The [configuration sweep](../results/raw/matmul_compare_128x1024x1024_float16_graph100_configs_20261012T202555Z_shared.json) at commit `64a134d88bd6239eab169aab8ecaf31dbc49faa0` measured M=128,K=N=1024 on a Tesla T4 with Triton 3.6.0. PyTorch's median was 22.42 µs. Every Triton variant passed the recorded FP32 tolerance; PyTorch FP16 had 86/131,072 mismatches with reduced-precision reduction enabled.

| BM / BN / BK / warps | Median | Registers/thread | Reported spills | Programs |
| --- | ---: | ---: | ---: | ---: |
| 16 / 32 / 32 / 4 | 194.90 µs | 116 | 0 | 256 |
| 32 / 32 / 32 / 4 | 153.36 µs | 168 | 0 | 128 |
| 64 / 32 / 32 / 4 | 141.57 µs | 196 | 0 | 64 |
| 32 / 64 / 32 / 4 | 112.94 µs | 204 | 0 | 64 |
| 64 / 64 / 32 / 4 | 157.13 µs | 255 | 96 | 32 |
| 64 / 64 / 64 / 4 | 839.92 µs | 168 | 484 | 32 |
| 64 / 64 / 32 / 2 | 1992.08 µs | 64 | 970 | 32 |
| 64 / 64 / 32 / 8 | 114.58 µs | 194 | 0 | 32 |
| 64 / 128 / 32 / 4 | 2129.06 µs | 64 | 1004 | 16 |

The fastest observed Triton configuration reduced median latency by 42.1% relative to the original tile in the same run, but remained 5.04× slower than PyTorch. It had no reported spills. Large spill counts accompanied the worst latency regressions. At the same 64/64/32 tile, increasing warps from four to eight eliminated reported spills and lowered the median from 157.13 to 114.58 µs; other effects of changing thread count are not isolated. The small difference between 112.94 and 114.58 µs needs an independent repeat before selecting between those candidates. Several series drifted lower toward the end, so the full samples remain important.

Inspection of the exact compiler version revealed a more fundamental limitation. The [Triton 3.6 matmul acceleration pass](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L344-L350) explicitly rejects the MMA path below compute capability 8.0 and falls back to FMA. NVIDIA lists the [T4 as compute capability 7.5](https://developer.nvidia.com/cuda/gpus). Therefore this standard `tl.dot` kernel is expected to compile to scalar arithmetic on this environment despite executing successfully on CUDA. The [upstream change](https://github.com/triton-lang/triton/pull/5066) also documents this fallback. The exact generated instructions from the Kaggle run have not yet been collected.

This changes the interpretation of the earlier sweeps: they tune the fallback implementation, and do not establish the performance of this kernel using Tensor Cores. The next step is to inspect generated PTX and rerun correctness and paired benchmarks on a supported GPU such as an A100 (8.0) or L4 (8.9). Compare PyTorch and Triton on that same GPU and software environment. Moving hardware does not itself establish a speedup over PyTorch, which also benefits from the new GPU. No further T4 tile sweep is planned on Triton 3.6.
