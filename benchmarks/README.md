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

For Triton 3.6 matmul tuning, use a GPU with compute capability 8.0 or above, such as an A100 or L4. The [3.6 compiler](https://github.com/triton-lang/triton/blob/v3.6.0/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp#L344-L350) falls back from matrix instructions to scalar FMA for compute capability below 8.0. The T4 has compute capability 7.5: a successful CUDA correctness test does not establish that its Tensor Cores are being used. Historical T4 results remain valid measurements of that setup, but should not be interpreted as measurements of a Tensor Core Triton matmul.

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

This inspects BM16/BN32, BM32/BN32, and BM32/BN64 at BK32 and four warps. It prints compute capability, registers per thread, spill count, shared memory per program, warp count, grid size, and a summary of generated PTX instructions. It launches each kernel once to obtain its compiled metadata; it is a diagnostic, not a timing benchmark. Resource counts alone do not establish the kernel's bottleneck.

`ptx.matrix_instruction_statements` counts matrix instruction statements such as `mma.sync` in the generated PTX, while `ptx.fma_instruction_statements` counts scalar FMA statements. These are static code counts, not execution counts or throughput measurements. Zero matrix statements alongside many FMA statements provides evidence of a scalar arithmetic path. Check the report on the actual GPU before selecting larger tiles. Missing PTX is reported as unavailable rather than interpreted as zero matrix instructions.

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

## Single-row transposed-weight experiment

`matmul_gemv_compare.ipynb` embeds the new kernel, GPU tests, and runner for transfer into the existing Kaggle session. It uses the isolated Python 3.12 / PyTorch 2.5.1 / Triton 3.1 environment. The script `matmul_gemv_compare.py` compares default PyTorch, BM64/BN64/BK64/W4, and FP32 row-reduction GEMV candidates (1, 4, 8 weight rows per program) on M=1,K=1536,N=8960. All implementations consume the same contiguous stored W[N,K]. No layout conversion is timed. This is a candidate-selection study, with three rounds and raw samples; confirm any selected winner separately. The rows1 candidate has since passed the frozen confirmation study; see the specialized matrix–vector result in the project README.

### Frozen GEMV confirmation

Run `python benchmarks/matmul_gemv_compare.py --confirm` in the isolated GPU environment, or copy the code cell from `matmul_gemv_confirm.ipynb` into the existing Kaggle session. This compares only default PyTorch and rows1 / 4-warps, with M=1,K=1536 and N=8960,8192,9216,1536,256. The neighboring widths 8192/9216 are synthetic robustness checks, not claimed Qwen projection dimensions. Seeds 2,3,4 are new relative to candidate selection. Shape order reverses in the middle round; paired graph measurement order alternates. Each output JSON preserves samples, accuracy, strides, fixed configuration, source hashes, and environment. Compare within each pair, since the selection study rotated five implementations.

### GEMV registers and spills

Run `python scripts/gemv_kernel_report.py` in the configured GPU environment, or copy the cell from `gemv_kernel_report.ipynb` into the existing Kaggle session. Defaults are M=1,K=1536,N=8960 and four warps for each of 1/4/8 weight rows per program. The diagnostic launches the same `_gemv` kernel with the same arguments as the wrapper and retains its compiled handle. It verifies outputs against an FP32 reference, prints registers per thread, Triton's `n_spills` field, shared bytes per program, threads and program count, and saves a timestamped JSON with environment and kernel hash. Use `--k`, `--n`, or `--output` to change the diagnostic inputs or destination.

These are static resource measurements. `spills` is not a runtime transaction count. Register count and shared-memory use alone do not establish actual occupancy or explain latency. Earlier tiled-matmul spill reports do not apply to this GEMV kernel. On the user-tested T4 at K=1536, PAD_K=2048, N=8960, rows1/4/8 reported 29/53/64 registers per thread respectively, zero spills, and 2048 shared-memory bytes per program each; all FP32 checks passed. See the [console summary](../results/raw/gemv_resources_20261013T035434061498Z_console_summary.json). With 128 threads per program and a T4 thread limit of 1024 per SM, the simple register budget permits eight programs for all three variants; actual occupancy was not measured. These counts do not support attributing the slower rows4/8 timings to spills or a lower register-limited resident-program count. The full diagnostic JSON remains on Kaggle.

## Real Qwen projection check

`qwen_linear_check.ipynb` installs pinned model dependencies and the local checkout (editable, with `--no-deps`) into the existing Kaggle Python 3.12 environment, verifies the package import, and runs `qwen_linear_check.py`. A fresh environment needs this editable install even when the repository has already been cloned. Internet and several GB of model downloads are needed on first use. The script resolves a model revision and loads Qwen2.5-1.5B in FP16 with safetensors and no remote code. The checkpoint config specifies BF16; this experiment explicitly uses FP16 to match the validated kernel.

For three fixed prompts, prefill creates the KV cache, greedy selection chooses the next token, and a single cached decode forward captures `model.layers.0.mlp.up_proj` input/output. Capture hooks exist only during decode and are removed even on failure. The unchanged model remains resident during isolated projection timing. Both PyTorch `F.linear` and rows1/4-warps GEMV use the same contiguous learned weights and captured input vector. Captured output and both implementations must satisfy rtol=atol=0.01 against an FP32 reference of the loaded FP16 tensors before timing. Failed checks are recorded and timing skipped without relaxing tolerance.

The benchmark uses 100 operations per CUDA graph, 10 warmups and 30 alternating samples per implementation per prompt. It records resolved model revision, dependency environment, source hashes, prompts/token IDs, accuracy and raw timings. This tests real layer data but does not replace a layer during generation or measure an end-to-end speedup. GPU/model validation is pending. CPU tests exercise decode-only capture, KV-cache forwarding and hook cleanup on errors.

The real-layer check validates leaf-module inputs/outputs during prefill and decode, and checks logits before token selection. A nonfinite tensor stops the run and saves its stage, shape, dtype and NaN/Inf counts in the result JSON. These diagnostic hooks are removed before isolated projection timing; the first failing boundary does not identify the exact internal operation responsible.

Attention during capture uses Transformers SDPA with PyTorch's math backend explicitly selected, which retains FP32 attention intermediates for FP16 inputs. The previous eager FP16 run produced NaNs at layer 0 attention output before `o_proj`; overflowing attention scores are the suspected mechanism. A synthetic regression reproduces FP16 score overflow and checks finite math-SDPA results. Model weights, captured projection inputs, and the timed linear operations remain FP16. Attention is outside the timed region; this is not an attention performance measurement. Full-model T4 validation of this change remains pending.

## One-projection decode integration

Run `python benchmarks/qwen_decode_compare.py` after installing the checkout into the pinned Kaggle interpreter. The default uses the same three prompts, 16 cached decode steps and 10 paired timing rounds per prompt. It pins the Qwen revision used in the successful real-layer check and uses math SDPA for both implementations. An explicit `DynamicCache` avoids legacy tuple-cache conversion.

`fastinfer.decode_linear.use_decode_gemv` temporarily replaces only layer 0's `up_proj.forward`, and only during decode. Dispatch requires inference mode/no gradients, eval mode, no bias, contiguous CUDA FP16 tensors, input `[1,1,1536]` and weights `[8960,1536]`. Other inputs use the original forward; the context restores it even after an exception. This experimental mutation is for serial benchmarking, not concurrent serving.

Each implementation first independently generates a fixed-length greedy sequence from a fresh prefill/cache. Every decode logit must be finite, logits must agree within `rtol=atol=0.01`, and token IDs must match exactly before timing. Both implementations receive two untimed warmup trajectories. Each timing trial gets another independent, untimed prefill; caches are never reused between trials. Timed runs execute a fixed number of steps even after EOS. Validation logits and token histories are not collected during timing. Dispatch counts verify the GEMV actually ran once per decode step.

Results include raw paired samples, median wall milliseconds per decode step, CUDA-event elapsed milliseconds per step, environment, hashes, tokens and accuracy. This is an eager full-model decode-loop measurement, including Python dispatch, cache growth and greedy argmax, with synchronization at trajectory boundaries. CUDA-event times can include host submission gaps; they are not pure kernel durations. Prefill, loading and tokenization are excluded. It is distinct from the earlier isolated CUDA-graph timings. Changing only one projection may yield no measurable full-model improvement; inspect repeated paired results before claiming a speedup. T4 integration validation is pending.

When integrated logits fail the unchanged accuracy gate, run `python benchmarks/qwen_decode_compare.py --diagnose`. This skips performance timing, compares an independent baseline repeat, records per-step logit errors and probability total variation, and captures the candidate projection input/output at every step. Both PyTorch and the captured GEMV output are then checked against an FP32 reference using that same candidate input. Projection hooks exist only in diagnostic mode and are removed on exit. Matching greedy tokens alone does not pass the numerical gate. The first T4 integration run kept identical greedy tokens for all three prompts but failed raw-logit tolerance (7054, 12419 and 2500 mismatches respectively); full decode performance was therefore not measured.

For an explicitly exploratory performance experiment after observing strict-logit failures, run `python benchmarks/qwen_decode_compare.py --exploratory-timing --repetitions 30`. This repeats the baseline and per-step projection checks before timing. When strict accuracy fails, timing additionally requires identical greedy tokens, a passing baseline repeat and passing FP32 checks for both projection implementations at every step. The strict `accuracy.passed` result remains false; `timing_status` records `exploratory_strict_logits_failed`. This mode is not a model-quality acceptance criterion. `--diagnose` and `--exploratory-timing` are mutually exclusive. Diagnostic hooks are absent during timing. Reports include paired wall-time differences and the number of rounds where the custom version was faster, plus validation and timed GEMV/fallback call counts. An apparent small saving requires confirmation in an independent run before a performance claim.
