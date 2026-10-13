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

Attention during capture uses Transformers SDPA with PyTorch's math backend explicitly selected, which retains FP32 attention intermediates for FP16 inputs. The previous eager FP16 run produced NaNs at layer 0 attention output before `o_proj`; overflowing attention scores are the suspected mechanism. A synthetic regression reproduces FP16 score overflow and checks finite math-SDPA results. Model weights, captured projection inputs, and the timed linear operations remain FP16. Attention is outside the timed region; this is not an attention performance measurement. The real-layer T4 run subsequently passed; decode integration accuracy is discussed below.

## One-projection decode integration

Run `python benchmarks/qwen_decode_compare.py` after installing the checkout into the pinned Kaggle interpreter. The default uses the same three prompts, 16 cached decode steps and 10 paired timing rounds per prompt. It pins the Qwen revision used in the successful real-layer check and uses math SDPA for both implementations. An explicit `DynamicCache` avoids legacy tuple-cache conversion.

`fastinfer.decode_linear.use_decode_gemv` temporarily replaces only layer 0's `up_proj.forward`, and only during decode. Dispatch requires inference mode/no gradients, eval mode, no bias, contiguous CUDA FP16 tensors, input `[1,1,1536]` and weights `[8960,1536]`. Other inputs use the original forward; the context restores it even after an exception. This experimental mutation is for serial benchmarking, not concurrent serving.

Each implementation first independently generates a fixed-length greedy sequence from a fresh prefill/cache. Every decode logit must be finite, logits must agree within `rtol=atol=0.01`, and token IDs must match exactly before timing. Both implementations receive two untimed warmup trajectories. Each timing trial gets another independent, untimed prefill; caches are never reused between trials. Timed runs execute a fixed number of steps even after EOS. Validation logits and token histories are not collected during timing. Dispatch counts verify the GEMV actually ran once per decode step.

Results include raw paired samples, median wall milliseconds per decode step, CUDA-event elapsed milliseconds per step, environment, hashes, tokens and accuracy. This is an eager full-model decode-loop measurement, including Python dispatch, cache growth and greedy argmax, with synchronization at trajectory boundaries. CUDA-event times can include host submission gaps; they are not pure kernel durations. Prefill, loading and tokenization are excluded. It is distinct from the earlier isolated CUDA-graph timings. Changing only one projection may yield no measurable full-model improvement; inspect repeated paired results before claiming a speedup. T4 integration results are recorded below; strict logits acceptance remains unmet.

When integrated logits fail the unchanged accuracy gate, run `python benchmarks/qwen_decode_compare.py --diagnose`. This skips performance timing, compares an independent baseline repeat, records per-step logit errors and probability total variation, and captures the candidate projection input/output at every step. Both PyTorch and the captured GEMV output are then checked against an FP32 reference using that same candidate input. Projection hooks exist only in diagnostic mode and are removed on exit. Matching greedy tokens alone does not pass the numerical gate. The first T4 integration run kept identical greedy tokens for all three prompts but failed raw-logit tolerance (7054, 12419 and 2500 mismatches respectively); full decode performance was therefore not measured.

For an explicitly exploratory performance experiment after observing strict-logit failures, run `python benchmarks/qwen_decode_compare.py --exploratory-timing --repetitions 30`. This repeats the baseline and per-step projection checks before timing. When strict accuracy fails, timing additionally requires identical greedy tokens, a passing baseline repeat and passing FP32 checks for both projection implementations at every step. The strict `accuracy.passed` result remains false; `timing_status` records `exploratory_strict_logits_failed`. This mode is not a model-quality acceptance criterion. `--diagnose` and `--exploratory-timing` are mutually exclusive. Diagnostic hooks are absent during timing. Reports include paired wall-time differences and the number of rounds where the custom version was faster, plus validation and timed GEMV/fallback call counts. An apparent small saving requires confirmation in an independent run before a performance claim.

Exploratory T4 result (2026-10-13): complete decode wall medians were PT/custom 33.165/33.549, 33.392/33.233 and 33.254/33.153 ms per step for the three prompts. Paired median savings (PT minus custom) were -147.40, -75.43 and -69.28 µs per step, with custom faster in 11/30, 12/30 and 12/30 rounds respectively. Ratios of separate medians and medians of paired differences are distinct statistics and can disagree in sign. No consistent full-decode speedup is established. Every candidate validation used16 GEMV calls and every timed candidate set used480, with zero fallbacks. Strict logits checks still failed; identical greedy tokens and local FP32 checks do not remove that limitation. Original raw timing samples remain on Kaggle; the repository stores a labeled console summary. Next investigate where full-decode time is spent before expanding kernel replacements.

## Original-model decode profile

Run `python benchmarks/qwen_decode_profile.py` in the configured Kaggle interpreter. This warms the original Qwen model with two short decode trajectories, creates a fresh cache through untimed prefill, and profiles four cached greedy decode steps using the same pinned model revision, FP16 weights and math SDPA as the full-decode benchmark. `--steps` and `--prompt` change the diagnostic workload; `--trace` optionally writes a detailed Chrome trace alongside the summary JSON.

Temporary annotations group attention, MLP, normalization, vocabulary projection and the selected layer-0 up-projection. The target up-projection is nested within MLP, so its time must not be added again to MLP totals. CPU totals describe host-side operations; GPU totals describe attributed device work. Neither is interchangeable with complete decode wall latency. The report records actual captured GPU activity-event count and explicitly flags unavailable GPU capture. Summed GPU activity durations may overlap and are not elapsed wall time. Profiler and annotation overhead affect execution, so these diagnostic durations must not replace the unprofiled performance measurements.

The console prints component totals and operations ranked by their own time, excluding child operations. The JSON preserves the counts, durations, environment and script hash. CPU tests verify unchanged outputs, region call counts and hook cleanup on failures. GPU profiling validation remains pending.

The first GPU profile contained both CPU annotation records and GPU annotation records with identical component names, producing duplicate rows with different time meanings. The component table now selects CPU annotation records and their attributed nested GPU work; GPU component annotations are retained separately in JSON. Summed non-component GPU durations exclude these component annotations. On the first four-step run, CPU component records attributed GPU work of 39.297 ms to MLP, 16.813 ms to attention, 8.975 ms to vocabulary projection and 5.719 ms to normalization; target up_proj was 0.464 ms nested in MLP. These are diagnostic totals across four steps. CPU records showed 6372 cudaLaunchKernel calls, which suggests launch overhead is worth investigating but does not establish the unprofiled bottleneck.

RMSNorm work begins with `kernels/rmsnorm.py`: a readable PyTorch reference that squares input values in FP32, calculates a per-token mean square and inverse square root with epsilon, scales the vector, casts it to the input dtype, then applies the learned weight. This matches the arithmetic order of Qwen2RMSNorm in Transformers4.46.3. Four CPU checks cover a hand-calculated vector, independent token scaling, zero and large low-precision inputs, and incorrect weight dimensions. This reference is retained as the baseline for the fused-kernel comparison below.

## First fused RMSNorm benchmark

Run the GPU tests first: `python -m unittest discover -s tests -p '*rmsnorm*.py' -v`, then `python benchmarks/rmsnorm_compare.py`. The fused kernel uses one program per token and four warps. Input, learned weight and output are contiguous FP16 or FP32; normalization arithmetic is FP32. The normalized values are cast back to the input dtype before learned-weight multiplication, matching Qwen's arithmetic order. The sum includes zero padding but divides by the actual hidden width. Floating-point fusion is disabled to preserve the explicit operation boundaries. The row-wise normalization structure follows the [official Triton tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html), with the RMSNorm equation rather than LayerNorm's mean subtraction.

The first paired benchmark uses FP16, epsilon 1e-6, seed 2, and seven shapes: one,16,128 tokens at width1536 plus one token at widths65,1024,4096,8192. Each implementation must pass `rtol=atol=0.002` against an FP32 reference; the Triton output must also agree with the Qwen-style PyTorch reference under that tolerance. Timing uses100 operations per graph,10 warmups,10 graph warmups and30 alternating samples. This measures isolated synthetic normalization, not learned model activations or complete model speed. The PyTorch baseline is the readable Qwen-style reference, not torch.compile. No tuning sweep is performed. The first T4 run passed all seven tests and in-run comparisons, with speedups recorded below.

First T4 result: all seven shapes passed FP32 and PyTorch comparisons. For(tokens,hidden), PT/Triton medians inµs were(1,1536)32.23/3.54,(16,1536)23.28/2.33,(128,1536)32.70/3.12,(1,65)14.42/1.89,(1,1024)17.10/1.89,(1,4096)17.47/2.38,(1,8192)17.94/3.30. Reported ratios ranged5.441–10.481×. This is one synthetic graph-replay run; repeat without tuning before moving to real model data. Multiple-launch reduction and lower intermediate memory traffic are the optimization hypothesis; this run does not isolate their individual contributions.

A separate repeat of the frozen synthetic benchmark again passed all seven shapes, with reported speedups5.399–10.878×. At width1536, PT/Triton medians were39.88/4.14µs (one token),28.82/2.87µs (16tokens) and42.11/3.87µs (128tokens). Absolute times differed from the first run, while the advantage persisted. This repeat reused seed2 and does not establish fresh-session or new-input confirmation.

### Compiled PyTorch comparison

Run `python benchmarks/rmsnorm_compare.py --include-compiled` to compare eager PyTorch, `torch.compile` of the exact same reference, and the manual Triton kernel. Compilation uses Inductor, fullgraph=True, dynamic=False and internal `triton.cudagraphs=False`; the benchmark's explicit outer CUDA graphs apply to all three implementations. Fullgraph compilation errors stop the run rather than silently using a partially compiled reference. Each implementation's initial call is executed and synchronized before warmup/capture, and its setup-plus-execution duration is recorded separately from steady-state timing. The same seed, shapes, epsilon and tolerances remain. All implementations must agree with FP32 and eager PyTorch before timing; failures are saved and stop the run. Thirty samples rotate measurement order among all three. Results preserve raw timings and both eager/Triton and compiled/Triton ratios. This is default Inductor compilation, not a search across all compiler optimization settings. Local CPU graph-capture checks passed using the eager compiler backend; GPU Inductor compilation and all first-run accuracy checks subsequently passed; results are below.

First compiled comparison: compiled PyTorch/manual Triton medians inµs were4.64/4.20(1x1536),2.86/2.63(16x1536),4.56/3.36(128x1536),1.76/1.80(1x65),1.88/1.86(1x1024),2.58/2.30(1x4096),4.32/4.51(1x8192). Ordinary PyTorch was substantially slower on every shape. All accuracy comparisons passed. Manual Triton had lower medians on five shapes and compiled PyTorch on two; small differences are not established advantages without confirmation. The remaining gap cannot be attributed to launches, layouts, registers or compiler-generated arithmetic from timing alone; generated code or profiling would be needed.

The same-seed compiled comparison repeated successfully: all accuracy checks passed. At width1536, compiled/manual medians were4.67/4.21µs (one token),2.74/2.56µs (16tokens) and4.21/3.12µs (128tokens). The128-token ratio was1.351× versus1.357× in the first run, approximately26% lower manual-kernel latency in both. Absolute times varied at some shapes. This does not establish new-input, fresh-session or model-level performance.

Run `python scripts/rmsnorm_compiled_inspect.py` to inspect the default128x1536 case. The script targets pinned PyTorch2.5.1 and uses its private `run_and_get_code` diagnostic helper; Inductor caching is temporarily disabled to ensure source is captured. It reuses the benchmark input generation order and compiler settings, checks FP32/eager agreement, saves full generated source and a JSON summary, and prints extracted Triton definitions and the main wrapper's launch calls. Candidate settings in generated decorators are not necessarily selected runtime settings. It also records our manual kernel's compiled registers, spills, shared memory and launch dimensions. This run is not a performance measurement; inspection alone cannot prove the cause of the remaining latency gap. Two CPU extraction tests pass; GPU diagnostic execution remains pending.

The generated128x1536 source contains one fused Triton kernel with two input-load loops: accumulation of squares, then reload for normalization/output. The manual source retains the input values across normalization. Generated code also shows normalized values kept in FP32 through weight multiplication, with no explicit intermediate FP16 cast, whereas the manual kernel preserves Qwen's explicit cast. Both pass the current tolerance; they are not guaranteed bitwise equivalent. Actual XBLOCK/RBLOCK/warps were absent from the first dump. The inspection script now executes its freshly generated wrapper and reports the unique runtime-selected launcher's parameters and registers/spills/shared memory; unresolved multiple candidates are explicitly labeled. This can reveal layout/resource differences but does not prove which difference causes the measured latency gap.

Selected compiler launcher at128x1536: XBLOCK1,RBLOCK2048,16warps (512threads/program),20registers/thread,zero spills,64shared bytes,128programs. Manual kernel:4warps (128threads/program),36registers/thread,zero spills,16shared bytes,128programs. Lower compiler registers/thread does not imply lower registers/program: nominal counts are10240 versus4608 respectively. With T4's1024-thread/SM limit, thread-count ceilings are2 versus8 resident programs, but both ceilings represent1024threads; these are not measured occupancies or evidence that fewer programs necessarily run slower.

Run `python benchmarks/rmsnorm_warp_check.py` for a controlled manual-kernel experiment at128x1536. It changes only num_warps among4,8,16 while retaining the same source,FP16 input,rounding,block2048,epsilon and fusion setting. Every variant must pass FP32 and PyTorch checks before paired graph timing; reports record registers,spills,shared memory and raw samples. Thirty timing rounds rotate the variants. This tests sensitivity of our kernel to warp count, not the full cause of its difference from the compiler-generated kernel. The original RMSNorm wrapper retains four warps. CLI/syntax checks pass locally; the GPU experiment results are recorded below.

Controlled warp result at128x1536:4/8/16 warps measured6.234/6.120/8.335µs, all accuracy checks passed, zero spills. Registers/thread were36/27/22, shared bytes16/32/64. Eight warps had the lowest median, approximately1.8% below four in this run;16 was approximately34% slower than four. This demonstrates warp sensitivity in the manual design, not the entire cause of the compiler gap. Absolute times differ from earlier sessions/runs and must not be directly compared as a controlled change. No default change follows from the small single-run8-warp advantage, especially without one-token testing.

## Real Qwen RMSNorm accuracy check

Run `python benchmarks/qwen_rmsnorm_check.py` to check the frozen four-warp kernel using the original model's learned normalization weights and actual decode inputs. For each of three fixed prompts, original-model prefill creates a fresh DynamicCache. Temporary hooks capture all57 normalization inputs and outputs during one cached decode step, then are removed. Each module must be captured exactly once. The script separately evaluates the readable reference and Triton kernel with that module's own epsilon/weights and checks both against FP32 plus the captured original output under rtol=atol=0.002. Full-model execution continues using original normalization throughout. The report records per-module errors, different-element counts, environment and source hashes. This is a local real-data accuracy check, with no timing or generation-quality claim. Three CPU hook tests pass; the T4 real-data check subsequently passed all57modules for each of three prompts. Largest reported differences were0,0,0.000488; full per-module results remain on Kaggle.

Run `python benchmarks/rmsnorm_warp_check.py --confirm-four-eight` to investigate the small eight-warp advantage before changing defaults. This mode compares only four and eight warps at hidden1536 with1,16,128tokens, uses fresh seeds3,4,5 and reverses shape order in the middle pass. Each case requires both FP32 and PyTorch accuracy before30 alternating paired samples. The report includes raw latency samples, paired four-minus-eight savings, median paired saving, win/tie counts and kernel resources. Positive saving favors eight. Nine cases are produced; this is same-session synthetic confirmation and does not justify defaults for untested widths. Default mode retains the original128-token four/eight/sixteen comparison. Two local paired-summary tests pass; the GPU confirmation results are recorded below.

Four/eight confirmation passed all nine cases. Paired median savings inµs (positive favors8) acrossseeds3/4/5 were-0.017/-0.002/+0.053 for1token,+0.005/-0.023/-0.034 for16tokens,and+0.032/+0.020/+0.138 for128tokens. Eight-warp win counts for128tokens were17/30,17/30,29/30. Eight was not consistently faster across token counts; four remains the general default. Small positive128-token savings can be revisited as shape-specific tuning, but do not establish broader benefits. Full raw samples remain on Kaggle; console summaries preserve the supplied rounded results. Next check the frozen four-warp kernel on actual model normalization data.

## Full decode with fused normalization

`decode_rmsnorm.py` provides scoped replacement of selected normalization modules for eval-mode, no-gradient, contiguous CUDA FP16 `[1,1,1536]` inputs with contiguous1536-element weights. It preserves each module's epsilon and existing weights, falls back for unsupported inputs, counts total/per-module calls and restores every forward on exit. `qwen_rmsnorm_decode_compare.py` compares only normalization replacement against the original eager model; the GEMV replacement is inactive. Shared `decode_trial` now accepts a replacement kind while preserving its existing GEMV default.

Run `python benchmarks/qwen_rmsnorm_decode_compare.py`. Default16cached steps,three prompts,30alternating paired rounds; both implementations use original untimed prefill and independent fresh DynamicCache instances. Original-model repeats, strict logits tolerance (`rtol=atol=0.01`) and identical greedy tokens must pass before timing. Each candidate trial asserts all57norms use the kernel once per step, with zero fallback calls (912kernel calls per16-step trajectory). Timed candidate totals are27360perprompt for30rounds, excluding validation/warmups. Raw wall/event timings, paired differences, per-module counts and hashes are saved. Local four dispatch/restoration tests plus five existing comparison tests pass; one CUDA integration test skips locally. GPU integration results are recorded below; real-data replay alone does not establish generation equivalence.

First integrated RMSNorm run: all five CUDA/CPU dispatch tests passed. All three candidate validations used912custom calls with zero fallback, and baseline repeats had zero mismatches. Greedy tokens matched, but strict logits failed:4322/28913/4781 mismatches with max errors0.024048/0.046875/0.035156. Timing was skipped. One-step isolated checks do not establish multi-step integrated numerical equivalence.

`qwen_rmsnorm_decode_compare.py --diagnose` now captures every actual candidate normalization input/output at every step, removes hooks after validation, and separately replays each restored original module on that same input. It compares original/Triton outputs to FP32 and each other under the existing0.002local tolerance, records912per-call results per prompt, and prints probability total variation. Diagnostic mode skips timing. Explicit `--exploratory-timing` performs these checks first and permits labeled timing only if every expected local call passes, baseline repeat passes, and greedy tokens match. Strict logits failure remains unchanged in the report. Default mode remains strict; hooks/copies are absent during timing. Four CPU diagnostic/gating tests and five shared-comparison regression tests pass. The first exploratory full-model timing result is recorded below.

First exploratory integrated RMSNorm timing: all912localchecks perprompt passed,baseline repeats had0mismatches,greedy tokens matched,and strict final logits still failed. Original/custom wall medians were33.475/31.617,33.909/31.879,33.954/32.049ms/step for France/Sky/Code. Reported speedups1.059/1.064/1.059×,approximately5.5–6.0% lower latency. Paired median savings1963.75/1972.62/2005.09µs,custom faster30/30,30/30,28/30rounds. Timed custom calls27360perprompt,zerofallback. Worst probability TV.003660/.004265/.001031. This is one exploratory run with ordinary eager model and math SDPA; not strict numerical acceptance, compiled whole-model performance or broad generation-quality evidence. Repeat the frozen experiment before a stronger claim.

The exploratory full-decode result repeated on the same three prompts: PT/custom medians33.379/31.509,33.474/31.787,33.400/31.638ms/step,reported ratios1.059/1.053/1.056×. Pairedmedian savings1763.90/1601.80/1715.61µs,custom faster30/30,29/30,29/30. All912local checks perprompt pass,tokensmatch,andstrict logits stillfail with identical accuracy metrics. Acrossbothruns median latencyreductionrangesapproximately5.0–6.0%. Do not run more identical timing repeats without new uncertainty; preserve originalrawsamples and broaden prompts/sequence length for quality evaluation. This remains an exploratory result against eager model/math SDPA,not strict logits acceptance or a compiledwholemodel benchmark.

## Broader normalization numerical behavior

Run `python benchmarks/qwen_rmsnorm_quality.py --max-new-tokens 64 --archive`. Sixteen fixed raw-text prompts cover the original three continuations plus arithmetic, code, science, narrative, translation, structured text and longer context. This uses the pinned base Qwen model and raw-text continuations, without a chat template. It is a numerical-behavior suite, not a scored language-task benchmark or a sampled-generation quality evaluation.

For each prompt, run original greedy generation twice and custom-normalization generation once, with fresh original prefill/caches. Stop at the configured EOS token or the cap of64newtokens, including the first token predicted by original prefill. That first token does not exercise the replacement. Record baseline-repeat stability, full token sequences/text, first divergence position and actual per-module kernel counts. If greedy histories match, compare their distributions directly. If they diverge, rerun custom normalization while feeding the baseline's chosen tokens, so the distributions are evaluated on the same histories. Tokens selected by the custom model on that shared history are recorded separately from the tokens fed to it. Arbitrary distributions from different histories are never compared as if their inputs were equal.

Reports retain the existing strict logits tolerance, probability total variation, per-position score gaps and argmax agreement, with no newly chosen acceptance threshold. Baseline-instability cases are flagged and excluded from summary comparisons. Cases where EOS is predicted during prefill are reported separately as not exercising custom decode. No timing is measured. Optional evidence ZIP includes this JSON and both available original raw normalization timing JSONs; a manifest reports missing evidence rather than inventing it. Download the ZIP from Kaggle to preserve the raw evidence beyond the session. Seven local tests cover stopping, token caps, dispatch, shared histories, divergence and archival. GPU broader evaluation completed; results follow.

Broader check result:16/16baseline repeatsstable,allcasesexercisedcustomdecode,15/16greedy sequencesidentical. Sciencecasefirstdiverged at token21. Zero casespassedstrictlogittolerance. Worst shared-historyTV0.0051970710046589375. Scores for the divergent case were compared using baseline-fed histories, not mismatched free-generation histories. Identical sequence fraction is not language-task accuracy, and token divergence alone does not determine which continuation is better. Need inspect original/custom texts and baseline top-two gap at the divergence before explaining the specific flip. Full JSON and both original timing files were packed into the emitted ZIP on Kaggle; preserve it outside the session. No further timing experiment is justified by this accuracy result alone.

At the science divergence (token21),the baseline top-two score gap was0.015625 and shared-historyTV0.0038997011724859476. The subsequent continuations differed in content,not merely formatting. A close ranking is consistent with sensitivity to small numerical changes; we have not isolated the exact layer/arithmetic contribution or scored the scientific correctness of either text. Retain the optimized normalization as opt-in experimental behavior and scope any performance claim accordingly.

## RMSNorm arithmetic diagnosis

Run `python benchmarks/qwen_rmsnorm_stages.py` with the pinned Kaggle environment.
The original model runs prefill and one cached decode for each of the three
original prompts. Screen all normalization modules and select the one with the
largest production Triton/model output difference per prompt. Compare squared
values, mean square, mean plus epsilon, inverse RMS, FP32 normalized values,
FP16 normalized values, and the weighted FP16 output on the same captured input.
Reports count exact differences; no accuracy tolerance is changed.

The diagnostic kernel adds intermediate stores and is separate from the normal
kernel. Require its output to exactly match the production kernel and the
PyTorch reconstruction to exactly match the captured model output before
interpreting the stage comparison. Even matching final outputs does not prove
every instrumented intermediate is identical to the production intermediate.
Hybrid calculations use Triton's mean or inverse RMS with the remaining PyTorch
operations to investigate how the difference reaches the final output. This is
not a timing experiment or proof of the cause of a particular generated-token
flip. Each selected input, weight, and intermediate is saved in a CPU tensor
snapshot alongside the JSON. GPU execution is pending; three local CPU tests
pass and one GPU cross-check test is skipped locally.

### Experimental mean reduction candidate

Run `python benchmarks/rmsnorm_mean_candidate_check.py` in the same Kaggle
session after the stage diagnosis. It replays the three saved CPU tensor
snapshots and five new synthetic seeds (6 through 10), without loading Qwen.
Use `--input-report PATH` to select an older complete stage JSON explicitly.

The separate `kernels/rmsnorm_mean_candidate.py` models the single-output,
contiguous FP32 mean path in PyTorch 2.5.1: 256 logical lanes, four accumulator
columns, block-half combination followed by warp shuffle offsets 1,2,4,8,16,
then multiplication by an FP32 reciprocal of1536. These are logical arithmetic
groups; the Triton launch still uses four physical warps. The candidate only
accepts one FP16 token of width1536 and preserves Qwen's FP16 rounding before
weight scaling. Production kernel and model dispatch remain unchanged.

Source basis: [Reduce.cuh](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/Reduce.cuh),
[MeanOps](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/SharedReduceOps.h),
and [mean kernel](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/ReduceMomentKernel.cu).
Following the source grouping is a hypothesis, not a promise of bitwise runtime
agreement. Exact mean/output differences and the unchanged .002 accuracy checks
are reported. Candidate outputs with and without a diagnostic mean store must
match exactly before timing. Alternating paired CUDA-graph timings use30samples
of100operations each, after warmup. This does not validate full-generation
behavior or speed. Two CPU statistics tests pass locally; two GPU arithmetic
and eligibility tests require Kaggle execution.

The first candidate run passed all five tests on T4. Its mean and final output
matched PyTorch exactly on all three selected real inputs and all five new
synthetic seeds. It eliminated the one output difference in the selected layer7
input. Candidate medians were6.707–7.183us versus3.970–4.180us for the current
kernel; paired savings were negative in every case (approximately-2.709 to
-3.034us). Exact agreement on eight inputs does not establish full-model
equivalence. The full raw JSON remains on Kaggle:
`rmsnorm_mean_candidate_20261013T075927668712Z.json`.

Run `python benchmarks/qwen_rmsnorm_decode_compare.py --kernel mean_candidate
--diagnose --steps 16` to evaluate the candidate inside all57 normalization
layers during decode for the original three prompts. Prefill and other model
operations remain original. This diagnostic checks912 local calls per prompt
against original normalization and the FP32 reference, reports exact local
agreement as well as unchanged tolerance checks, and compares full-model logits
and greedy tokens. No timing is collected in diagnostic mode. The selected
kernel and source hash are recorded; candidate reports have a
`qwen_rmsnorm_decode_mean_candidate_` prefix. The default selection remains the
current kernel. Five local normalization-decode tests and five shared comparison
tests pass; actual candidate model execution is pending on Kaggle.

Candidate full-model diagnosis subsequently matched original normalization
exactly on all2736calls across three prompts and16cachedsteps each. Final
logits and greedy tokens also matched exactly. The first strict-gated timing
run reported original/candidate medians33.835/31.712,34.097/32.190,and
34.002/31.955ms per decode step. These correspond to5.6–6.3percent lower
latency, with candidate-faster paired rounds29/30,28/30,and30/30.
Raw diagnostic and timing reports remain on Kaggle as
`qwen_rmsnorm_decode_mean_candidate_20261013T080422814381Z.json` and
`qwen_rmsnorm_decode_mean_candidate_20261013T080534845270Z.json`.
This compares against the uncompiled FP16 model with math SDPA on T4; it is
not a direct model-level comparison between the two Triton implementations.
The candidate is slower than the first Triton kernel in isolated same-run
normalization timing, while improving this original-model decode baseline.

Repeat the frozen timing command with `--kernel mean_candidate --steps 16
--repetitions 30`, then run `python benchmarks/qwen_rmsnorm_quality.py
--kernel mean_candidate --max-new-tokens 64 --archive`. The quality suite
retains the same16prompts, EOS stopping and baseline-fed comparison after
divergence. Candidate selection applies to both independent and baseline-fed
decode. Reports identify/hash the selected kernel and include exact-logit
agreement alongside the unchanged strict tolerance. Candidate ZIPs include
the latest two complete strict-passing candidate timing reports with matching
kernel source hashes; diagnostic runs and current-kernel timings are excluded.
Missing timing evidence is reported in the archive manifest. Nine local suite
tests pass. Broader candidate generation and the timing repeat completed:

- Repeat original/candidate medians were32.990/31.402,33.382/31.598,and
  33.180/31.187ms per step. Paired median savings were1880.08,2023.22,and
  1930.71us, with candidate faster29/30,30/30,and30/30rounds. All three
  validation trajectories again had exact logits and identical tokens.
- Across both timing runs, latency was4.8–6.3percent lower. Each timed prompt
  recorded27360candidate normalization calls and zero fallbacks.
- All16broader cases exercised custom decode, had stable baseline repeats,
  passed strict logits tolerance, and had exactly equal logits and identical
  greedy tokens. Worst aligned probability total variation was0.0. The science
  case matched through the64-token cap, resolving the previously observed
  divergence for this tested prompt.

Raw repeat report: `qwen_rmsnorm_decode_mean_candidate_20261013T081509571489Z.json`.
Raw quality report: `qwen_rmsnorm_quality_mean_candidate_20261013T081712201293Z.json`.
Evidence ZIP: `qwen_rmsnorm_quality_mean_candidate_20261013T081712201293Z.evidence.zip`.
The ZIP and all three original JSON reports are now archived in `results/raw/`.
ZIP integrity passed and its manifest reports no missing timing evidence. Console
summaries are labeled separately and are not the original raw samples. This is
confirmation on the same GPU session and small curated suite, not a universal
equivalence guarantee or fully compiled model comparison.
