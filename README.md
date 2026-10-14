# FastInfer

FastInfer is an experimental LLM inference runtime centered on custom Triton GPU kernels and reproducible performance measurements. Python handles orchestration and benchmarking, while PyTorch provides reference implementations.

## Implementation scope

- **Triton:** GPU kernels for matrix multiplication and transformer operations, compared against PyTorch baselines.
- **Python:** model execution, KV-cache management, batching, benchmark runners, and tests.
- **C++:** selected runtime components, such as the page allocator or scheduler data structures, after their Python implementations are validated.

## Implemented

- An elementwise SwiGLU reference and fused Triton kernel: seven tests and five initial FP16 benchmark accuracy checks passed on T4. The initial kernel beat eager PyTorch but was slightly slower than the compiled reference; scheduling experiments are pending.

- A PyTorch reference for two-dimensional matrix multiplication.
- A readable stable softmax reference and a one-program-per-row Triton kernel; initial reference/GPU checks passed on the pinned Kaggle setup. Two frozen FP16 runs measured1.10–3.25× speedups over native PyTorch softmax across nine synthetic shapes. A first compiled-reference comparison passed accuracy; results are below. Three FP16 shape-specific warp choices passed confirmation across three new seeds; automatic selection is limited to the tested T4 software setup.
- A tiled float16 Triton matrix multiplication with K-block accumulation, validated against PyTorch on tested CUDA shapes.
- A specialized single-row FP16 matrix–vector kernel with FP32 reductions for contiguous stored linear weights.
- A readable PyTorch RMSNorm reference and an initial fused Triton implementation preserving Qwen's rounding order; all seven CPU/GPU checks passed on the tested setup, with a first isolated performance comparison recorded below.
- CPU correctness tests for the reference and environment reporter.
- CUDA correctness tests for the Triton kernel, including partial tiles and multiple K blocks.
- A CUDA-event benchmark for the PyTorch matmul baseline.
- A paired PyTorch-versus-Triton graph-replay benchmark runner.
- An environment reporter that records hardware and software versions.

Measured results cover the [PyTorch matmul baseline](notes/matmul_baseline_analysis.md) and an initial [paired PyTorch–Triton comparison](notes/matmul_triton_comparison.md).

## Specialized matrix–vector result

Implemented a specialized FP16 matrix–vector kernel achieving approximately 1.11–1.20× speedup over PyTorch 2.5.1 across five tested widths on a T4, using CUDA-graph timing and passing FP32-reference accuracy checks.

The frozen configuration processes one stored weight row per Triton program with four warps. Inputs and outputs are FP16; products and reductions use FP32. It computes `a @ weight.T` for `a[1, 1536]` and contiguous `weight[N, 1536]`. Confirmation covered `N = 256, 1536, 8192, 8960, 9216`, with three new random seeds (2, 3, 4) and no retuning. PyTorch's `allow_fp16_reduced_precision_reduction` remained enabled. Both implementations had zero mismatches under `rtol=0.01, atol=0.01` against an FP32 reference in all 15 paired comparisons.

| Output width N | PyTorch median range (µs) | GEMV median range (µs) | Paired speedup range |
| ---: | ---: | ---: | ---: |
| 256 | 4.10–4.13 | 3.41–3.44 | 1.201× |
| 1536 | 22.16–22.24 | 19.99–20.04 | 1.106–1.112× |
| 8192 | 107.58–107.80 | 91.17–91.42 | 1.177–1.181× |
| 8960 | 117.14–117.30 | 99.24–99.54 | 1.178–1.180× |
| 9216 | 120.10–120.51 | 102.04–102.26 | 1.177–1.179× |

Ranges span three round medians, not confidence intervals. Measurements used one Tesla T4, Python 3.12.15, PyTorch 2.5.1+cu121, and Triton 3.1.0, with 100 operations per graph replay and 30 alternating timing samples per implementation per round. These are synthetic FP16, single-row, fixed-input graph benchmarks; a second confirmation in a fresh GPU session also won all 15 paired comparisons with zero accuracy mismatches. They do not establish an eager-call, batched-matmul, or end-to-end model speedup. Widths 8192 and 9216 are neighboring synthetic checks; 256, 1536 and 8960 match selected Qwen2.5-1.5B projection dimensions. Those synthetic runs did not test actual model weights or native BF16 execution.

The [console-summary data](results/raw/gemv_confirmation_20261012T230940025682Z_console_summary.json) records the supplied rounded medians. The full raw-sample JSON remains on Kaggle and has not yet been archived here. The [fresh-session console summary](results/raw/gemv_confirmation_20261013T040026340595Z_console_summary.json) records speedups of 1.105–1.203× across the same five widths and seeds, using the frozen configuration. Original full raw samples from both sessions still need to be archived here. A subsequent real-weight projection check is recorded below; an exploratory full decode measurement is recorded below.

Reproduce with the [confirmation notebook](benchmarks/matmul_gemv_confirm.ipynb), or run `python benchmarks/matmul_gemv_compare.py --confirm` in the configured GPU environment. See the [benchmark instructions](benchmarks/README.md#frozen-gemv-confirmation).

## Real Qwen projection and decode integration

On three real decode inputs from Qwen2.5-1.5B, the first layer's `up_proj` took 116.56–116.60 µs in PyTorch versus 99.14–99.30 µs with our frozen GEMV (1.174–1.176×). Captured model outputs, PyTorch replay and GEMV all passed the FP32-reference tolerance. The model was loaded in FP16 with math SDPA attention; native BF16 execution was not tested. These are isolated CUDA-graph projection timings, not a whole-model speedup. The [console summary](results/raw/qwen_real_linear_20261013T042024726496Z_console_summary.json) preserves the supplied rounded measurements; full raw samples remain on Kaggle.

An opt-in [decode integration](decode_linear.py) now routes that one projection through GEMV for eligible single-token inputs and restores the original forward afterwards. The [full decode comparison](benchmarks/qwen_decode_compare.py) checks logits and greedy token sequences before measuring the complete cached decode loop. An exploratory T4 run completed: baseline repeats were identical, all per-step projection FP32 checks passed, and greedy tokens matched, but strict logits tolerance failed. Complete decode took roughly 33 ms per step; the custom version won only 11–12 of 30 paired rounds per prompt, with negative paired median savings. This establishes no consistent whole-model speedup. See the [rounded console summary](results/raw/qwen_decode_20261013T051943862663Z_console_summary.json). See the [methodology](benchmarks/README.md#one-projection-decode-integration).

## Fused RMSNorm first result

The initial FP16 fused RMSNorm comparison on the Kaggle T4 passed all seven tests and all in-run FP32/PyTorch accuracy checks. Across seven synthetic shapes, the kernel measured 5.441–10.481× speedup over the Qwen-style PyTorch reference using amortized CUDA-graph timing. At hidden width1536, PyTorch/Triton medians were32.23/3.54µs for one token,23.28/2.33µs for16tokens and32.70/3.12µs for128tokens. This is a first-run isolated comparison, not a model-level improvement or a comparison against compiled PyTorch. A separate same-seed repeat also won all seven shapes (5.399–10.878×), with all accuracy comparisons passing. A first compiled-PyTorch comparison also passed all checks: at width1536, compiled/manual medians were4.64/4.20,2.86/2.63 and4.56/3.36µs for1,16,128tokens. Compiled PyTorch had lower medians at widths65 and8192; the other shapes favored the manual kernel in this run, with some differences too small to interpret without repeats. See the [compiled console summary](results/raw/rmsnorm_compiled_comparison_20261013T061150230668Z_console_summary.json). A second same-seed compiled comparison reproduced the width1536 advantage; the128-token case measured1.351× versus1.357× in the first run (approximately26% lower latency). Generated-code inspection identified different warp counts and input-read structure; controlled warp tests retain four as the general default. A real Qwen data check passed all57 normalization modules on three prompts (171cases), with largest reported differences0,0,0.000488. The first full-model normalization replacement ran all912calls per16-step trajectory with zero fallbacks and matching greedy tokens, but failed strict logits tolerance; timing was skipped. Subsequent multi-step local checks passed; exploratory full-decode performance is recorded below. The [console summary](results/raw/rmsnorm_first_comparison_20261013T055701423727Z_console_summary.json) stores rounded supplied measurements; full timing samples remain on Kaggle.

## RMSNorm candidate: decode speedup with matching tested outputs

An opt-in RMSNorm candidate reduced cached decode latency by **4.8–6.3%**
across two runs on a Tesla T4, compared with the original uncompiled FP16
Qwen2.5-1.5B model using math SDPA attention. Both runs passed the existing
strict accuracy checks with **exactly matching logits and generated tokens**
on all three timed prompts. Each prompt used16decode steps and30alternating
timing pairs; prefill was excluded. The repeat measured:

| Prompt | Original model (ms/step) | Candidate (ms/step) | Speedup | Candidate faster rounds |
| --- | ---: | ---: | ---: | ---: |
| France | 32.990 | 31.402 | 1.051× | 29/30 |
| Sky | 33.382 | 31.598 | 1.056× | 30/30 |
| Code | 33.180 | 31.187 | 1.064× | 30/30 |

The candidate follows the mean-reduction addition grouping from PyTorch2.5.1
for one1536-feature token. A separate decode diagnostic found exact agreement
on all2736normalization calls across the three prompts. The broader16-prompt
suite, with up to64generated tokens and EOS stopping, also reported stable
baselines, exact logits and identical greedy token sequences in **16/16cases**.
The earlier photosynthesis divergence did not occur with this candidate.

The candidate takes approximately7µs versus4µs for the first Triton kernel
in isolated same-run normalization tests. Full-model timings here compare
the candidate against the original PyTorch model; they do not directly rank
the two Triton variants. The default kernel remains unchanged. Use
`--kernel mean_candidate` in the decode and numerical-behavior benchmarks.
These findings cover the tested batch-one FP16 setup and curated prompts;
they do not establish equivalence on arbitrary inputs, a language-task quality
score, or an advantage over a fully compiled model. The
[console summary](results/raw/rmsnorm_mean_candidate_confirmed_console_summary.json)
records supplied results. The
[original evidence ZIP](results/raw/qwen_rmsnorm_quality_mean_candidate_20261013T081712201293Z.evidence.zip)
is archived locally with the full quality JSON and both original raw timing
reports. Archive integrity passed and its manifest reports no missing files.

## Initial RMSNorm kernel: exploratory full-decode result

Replacing all 57 Qwen normalization modules during single-token decode reduced median wall latency by approximately **5.0–6.0%** across two exploratory T4 runs. This compares against the original uncompiled FP16 model, with math SDPA attention, fresh caches, 16 decode steps and 30 alternating pairs per prompt; prefill is excluded.

| Prompt | Original model (ms/step) | Fused RMSNorm (ms/step) | Reported speedup | Custom faster rounds |
| --- | ---: | ---: | ---: | ---: |
| France | 33.475 | 31.617 | 1.059× | 30/30 |
| Sky | 33.909 | 31.879 | 1.064× | 30/30 |
| Code | 33.954 | 32.049 | 1.059× | 28/30 |

All 912 local normalization checks per prompt passed, baseline repeats agreed, and greedy token sequences matched. **Strict final-logit tolerance still failed**, so these are exploratory timing results rather than a model-quality acceptance claim. Worst probability total variation ranged 0.001031–0.004265 on the tested trajectories. Every timed candidate used 27,360 custom normalization calls per prompt with zero fallbacks. A second run reproduced the advantage: original/custom medians were 33.379/31.509, 33.474/31.787 and 33.400/31.638 ms per step, with custom faster in 30/30, 29/30 and 29/30 paired rounds. The [repeat console summary](results/raw/qwen_rmsnorm_decode_20261013T070551021237Z_console_summary.json) preserves those results. This repeats the same three prompts; a broader numerical-behavior check is recorded below; full raw evidence has been packaged on Kaggle and still needs downloading. The [console summary](results/raw/qwen_rmsnorm_decode_20261013T065623423837Z_console_summary.json) preserves rounded supplied results; original raw samples remain on Kaggle.

## Initial RMSNorm kernel: broader numerical-behavior result

On 16 fixed raw-text prompts with up to 64 generated tokens and EOS stopping, baseline repeats were stable and the fused-normalization model matched the original greedy sequence on **15/16 prompts**. The science continuation first differed at token 21. Strict final-logit tolerance failed on every prompt. Worst probability total variation on shared token histories was 0.005197 (approximately 0.52 percentage points). This is agreement on a small curated suite, not a task-accuracy score or a general quality guarantee. The replacement remains experimental and opt-in; its measured performance benefit does not establish exact output preservation. The [console summary](results/raw/qwen_rmsnorm_quality_20261013T071906163552Z_console_summary.json) records the supplied results. Original JSONs and both timing runs are packaged in an evidence ZIP on Kaggle; local archival and examination of the divergent continuation remain pending.

## Softmax performance versus native PyTorch

The first FP16 CUDA-graph benchmark measured lower Triton latency on all nine
tested shapes, with reported native-PyTorch/Triton median ratios1.101–3.251×.
All accuracy checks passed. Native PyTorch means the built-in
`torch.softmax(x, dim=-1)`, distinct from our readable sequence of operations.
For16rows of1024scores, native/Triton medians were5.651/2.194µs (2.576×);
for128rows of1536scores they were11.460/3.525µs (3.251×). Paired savings
were positive on all nine shapes, with Triton faster in29/30rounds for1x65
and30/30for every other shape. These are isolated fixed-input graph timings,
not an inference-model integration result. The
[console summary](results/raw/softmax_comparison_float16_20261013T235704423966Z_console_summary.json)
stores supplied rounded results; full raw timings remain on Kaggle.

A frozen same-session repeat passed every accuracy check and favored Triton
in30/30paired rounds on all nine shapes (1.111–3.245×). Native/Triton medians
for16x1024were5.592/2.192µs and for128x1536were11.432/3.523µs.
The [repeat console summary](results/raw/softmax_comparison_float16_20261014T001552441939Z_console_summary.json)
records those supplied measurements. No kernel settings were tuned between
runs. A warp comparison remains future work.

The first default-Inductor compiled-reference comparison passed accuracy on
all nine shapes. The compiled reference was faster at128x128 (1.929vs2.093µs),
while the three small differences at1x65,1x128,and16x128are near ties.
Manual Triton had lower latency on the other five shapes, with compiled/manual
ratios1.063–1.628× and winning28–30of30paired rounds. The largest difference
was128x1536: compiled5.633µs versus manual3.459µs. This is one compiled run,
not a confirmed advantage over every compiler configuration. Native PyTorch
remained slower in30/30rounds for every shape. The
[compiled console summary](results/raw/softmax_compiled_comparison_float16_20261014T003126243894Z_console_summary.json)
records supplied results; original raw samples remain on Kaggle. The custom
default is still four warps. A controlled1/2/4/8warp comparison is prepared.

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

To compare PyTorch and Triton on the same float16 inputs and graph timing method, run:

```bash
PYTHONPATH=.. python benchmarks/matmul_compare.py --m 1 --k 256 --n 256
```

The comparison checks correctness before timing and saves samples for both implementations in one JSON file. See `benchmarks/README.md` for the method and limits.

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


## Confirmed softmax warp selection

On Tesla T4 with PyTorch2.5.1+cu121 and Triton3.1.0, default FP16 softmax
now selects one warp for shapes1x65 and128x128, and two for128x1536.
Other shapes, FP32, other GPUs or software versions use four warps.
Explicit `num_warps=4` preserves the previous baseline.

Across fresh seeds4/5/6, the candidates beat four warps in89/90,90/90,
and87/90 paired rounds respectively. All accuracy checks passed with zero
reported spills. Latency reductions from per-case medians were approximately
9–11%,11–13%,and2–2.5%. These are isolated synthetic CUDA-graph results;
absolute timing varied during the run. See the
[console summary](results/raw/softmax_warps_confirmation_20261014T010824111458Z_console_summary.json).
The selected implementation passed all 15 CPU/GPU tests on Kaggle. Two
frozen FP16 CUDA-graph runs across nine synthetic shapes passed all accuracy
checks (`rtol=0.002, atol=0.000002` against native and an FP64 oracle).
Triton beat native PyTorch in 30/30 paired rounds on every shape in both runs,
with 1.13–3.37× speedups from implementation medians.

Compared with `torch.compile(reference_softmax)`, the clearest repeated gains
were 1.14–1.15× for 1×65, 1.14× for 1×1536, 1.65–1.66× for 128×1536,
and 1.19× for 1×8192, each winning 30/30 rounds in both runs. The small
128-column cases remain near ties (1.01–1.04×, 19–23/30 wins); do not infer
universal superiority over compilation. Compilation and warmup were excluded.
These results establish isolated kernel gains, not faster model attention.

[First selected-run console summary](results/raw/softmax_compiled_comparison_float16_20261014T011348444789Z_console_summary.json)
and [frozen-repeat console summary](results/raw/softmax_compiled_comparison_float16_20261014T011645627219Z_console_summary.json)
preserve the supplied rounded results. The [evidence archive](results/raw/softmax_evidence_20261014T011658Z.zip)
is downloaded and verified: 11 JSON members, including seven full raw reports
now imported into `results/raw/`. The
[verification manifest](results/raw/softmax_evidence_20261014T011658Z.manifest.json)
records file hashes and checks. Final source hashes match local kernel and
benchmark code; raw medians, paired comparisons, accuracy and selected warp
resources were verified. See the [softmax findings](notes/softmax_results.md).


## Confirmed elementwise SwiGLU scheduling

The fused operation is `SiLU(gate) * up`, with explicit input-dtype rounding
after SiLU. Shapes describe token vectors and intermediate features:1x8960
is one token vector,16x8960 sixteen vectors. Multiple vectors can come from
one prompt or several requests; this synthetic benchmark does not measure
a request-batching system or learned projections.

Across seeds4/5/6,512elements/program and4warps at FP16shape16x8960 reduced
latency approximately5.5–6.6% versus original256/4, winning89/90paired rounds.
Versus the compiled reference, savings were approximately2.2–3.2% from medians,
with72/90wins. All accuracy checks passed and all manual variants reported
zero spills. Other shapes showed no consistent reason to change the default.
Automatic selection now uses512/4 only for this exact shape onTeslaT4,
torch2.5.1+cu121 andTriton3.1.0; otherwise256/4. Explicit overrides remain.

The [confirmation console summary](results/raw/swiglu_configuration_confirmation_float16_20261014T021553529186Z_console_summary.json)
preserves rounded results; full samples remain on Kaggle. Absolute timing
varied substantially; comparisons are paired within each case. Generated
compiled code keeps SiLU in FP32 through multiplication while our kernel
preserves the eager FP16 boundary; both passed tolerance, not bitwise checks.
No full-model performance claim follows.

The final automatic-selection run passed all11tests and all five benchmark
accuracy checks. Resource metadata confirms512/4 for16x8960 and256/4elsewhere,
with zero reported spills/shared memory. Eager speedups were1.46–2.10x,
winning30/30paired rounds on every shape. Compiled was slightly faster at the
three single-row shapes;16x8960 was only0.8%lower latency than compiled
(21/30wins) and128x8960 was essentially tied. Thus the controlled tuning
improvement over our original kernel is supported, but a substantial final
advantage over compiled PyTorch is not established. See the
[final console summary](results/raw/swiglu_comparison_float16_20261014T022339063468Z_console_summary.json).
Full SwiGLU raw reports and generated code still need to be archived locally.


## Remaining model validation

Standalone softmax is validated on synthetic finite rows only; it has no
learned weights and has not been integrated into Qwen attention. Its kernel
speedups do not establish model-level attention gains. Tiled attention,
which combines scores, normalization and value accumulation across tiles,
is a separate later milestone.

The real-input SwiGLU check is prepared: capture learned gate/up projection
outputs and their actual product across allMLPs for16-token prefill and
single-token decode, then benchmark one isolated stage against eager and
compiled PyTorch. GPU execution is pending; no model replacement or
full-model performance claim is made.
