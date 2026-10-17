# SwiGLU findings

The implemented stage computes `SiLU(gate) * up`; learned gate/up/down projections remain separate. Each program processes a fixed chunk of elements. FP32 arithmetic includes an explicit cast of the SiLU result to the input dtype before multiplication.

## Synthetic correctness and tuning

Eleven kernel/dispatch/order/reference tests passed before the final selected run. Five FP16 synthetic shapes passed comparisons against eager PyTorch and a staged FP64 oracle. Final selected timing measured1.46–2.10x versus eager, with30/30paired wins at every shape; compiled was slightly faster at single-row shapes and nearly tied at16/128rows.

Confirmation across seeds4/5/6 supported512elements/4warps for16x8960:89/90wins over original256/4 and72/90over compiled. Other shapes retained256/4. The128row1024/8candidate did not repeat its initial advantage. Automatic selection is limited to the exact16x8960FP16shape onTeslaT4/torch2.5.1+cu121/Triton3.1.0. Explicit settings preserve comparisons.

Generated16x8960code selected512/8 with280programs. Our original256/4used560programs. Values per program determine launched programs; warps determine cooperating threads. Matching generated geometry did not reproduce its performance. Generated code also keeps SiLU inFP32 through multiplication, so scheduling alone cannot explain differences.

## Real Qwen inputs

The original FP16 Qwen2.5-1.5B at revision8faed761d45a263340a0528343f099c05c9a4323 used math SDPA. Hooks captured gate_proj/up_proj outputs and the product consumed by down_proj. All28MLPs passed in each of four cases (112checks); largest Triton/model product difference0.00048828125. This is tolerance agreement, not exact equivalence.

Only layer0 was timed after hooks were removed, with the model resident. Compilation, model execution and capture were excluded. CUDA-graph samples replayed100operations over30balanced rounds.

| Capture | Eager µs | Compiled µs | Triton µs | Eager/Triton | Compiled-minus-Triton paired µs | Wins against compiled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill16 | 9.651 | 5.159 | 5.140 | 1.878× | +0.016 | 21/30 |
| The capital of France is | 5.776 | 2.779 | 2.827 | 2.043× | -0.042 | 3/30 |
| Explain why the sky appears blue in one sentence. | 5.867 | 2.830 | 2.975 | 1.972× | -0.103 | 9/30 |
| def add(a, b): | 5.400 | 2.598 | 2.642 | 2.044× | -0.040 | 6/30 |

Triton beat eager30/30rounds in all four cases. Prefill was essentially tied with compiled; compiled was slightly faster in all three decode cases. This supports fusion versus eager, not superiority to compilation or faster full-model generation.

## Archive verification

The [archive](../results/raw/swiglu_evidence_20261014T023525573754Z.zip) contains23manifest-listed members plus its manifest:9reports,2generated files and12source snapshots. ZIP CRC, exact Kaggle SHA256 and every member hash passed. Five full raw reports and two generated evidence files were imported without overwriting different data. Across26timed cases, positive finite30sample arrays, medians, ratios and paired statistics were independently checked. All112model checks passed; final selected and real-Qwen source hashes match current code. Older reports refer to older source versions. Generated code was not executed during archive verification. See [verification record](../results/raw/swiglu_evidence_20261014T023525573754Z.verification.json).

## What remains

SwiGLU replacement inside Qwen would require full-model numerical and timing checks before any end-to-end claim. Standalone softmax remains synthetic-only: attention score capture and model integration are untested. Tiled attention is a separate later milestone and should be compared with the existing PyTorch attention baseline.


## Bounded numerical diagnosis and lab closure

Both stage replacements passed local tolerance but failed the full-model
strict logit check on all three16step prompts; tokens matched and timing
was skipped. This prevents any full-model speedup claim. The unchanged
first decode step was screened: compiled first differed atMLPlayer0 and
matched a compiled proxy without the FP16 intermediate cast. Manual first
differed atMLPlayer1, with tiny FP32 SiLU changes causing7FP16activation
and product differences. All reconstruction/production cross-checks passed.

Twenty-five tests passed before a replay of that saved layer1pair. Of8960
positions, arithmetic variants differed from native SiLU/product as follows:

| Variant | FP32 SiLU differences | FP16 SiLU differences | FP16 product differences |
| --- | ---: | ---: | ---: |
| Fast exponential, reciprocal multiply | 7038 | 7 | 7 |
| Library exponential, reciprocal multiply | 2547 | 0 | 0 |
| Fast exponential, direct division | 7038 | 7 | 7 |
| Library exponential, direct division | 2547 | 0 | 0 |
| Library exponential, round-to-nearest division | 0 | 0 | 0 |

For this input, changing the exponential removed all7FP16output differences.
Explicit round-to-nearest division additionally removed FP32SiLU differences.
Ordinary direct division produced the same comparison counts as reciprocal
multiplication; those counts alone do not establish the generated instruction
sequence. The variants are instrumented accuracy experiments, not promoted
production kernels. Generalization, latency and full-model agreement remain
unmeasured for these variants.

The core kernel lab is complete: implementation, correctness, controlled
benchmarking/tuning, real model inputs, evidence and a bounded numerical
investigation. Full-model precision/performance remains an integration issue.
The final archive created on Kaggle is `swiglu_evidence_20261014T034251356392Z.zip`
(SHA2565669247b7a8ed9d0130066373793656c32d2fcf05fea6cec06b828d94e964151),
including14reports,2generated files,18sources,2tensor snapshots and5PTXfiles.
Its local download/verification is pending; the earlier archive remains verified.

## Separate precise candidate (prepared 2026-10-15; GPU results pending)

At the user's request, continue the saved-input finding through a bounded
integration check. `kernels/swiglu_precise_candidate.py` returns only the product,
using `libdevice.exp`, `div_rn`, the intermediate FP16 SiLU cast, and the existing
launch policy with fusion disabled. The original kernel and defaults remain intact.
No warp or block-size sweep is part of this experiment.

Fresh GPU tests cover seeds21–23, short/tail/3D/model-sized shapes, and every
finite FP16 gate value with three bounded multipliers. They compare with eager,
the staged FP64 oracle, and the instrumented arithmetic candidate. Exact output
agreement with the diagnostic is required; eager exact-match counts on real Qwen
calls and exact full-model logits are recorded separately from tolerance passes.
Testing every gate value with three multipliers does not exhaust all input pairs.

The notebook workflow checks all28MLPs on four real-input captures before a
three-prompt,16-step full-model comparison. Existing tolerances, stable-baseline,
identical-token and dispatch checks remain required. Timing starts only after
that comparison succeeds. A compiled stage failing its own model accuracy check
is excluded from full-model timing; isolated compiled performance remains a
clearly scoped comparator with its own local tolerance/difference counts. This
is still not whole-model compilation. Four local dependency-free workflow/gate
tests passed; CUDA numerical checks and measurements have not run locally.

### Kaggle accuracy result, 2026-10-16 (user-supplied console)

All31SwiGLU tests passed on TeslaT4 in10.905s, including the precise candidate's
fresh inputs and finite-FP16-value coverage. Actual Qwen products matched exactly
for all28MLPs in each of four captures (112/112). Reports:
`qwen_real_swiglu_precise_candidate_20261016T215558165826Z.json` and
`qwen_swiglu_decode_precise_candidate_accuracy_20261016T215603507309Z.json`.

With only cached-decode SwiGLU replaced, all three16-step prompts had zero strict
logit mismatches and identical tokens; the runner confirmed exact logits3/3.
Each prompt exercised448candidate calls with no fallback (1344total). The compiled
stage still had12652/20993/2951strict logit mismatches and is ineligible for
full-model timing. No speed measurements were taken in this accuracy-only run.
Exactness is established for these tested trajectories, not universally.

Kaggle archived `swiglu_evidence_20261016T215631359586Z.zip`, reported SHA256
`79599c37852e9ceb5f964fa596e681ebaef23eec9565714044f6585e23b962fb`.
It contains14JSON reports,2generated files,20sources and no snapshots/assembly.
This observation comes from supplied console output; the new raw reports/archive
have not been downloaded and independently verified locally. Next: isolated
real-input eager/compiled/candidate timing, then strictly gated full decode.

### Precise candidate performance, 2026-10-17 (user-supplied console)

Real-input report `qwen_real_swiglu_precise_candidate_20261017T233845018075Z.json`
again found112/112products exact. Isolated layer0 graph-replay times in microseconds
(eager/compiled/precise):prefill16=8.720/4.619/5.202; three decode inputs=
5.187/2.577/2.627,5.174/2.552/2.643,4.597/2.274/2.327. Precise beat eager30/30rounds
in each case; compiled was faster (precise wins1/5/1/5rounds respectively).

Full decode report `qwen_swiglu_decode_precise_candidate_20261017T234045192294Z.json`
passed local/model strict accuracy with identical tokens on all3prompts. Compiled
failed strict logits again and was excluded. Eager/precise median wallms perstep:
France35.0302/38.6066;sky35.4079/38.8275;code35.1250/38.7838. Thus the integrated
replacement was approximately9.7–10.4%slower, losing all30paired rounds perprompt.
Paired median savings:-3607.20/-3539.45/-3638.42us perstep. Each prompt timed
13440custom stage calls and0fallback. This is a clear slowdown in this run, not
an end-to-end optimization. Do not promote it as a faster model implementation.

The isolated benchmark uses CUDA graph replay; full decode includes CPU dispatch,
validation of shapes/modules, reshapes, allocation and launch submission. Source
inspection confirms those operations occur in the replacement, including automatic
configuration lookup percall. Their contribution has not been measured; do not
attribute the3.4–3.7ms regression to one cause without a control/profile. A useful
next control is the same MLP wrapper calling the original eager SiLU/multiply,
then comparing wrapper and precise launch overhead. No further warp sweep warranted.
Raw reports remain on Kaggle; these measurements are console observations only.

Prepared a three-way replacement control using `--wrapper-control`: original
Qwen, the same wrapper calling eager SiLU/multiply, and the same wrapper calling
the precise Triton candidate. No reshapes, eligibility checks, counters or kernel
arithmetic were changed. Both wrapped paths must produce exact model logits and
pass all local/dispatch checks before timing; a failure skips the comparison.
No compiled stage is involved. Existing balanced30round timing is reused. Three
local gate tests passed via their actual functions extracted without Torch;
GPU execution of this new comparison is pending. This measures net replacement
path effects and cannot assign a cost to each Python check or allocation.
