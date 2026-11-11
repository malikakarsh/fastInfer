# Normal Qwen2.5-7B on MT-Bench

This experiment compares the original, unquantized Qwen2.5-7B against the same
target with a Qwen2.5-Coder-1.5B-Instruct draft. The draft defaults to the model
downloaded earlier; `--draft Qwen/Qwen2.5-1.5B-Instruct` selects the other cached
candidate. Draft acceptance is empirical: a coder/instruction-tuned draft is
not guaranteed to agree well with a general base target. Token-to-ID mappings
are checked exactly; unused padding rows in model output layers need not have
equal sizes. EOS follows the target's generation configuration in both paths.

Run from the `fastinfer` Git repository directory:

```bash
# First, a short pilot with one prompt from each of the eight categories.
python benchmarks/speculative_mt_bench.py \
  --limit 8 --max-new-tokens 16 --pairs 3

# All 80 first-turn prompts, with a longer generation budget.
python benchmarks/speculative_mt_bench.py \
  --max-new-tokens 128 --pairs 3

# Target-only measurement, if desired.
python benchmarks/speculative_mt_bench.py \
  --mode baseline --limit 8 --max-new-tokens 16 --pairs 3
```

The device automatically selects CUDA, then MPS, then CPU based on availability.
Use `--device cpu`, `--device mps`, or `--device cuda` to select explicitly.
Default precision is BF16. Full cached models must already be present in the
Hugging Face cache; the script does not download models or datasets. Model
snapshots, versions, precision, CPU thread count, dataset/source hashes and raw
samples are recorded in timestamped `results/raw/speculative_mt_bench_*.json`.
Reports are saved after each pair and on errors/interruption. Do not run two
timing jobs at once or compare timings from different devices as though paired.

The dataset at `benchmarks/data/mt_bench/raw/question.jsonl` was verified locally
to contain 80 unique questions, ten in each category. Only the first turn is
used; category order is interleaved, which also makes small pilots diverse.
Default input formatting is `Question: {first turn}\nAnswer:` because the target
is a base model with no instruction chat template. `--prompt-format raw` uses
the original first-turn text unchanged. No prompt truncation is performed.
This is a latency workload using MT-Bench prompts, not an official MT-Bench
quality evaluation, and first-turn coverage is not a long-context benchmark.

Both paths use fresh full-attention DynamicCaches for each trial, greedy argmax,
the same EOS rules and token budget, no compilation or custom Triton kernels.
The candidate verifies a block of proposals in one target call. Rejected states
are removed from both caches; only the committed prefix is reused. Negative
crop counts support the installed Transformers 5.22.1 API and older APIs.
The initial target prefill emits the first token in both paths. The candidate
also prefills the draft, then carries one final committed token into the next
verification call. A fully accepted block yields an extra target token.

Before timing each prompt, run both paths once and independently repeat the
baseline. Exact token agreement gates timing. Every timed trial is also checked;
failed cases remain in the report and are excluded from accepted summaries.
The baseline and candidate measurement order alternates across paired rounds.
Both models remain resident for both paths during comparison runs.

`total_s` includes prefill, generation, draft work, cache operations, Python
dispatch, and final device synchronization. Loading, tokenization, text decoding,
validation comparisons and report writing are excluded. `prefill_s` includes
initial target prefill/first-token selection and, for speculation, draft prefill.
`decode_s` excludes that prefill interval and covers only the subsequent tokens;
the report names this boundary so draft prefill is not silently omitted from the
total comparison. Both total and decode speed ratios are saved. Proposed counts
include discarded proposals; accepted counts count only matching proposals.

Summaries include category-level results, prompt-level ratios, paired time
savings/wins, and aggregate speedup computed as the sum of baseline prompt
median times divided by the sum of candidate prompt median times. Failed cases
are listed explicitly; a partial successful subset cannot establish an all-80
claim. Three rounds provide an initial measurement, not a confidence interval.
Report prompt/output lengths and broaden lengths/repetitions before a stronger
performance claim. Matching greedy tokens does not imply identical logits.

CPU checks: all 11 speculative test groups pass, including real DynamicCache
rollback with history-dependent toy models, different padded vocabulary sizes,
EOS/rejection/budget edges, a tiny random Qwen2 model against uncached greedy,
category selection, and partial-report summaries. Run from the parent workspace:

```bash
python -m unittest discover -s fastinfer/tests -p 'test_speculative*.py' -v
```

The first real CPU smoke report, `speculative_mt_bench_20261106T220541392425Z.json`,
validated repeat stability for four baseline tokens; its single timed total was
1.3833 s including prefill. The eight-token one-prompt comparison,
`speculative_mt_bench_20261106T220622139568Z.json`, matched tokens but measured
0.579x total speedup and 13.3% draft acceptance in one pair. These smoke checks
are not broad benchmark results or evidence of an inference speedup.

## Diagnosing the user's MPS token divergence

The user-run BF16/MPS pilot report
`speculative_mt_bench_20261106T221224181734Z.json` has seven token-correct cases
and one failure, humanities question 151. Baseline repeats match. The first
divergence is generated token 3: baseline chooses token 1075 (` like`) while
speculation chooses token 1741 (` such`), followed by token 438 (` as`). All
12 proposed tokens were accepted by block verification. Thus the first block
can be reconstructed from the saved output; this divergence precedes rollback.
No new model run was made to inspect this saved report.

Run this accuracy-only replay yourself from the repo directory:

```bash
python benchmarks/speculative_divergence.py \
  --report results/raw/speculative_mt_bench_20261106T221224181734Z.json \
  --case-id 151 --device mps
```

It loads only the original target snapshot, inherits the report's precision,
and evaluates each path twice with fresh state. It does not run generation,
load the draft, or time performance. Each path uses the common generated
history before the first divergent prediction; later generated text cannot
confound the score comparison. It records full-vocabulary comparison metrics,
top-ten logits, top-two score gaps, the scores for both disputed tokens, selected
logit rows, cache lengths, prompt/block token IDs, repeat stability, and source
provenance in a separate timestamped JSON report.

Paths:

- Sequential cached evaluation, matching the ordinary decoder.
- The reconstructed original verification block, matching its causal logit row.
- Cached evaluation of only the common prefix as one block.
- Same verification shape with only future tokens changed, checking causality.
- Original verification backbone with just the relevant row sent to the LM head,
  testing output-projection shape sensitivity separately.
- Uncached evaluation of the complete common prefix as another numerical control.

The replay deliberately rejects later-block or rejected-proposal cases without
a saved trace. It cannot reconstruct proposals that were never recorded. A
stable reproduced ranking flip with unchanged logits under same-shape future
perturbation supports execution-shape numerical sensitivity. Changes under
future-token perturbation require investigation of masking and repeat noise.
Uncached/cached differences alone do not prove a cache bug because computation
shapes also differ. The original benchmark kept both models resident, whereas
this replay keeps only the target resident; failure to reproduce is informative
but not proof the original failure disappeared.

This diagnostic is prepared for user execution. No MPS replay result is yet
available, and no arithmetic or cache implementation change is justified solely
by the saved output strings.

## Finding the first responsible module

The user ran the six-path replay on MPS/BF16. It reproduced an exact sequential
tie (both disputed logits 28.75), while the block path has `like=28.75` and
`such=28.875`. All paths repeat exactly, changing future tokens has no effect,
and selecting one LM-head row does not remove the flip. The uncached prefix
also favors `such`, with different absolute logits. The 0.125 change is one
BF16 representable step in this magnitude range; this does not identify the
operator that first introduced the change. These observations alone establish
neither an error in the backend nor which evaluation is more accurate.

For a module-level trace, run yourself:

```bash
python benchmarks/speculative_layer_trace.py \
  --report results/raw/speculative_mt_bench_20261106T221224181734Z.json \
  --case-id 151 --device mps
```

It captures the common causal rows after embeddings, normalization, attention
projections, attention output, residual boundaries, MLP projections/activation,
layer outputs and final projection. Both common generated-token positions are
compared, so a difference introduced at an earlier row and propagated through
cached keys/values is visible. Prefill and shared-prefix cache tensors are also
compared. Hooks are inactive during prefill; final hooked logits must match
uninstrumented controls or the report is marked `hook_control_failed`.

If an independent per-token module (Linear, RMSNorm or SiLU) receives identical
causal input rows but produces different outputs, the script replays the first
such module on fixed input rows in single-token and block shapes. It checks
repeat stability and compares with CPU FP64 using the same stored weights and
inputs, plus an isolated FP32 operator control. For RMSNorm the low-precision
reference preserves its intermediate output-dtype cast before weight scaling;
the FP32 control uses the unrounded FP64 equation as its reference. FP64
conversion happens on CPU, since MPS does not support FP64. These controls do
not load a full FP32 copy of the 7B model.

Outputs are saved to a timestamped `speculative_layer_trace_151_*.json`. The
trace reports module boundaries, not every primitive operation. An attention
boundary may still require a subsequent Q/K/V and mask replay. Any selected
operation can also be a later source of shape sensitivity rather than the
origin of the first overall difference; inspect `first_differing_stage` and
prefill-cache comparisons together. The independent replay restores contiguous
inputs; original shapes/strides are saved so that limitation is explicit.
CPU FP64 is a high-precision numerical reference, not proof of correct rounding.
Neither a module mismatch nor a different greedy token automatically proves a
PyTorch bug. The next step depends on the observed operator and error size.

The original score reporter was corrected to choose its winner with `argmax`,
matching generation's first-index tie policy, rather than using `topk`'s tie
ordering. The numerical score table and vector comparisons remain applicable.
No model or operation replay was executed by the agent while preparing this
follow-up. Syntax checks passed; helper tests were added for user execution.

The user's first layer-trace run preserved control logits and found identical
prefill caches and identical inputs to `model.layers.0.self_attn.q_proj`. Its
output is the first differing stage: one value at the second common token,
absolute difference 4.76837158203125e-7. This identifies a shape-sensitive query
projection as the first observed source; it does not prove that this single
difference alone causes the final ranking flip, since later operations may add
their own differences. The independent probe then failed with `Undefined type
Double`: a combined `.to(device="cpu", dtype=torch.float64)` attempted FP64
conversion on MPS. The helper now explicitly performs `.detach().cpu().double()`
in separate steps for inputs, weights, biases and comparison values. No model
replay was run by the agent after fixing this; rerun the same user command to
finish the probe. A regression check records the required conversion order.

The completed user probe (`speculative_layer_trace_151_20261106T224728739089Z.json`)
reproduced the single query-projection difference with fixed inputs and weights.
Both shapes repeat exactly. The block result matches the FP64 reference rounded
to BF16 for all 7,168 compared values, whereas the single-token result differs
at one value by 4.76837158203125e-7. FP32 shapes also differ slightly, with maximum
difference 3.814697265625e-6. This supports shape-sensitive accumulation/rounding
as an explanation, not a demonstrated large arithmetic error. The FP64 reference
is approximate. The exact accumulation algorithm is not yet established.

To check whether the first differing value actually influences the final token:

```bash
python benchmarks/speculative_layer_trace.py \
  --report results/raw/speculative_mt_bench_20261106T221224181734Z.json \
  --case-id 151 --device mps --causal-check
```

The enhanced operator report shows the first differing coordinates, both output
values, FP64 reference, midpoint between outputs, distance from that midpoint,
and FP32 values. Causal checks replace only the differing independent-module
coordinates in sequential/block runs with values from the opposite path, keeping
each execution shape and every other coordinate unchanged. Source inputs and
outputs must exactly match recorded controls before patching. Self-to-self swaps
check that cloning and hooks preserve logits. If these controls fail, causal
results are invalid. A winner change is evidence of a causal contribution in
that direction; absence of a change means that local difference alone does not
explain the ranking flip. Later operations can contribute independently.
No agent-run model inference or operator replay was performed for this extension.

The user-run causal check (`speculative_layer_trace_151_20261106T225155642762Z.json`)
passes both self controls, and swapping the single q-projection difference in
either direction leaves ALL final logits unchanged. Thus this difference is
not sufficient to explain the final output for the tested intervention. Its
FP64 value lies 9.72795533016324e-9 below the midpoint of the two BF16 outputs;
the isolated FP32 single-row result lands exactly on that midpoint. These values
are consistent with accumulation/rounding-boundary sensitivity, but the precise
backend implementation sequence has not been established.

The trace shows identical attention `o_proj` inputs despite the earlier q-proj
difference. Its outputs independently differ at three values (maximum absolute
error 0.0001220703125). This is the next isolated candidate and has no bias.
The module selector allows probing it without mislabeling q-proj as the cause:

```bash
python benchmarks/speculative_layer_trace.py \
  --report results/raw/speculative_mt_bench_20261106T221224181734Z.json \
  --case-id 151 --device mps \
  --probe-module model.layers.0.self_attn.o_proj --causal-check
```

The first-differing-stage line will still correctly name q-proj; the selected
operator replay and guarded causal swaps will use o-proj. The selector refuses
to isolate modules whose corresponding causal input rows are already different.

The user-run `o_proj` check
(`speculative_layer_trace_151_20261106T225534098946Z.json`) establishes a causal
contribution. Three BF16 outputs differ on identical input rows; both shapes
repeat exactly. Single-token outputs match the CPU FP64 reference rounded to
BF16 for all 7,168 values; block outputs differ at three values. All three
reference values lie close to rounding midpoints (distances about 9.17e-9,
-2.91e-9 and -1.42e-9).

Both self controls preserve logits. Replacing just those three sequential
outputs with block values changes the final winner from `like` to `such`
(watched scores 28.75 and 28.875). Replacing the block values with sequential
values leaves the block winner at `such`. Thus this intervention is sufficient
to flip the sequential ranking; it does not establish these three values as a
necessary or exclusive cause, because later operations also differ by shape.
The earlier q-projection perturbation alone had no final logit effect.

The evidence identifies a causal contribution from shape-sensitive BF16
projection arithmetic near rounding boundaries. It does not establish the
backend's precise accumulation sequence or prove violation of a PyTorch
accuracy guarantee. Isolated FP32 controls use their own implementation paths
and do not expose the internal BF16 accumulator directly. A backend bug claim
would need a minimal operator reproducer and a demonstrated accuracy-contract
violation beyond the expectation of bitwise equality across execution shapes.
The agent only inspected saved JSON and updated notes for this result.

## User-run FP16 MPS pilot result

`speculative_mt_bench_20261106T231424332848Z.json` completes all eight first-turn
pilot prompts with exact baseline/speculative token agreement and stable
baseline repeats. PyTorch 2.15.0, Transformers 5.22.1, MPS, FP16, gamma=3,
three alternating paired rounds, budget16; reasoning stops at EOS after4tokens.
Both models remain resident. Sum of baseline prompt medians8.217695s versus
speculative6.683315s yields1.229584x aggregate speedup (18.67% less total latency).
Six prompts improve in all3rounds each; writing and reasoning lose all3rounds.
Total speed ratios: writing0.9388, roleplay1.3485, reasoning0.7632, math1.3100,
coding1.3688, extraction1.2751, STEM1.2138, humanities1.5629.

The user also ran the question151 single-decision replay in FP32 and twice in
FP16. Every replay path chooses `such`; FP32's sequential lead is about0.100935
and FP16's0.09375. Those controls resolve that particular BF16 ranking mismatch
without establishing universal sequence equivalence. The pilot checks complete
short outputs, but only one prompt/category; it is preliminary, not all80prompt
coverage or evidence of speedup for all lengths. Next user-run expansion should
freeze model snapshots, FP16 and gamma3, use all80firstturnprompts and128-token
budget, retain token checks and per-category slowdowns. No agent-run inference.

## Three failures in the full FP16 run

The user-run all80/128-token FP16 MPS report
`speculative_mt_bench_20261106T232021872151Z.json` has77token-correct timed cases
and3failures: math115 first differs at generated token15, STEM148 at41, and
extraction139 at58. All three original baseline repeats match. The ratio of
aggregate timings on the successful subset is1.3853x; that subset excludes the
failures and is not an accepted all80correct speedup result. Reporting this
matching-subset speedup is valid when paired with the full 77/80 (96.25%)
agreement count and explicit exclusions. It does not predict which future
prompts will match. See the [reporting policy and next analysis](speculative_decoding.md#reporting-the-existing-fp16-run).
The earlier BF16
first-block diagnosis must not be assumed to explain these later FP16 failures.

Trace the actual later rounds, using only these selected prompts:

```bash
python benchmarks/speculative_failure_trace.py \
  --report results/raw/speculative_mt_bench_20261106T232021872151Z.json \
  --case-ids 115 148 139 --device mps
```

Use `--case-ids 115` to start with only the earliest failure. Models are loaded
once from the original local target/draft snapshots. Precision, gamma, output
budget and formatting inherit the saved benchmark. This is a diagnostic, not
a speed measurement. It repeats the baseline, checks against saved baseline
tokens, and traces native `cached_decode` through an optional observer. Normal
benchmarks leave that observer disabled. The trace stops speculation at its
first actual mismatching emitted token; proposals are recorded as they occur,
including rejected proposals. It does not guess a later verification block
from output text.

Checks include independent target/draft token lineage versus cache lengths,
logit-row causal context, target rotary positions when observable, crop lengths,
and exact preservation of retained KV values before/after cropping. An invariant
failure is recorded as an error requiring investigation of implementation or
cache API behavior. Position hooks and cache copies may perturb runtime state;
baseline matches to saved output and probe repeat controls are recorded.

At the mismatch it saves a `.pt` replay artifact containing plain CPU tensors
and metadata: exact prior target-cache values/strides, prefix IDs, actual
verification input IDs, selected logit row, and baseline/actual logits. No model
weights are stored. Cache artifacts may require tens of MB per case. A later
minimal reproducer can load these locally generated artifacts with
`torch.load(..., weights_only=True)` without rerunning the draft.

Probe paths (each repeated twice):

- Same captured cache and the exact original verification block, preserving
  cache tensor strides as well as values.
- Same captured cache, consuming verification inputs sequentially through the
  disputed row. This changes within-call shape while retaining prior state.
- A fresh cache rebuilt using the exact target forward/crop schedule up to the
  failed round, followed by the same block; its values are compared to capture.
- A fresh cache built from prompt prefill plus sequential committed tokens,
  followed by the same block.
- A fresh cache built by prefilling the entire prior prefix in one call,
  followed by the same block.
- Uncached evaluation of the complete common causal history.

Read scores, structural checks and repeat stability together. A same-cache
single/block ranking difference supports within-call shape sensitivity. Fresh
history also changes earlier arithmetic, so a difference from a fresh cache
alone does not establish a rollback bug. A structural invariant error, unstable
baseline, or failure of the same-cache block repeat requires separate diagnosis;
the script does not automatically label any outcome a PyTorch bug. Reports
are separate timestamped `speculative_failure_trace_*.json` files. No original
benchmark report is overwritten. The agent did not run model inference while
preparing this; only syntax, whitespace and manual-tensor helper checks apply.

## Case 148: layer trace from the identical captured cache

The user ran `speculative_failure_trace_20261107T175433745245Z.json` for the three
failures. All lineage, position, crop and retained-value checks passed; probe
repeats were exact, and rebuilding the original target call/crop schedule
reproduced the captured cache values. Case115 depends on earlier numerical
history: same captured cache yields the speculative winner even with serial
inputs. Case148 changes winner between serial/block evaluation from the exact
same cache. Case139 varies across history/evaluation controls. These controls
do not establish a backend bug or prove implementation correctness generally.

Case148 is the clean current-forward-pass investigation. Its captured prefix
has76tokens; the verification input is `[13,220,19,13]` at positions76–79. Row3
supplies a bonus token, generated token41: baseline chooses `Install` while
block verification ties it with `Design` and chooses `Design`. Both full-model
scores and baseline reference are stored in the locally generated replay file.

Run a target-only trace from the artifact, without redoing draft generation:

```bash
python benchmarks/speculative_cached_layer_trace.py \
  --report results/raw/speculative_failure_trace_20261107T175433745245Z.json \
  --case-id 148 --device mps --causal-check
```

This verifies the replay artifact's hash, loads plain tensors with
`weights_only=True`, restores the same KV values/strides independently for each
path, and loads the original unquantized target at the artifact's precision.
It checks the restored cache, retention of its original prefix after forwards,
repeat stability, exact agreement with captured block logits, and hook controls.
Layer comparisons align all four causal input rows; new shared K/V states are
also compared. No prefill, draft model, prior-history reconstruction, generation,
or speed measurement occurs in this script.

With `--causal-check`, it tests up to four independent equal-input/different-output
modules, in execution order. Each module gets both self controls and both
guarded cross-swaps. It stops at the first demonstrated winner-changing
intervention, then probes that operation against CPU FP64 and isolated FP32.
Without a successful causal swap it probes the first eligible operation and
keeps the lack of a causal result explicit. Use `--max-candidates N` to adjust
the cap or `--probe-module NAME` to select one eligible operation. Screening
truncation is recorded; a negative bounded screen cannot establish that no
operation contributes. Failed replay/hook controls prevent operation attribution.

The earlier intervention helper now accepts an optional cache factory for these
same-state tests; its original prompt-prefill behavior is preserved by default.
Results are separate `speculative_cached_layer_trace_148_*.json` reports. Existing
artifacts and reports are unchanged. Model/operator execution remains with the
user. The new validation/selection tests operate only on manually supplied CPU
tensors; they do not evaluate a model or GPU operator.

The user's completed case148 result
(`speculative_cached_layer_trace_148_20261107T191349588466Z.json`) passes exact
replay, repeat and hook controls and reproduces sequential `Install` versus
block `Design` from the same cache. The first discrepancy is layer0q_proj:
eight FP16 values on identical input rows, maximum absolute difference
0.00048828125. The independent fixed-input Linear probe reproduces the same
eight differences stably. Among14,336outputs, the rounded FP64 reference differs
at11block values and7single-row values; neither shape matches it everywhere.

Both self controls pass. Swapping the eight sequential q-projection outputs
into the block computation restores `Install`; the reverse swap leaves the
sequential winner at `Install`. This establishes a causal contribution of
current-forward projection shape sensitivity for case148, not its necessity
or exclusivity across all numerical differences. It is not a diagnosis of the
separate history-sensitive failures115and139.

## Operator-only experiments for stronger bug evidence

Run the focused suite using case148's completed cached-layer report:

```bash
python benchmarks/mps_linear_bug_check.py \
  --layer-report results/raw/speculative_cached_layer_trace_148_20261107T191349588466Z.json \
  --device mps
```

It does not instantiate the7Bmodel or import Transformers. It memory-maps only
the layer0q-projection weight/bias, input-normalization weight, and four embedding
rows from the original checkpoint, reconstructs the normalization, and requires
the observed projection's mismatch count, maximum difference, and listed
coordinate outputs to match the saved report. A fingerprint failure stops the
suite instead of silently testing unrelated input. It then exports a roughly26MB
tensor fixture, enabling future tests without checkpoint/normalization access.

The following tests run in separate processes for inherited settings, fast math
explicitly disabled, and a Metal preference request with fast math disabled:

- Sparse small-integer dot products at the actual K/N dimensions and row counts
  1/2/4/8. At most32nonzero +/-1products and bias magnitude<=2ensure all products
  and all partial sums are exactly representable in FP16 in every reduction
  order. These provide an exact-result test without a rounding explanation.
- One-hot multiplication and zero-input/bias identity controls with original
  weight/bias values but subnormals explicitly zeroed for these hard checks,
  avoiding a denormal-policy false positive. The real-input tests retain all
  original values and record their subnormal counts.
- FP16 and FP32, bias enabled/disabled, with 3Dlinear, 2Dlinear, addmm,
  mm-then-bias, and explicit FP32linear followed by a cast. Each is evaluated
  jointly and row-by-row with repeated results.
- CPU FP64 and output-rounded references, plus a conventional FP32-accumulation
  forward-error model allowing final output rounding and a possible separate
  low-precision bias stage. This bound is explicitly conditional, not a claimed
  PyTorch guarantee. A violation requires checking the actual backend policy.
- Exact rational dot sums for up to16original/current differing cells, and
  nearest-even FP16 rounding selected by exact rational distance. This removes
  uncertainty from the reference sum; a one-round exact-dot result is still not
  automatically the operation's required accumulation scheme.

The settings flags request behavior; they do not prove which internal kernel
dispatched. A confirmed matching upstream bug must not be inferred merely from
a flag changing outputs. Each worker records versions/settings and full results.
The summary separates exact-arithmetic failures, conditional precision-model
violations, unstable outputs, and `no_correctness_violation_demonstrated`.
Passing is not proof that the backend or our full decoder is bug-free; it means
these experiments did not establish a correctness violation. The experiments
may support reporting numerical sensitivity without supporting a confirmed-bug
claim. Exact failure would be strong reproducible arithmetic evidence through
the PyTorch/MPS API, with ownership between PyTorch and underlying Apple kernels
still requiring investigation.

Reports, summary and fixture are saved in a new timestamped
`results/raw/mps_linear_bug_check_*/` directory. To reuse the fixture later:

```bash
python benchmarks/mps_linear_bug_check.py --fixture PATH_TO/q_proj_fixture.pt --device mps
```

Four reference/helper tests pass without model, MPS or linear evaluation; syntax
and whitespace checks pass. No agent-run MPS/linear experiments were performed.

The user completed `mps_linear_bug_check_20261108T000221341332Z`. Input/output
fingerprint reconstruction matches the captured case. All three configurations
pass180/180exact tests each (540total); conditional FP32-model bound failures
and repeat instabilities are zero. The original FP16 linear outputs reproduce
in every configuration. Explicit fast-math-off does not change the original
linear result, and requesting Metal preference changes addmm/mm behavior but
does not remove the linear discrepancy. Different spellings/precision paths
remain shape-sensitive.

The correct finding is `no_correctness_violation_demonstrated`, not proof that
the backend is universally correct. These experiments support bounded,
repeatable numerical sensitivity and do not establish a PyTorch/MPS bug. The
3448rounded-reference differences for separate mm-then-bias reflect comparison
against a single-round mathematical reference while that spelling introduces
an intermediate FP16 rounding; they are not3448exact-test failures. The full
80-prompt benchmark still has three unresolved exact-output failures.

The standalone fixture SHA256 is
`f77def2b05e9654dfdc5603a8e9417770f4d93c934eb176035e7a86be68e1753`.
It and the suite provide a compact suspected numerical-accuracy reproducer for
an upstream report, with the absence of demonstrated arithmetic violations
stated plainly. No additional model/operator tests were run by the agent.

The result is concrete evidence for a suspected MPS numerical-accuracy report:
identical inputs/weights/cache, different execution shapes, stable local output
differences, and a controlled intervention that changes the final ranking.
It does not by itself prove a violation of the backend's accuracy contract or
identify the precise accumulation algorithm. An upstream report should preserve
that distinction and use an isolated operator reproducer if available. The
agent inspected saved JSON and updated notes only; no new inference occurred.


## Quantized target comparison (prepared; execution pending)

Use the locally cached `unsloth/Qwen2.5-7B-bnb-4bit` target and keep
`Qwen/Qwen2.5-Coder-1.5B-Instruct` as the unquantized FP16 draft. The target
checkpoint uses NF4 with double quantization; `--dtype float16` overrides its
BF16 compute default while preserving packed 4-bit weights. Nonquantized target
modules also use the requested dtype. KV caches are not quantized by this change.

The runner requires an installed bitsandbytes build advertising the requested
backend and Accelerate. It loads the entire target directly onto that device;
it does not fall back to CPU offload or dequantize the whole model to FP16.
Before generation, it checks that Linear4bit modules and quantization state are
present and that their compute dtype matches. These checks do not establish
kernel compatibility or performance; user execution is still required. Current
bitsandbytes upstream lists Apple Silicon 4-bit support; individual installed
builds may differ. See [backend support](https://github.com/bitsandbytes-foundation/bitsandbytes).

Install the two additional dependencies in the environment you intend to test.
Preserve the existing Torch and Transformers versions for comparisons:

```bash
python -m pip install accelerate bitsandbytes
```

Start with the same eight-category, 16-token pilot:

```bash
python benchmarks/speculative_mt_bench.py \
  --target unsloth/Qwen2.5-7B-bnb-4bit \
  --target-quantization bnb-4bit \
  --draft Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --device mps --dtype float16 --mode compare \
  --limit 8 --max-new-tokens 16 --gamma 3 --pairs 3
```

Then remove `--limit 8` and set `--max-new-tokens 128` for all 80 first turns.
Both paths use the same quantized target. Exact-token checks compare ordinary
quantized decoding against speculative quantized decoding, not against the
original FP16 target. Quantization may change that target's outputs and draft
acceptance. Speedups are relative to quantized ordinary decoding; they do not
measure quantization speedup over the earlier FP16 run. The report records
checkpoint paths, quantization configuration, compute dtype, library versions
and passing-subset timings while retaining all failures. No quantized model
execution or timing has been performed by the agent.
