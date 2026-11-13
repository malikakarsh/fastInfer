# Cases for investigating speculative numerical divergence

This accuracy diagnostic has user-run all-80 results recorded below. It is not
an adaptive precision policy, speed benchmark or demonstrated repair method.
The agent verified saved reports without rerunning model inference.

## Completed all-80 result: November 13, 2026

Evidence: [full replay report](../results/raw/speculative_history_cases_20261113T071415805347Z.json),
using the [original quantized benchmark](../results/raw/speculative_mt_bench_20261111T072715614993Z.json).

```bash
python benchmarks/speculative_history_cases.py \
  --report results/raw/speculative_mt_bench_20261111T072715614993Z.json \
  --all-cases --repeats 4
```

All **80/80 prompts completed**: the 34 saved failures and 46 saved passing
controls. Every original failure reproduced at its saved first-mismatch
position. Each of the four cells ran four times: **1,280 replay evaluations**.
All recorded full-vector repeat checks were exact, and repeated saved score
descriptions agree. Baseline generation matched its independent repeat and the
saved output in all cases. Source-report and diagnostic-code hashes matched
when audited.

| Winner pattern | Cases | Interpretation at the examined decision |
| --- | ---: | --- |
| Cache-history effect only | 11 | Cache replacement changes the winner; serial/block switching does not |
| Current-execution effect only | 5 | Serial/block switching changes the winner; cache replacement does not |
| Both factors or interaction | 18 | Both factors affect the winner, possibly only in combination |
| No winner effect | 46 | All four combinations choose the same token |

The first three rows account for all 34 saved failures. All 46 saved passing
controls had no winner effect at their selected decision. Cache values and
logits can still differ in these passing cases; cache disagreement alone is
not sufficient evidence of a token mismatch.

Examples (leading spaces omitted for readability):

| Case | Ordinary cache + serial | Ordinary cache + block | Speculative cache + serial | Speculative cache + block |
| --- | --- | --- | --- | --- |
| 147 | `the` | `the` | `ing` | `ing` |
| 139 | `_` | `1` | `_` | `1` |
| 151 | `measures` | `measures` | `measures` | `refers` |

Every case passed cache-restoration, lineage, rotary-position and retained-cache
checks. Ordinary-cache/serial reproduced baseline logits exactly, while
speculative-cache/block reproduced actual verification logits exactly.

### Score gaps and precision observation

At the selected actual speculative decisions, all 34 failures had top-two
score gaps of **0 or 0.125**, including **14 exact ties**. The 46 passing
controls had gaps of **at least 0.25**. A gap threshold therefore separates
these selected observations; this is not a validated detector across all
generation positions or unseen prompts. We have not shown that history features
improve prediction over a score-gap-only policy.

Although the report requests FP16 compute, **all recorded replay logit vectors
are BF16** (`original_dtype: torch.bfloat16`). This does not invalidate the
controlled comparisons, but the run must not be described as uniformly FP16.
Which operation introduced BF16, and whether it contributed to divergence,
remain unresolved. Casting already-rounded scores to FP16 would not recover
discarded distinctions.

### Scope and next evidence

This covers one selected decision per prompt, not every generated token. Cases
were selected using saved failure positions; the passing controls do not sample
all low-margin passing decisions. No precision intervention, repair latency,
predictor improvement or backend bug is established.

Next, audit actual layer/input/output dtypes, then collect score gaps across
generation to identify close passing decisions. Those provide stronger controls
for testing whether numerical history contributes useful predictive information
beyond the current score gap. The proposed history-aware repair remains a
research hypothesis, not a result of this run.

## First quantized pilot

The saved full quantized run contains 34 token-divergent prompts and 46 passing
prompts. The default plan selects three failures spanning the ranked mismatch
positions and three distinct passing prompts, preferring matching decision
positions, categories and prompt lengths. Selection does not inspect new logits.

| Group | Prompt | Category | Planned generated token (1-based) |
| --- | ---: | --- | ---: |
| Saved failure | 147 | STEM | 4 |
| Saved failure | 151 | Humanities | 46 |
| Saved failure | 146 | STEM | 128 |
| Saved passing control | 145 | STEM | 4 |
| Saved passing control | 153 | Humanities | 46 |
| Saved passing control | 142 | STEM | 128 |

These are a diagnostic sample, not an unbiased estimate of failure frequency.
A passing control means the whole sequence passed in the saved run; the new
diagnostic stops at the selected decision and does not revalidate its later
speculative continuation. New earlier mismatches are captured and reported.

Print the selection without loading models or running inference:

```bash
python benchmarks/speculative_history_cases.py \
  --report results/raw/speculative_mt_bench_20261111T072715614993Z.json \
  --plan-only
```

Run the six selected cases:

```bash
python benchmarks/speculative_history_cases.py \
  --report results/raw/speculative_mt_bench_20261111T072715614993Z.json \
  --failure-count 3 --control-count 3 --repeats 2
```

The target remains packed NF4 with the saved FP16 compute setting; the draft
stays unquantized. This reuses the benchmark loader and saved checkpoint paths.
No model downloads or automatic library upgrades are requested. Baseline
generation must match its repeat and the saved output before attribution.
Changes to library versions/backend availability can change reproduction.

## All 80 quantized prompts

After the pilot, investigate every source prompt with:

```bash
python benchmarks/speculative_history_cases.py \
  --report results/raw/speculative_mt_bench_20261111T072715614993Z.json \
  --all-cases --repeats 2
```

This includes all 34 saved failures and all 46 saved passes, each exactly once.
Use `--all-cases --plan-only` to inspect coverage without inference. The flag
cannot be combined with failure/control counts and refuses incomplete coverage
if a source case is unsupported, rather than silently skipping its selection.
Runtime reproduction failures remain visible in the output report.

Each failure is examined at its first saved mismatch; each passing prompt at
one predetermined position matched to a failure where possible. This covers
all prompts, not every token decision. There is no speed benchmark or repair.
At repeats=2, the run schedules 160 complete target-only generations, up to 80
partial speculative generations, one reference-cache rebuild per eligible
decision, and up to 640 cell replays. Serial replays contain multiple forward
calls. This can take substantially longer than the pilot. A fresh JSON records
progress and is saved after each case and on interruption.

## What is compared

At the first mismatch, or the preselected matching decision, hold the preceding
token IDs and pending input IDs fixed. Reconstruct the ordinary target cache
with original prompt prefill followed by individual continuation tokens, and
capture the speculative cache immediately before its actual verification call.
Then run this two-by-two comparison from independent restored cache copies:

| Starting cache | Current inputs individually | Current inputs together |
| --- | --- | --- |
| Built through ordinary history | `ordinary_serial` | `ordinary_block` |
| Built through speculative history | `speculative_serial` | `speculative_block` |

Each cell is repeated twice by default. Serial evaluation stops at the pending
row predicting the selected token. Block evaluation retains the actual block
length, including later draft inputs, and reads that same corresponding row.

Required controls include exact baseline-logit reproduction by ordinary/serial,
exact actual-verification-logit reproduction by speculative/block, stable
repeats, exact restored KV values, token lineage, rotary positions and retained
cache values after cropping. Any failed control prevents pattern attribution.
Prefill/length-only source mismatches are explicitly excluded from this pilot.

## How to interpret a case

- `current_execution_winner_effect_only`: changing serial/block execution changes
  a winner, while exchanging the two starting caches does not change the winner
  in either execution style.
- `cache_history_winner_effect_only`: exchanging caches changes a winner, while
  switching execution style does not change the winner for either cache.
- `both_factors_or_interaction`: the winner depends on both factors, possibly
  only in combination. Do not force this into a single-cause category.
- `no_winner_effect_observed`: all four choose the same token at that decision;
  numerical cache/logit differences may still exist.

These labels describe observed winners, not sole causes or backend bugs. No
FP32 recomputation or successful repair is established by this experiment.

The JSON records each cell's winner, scores, top-two gap, pairwise logit
differences and oracle-only per-layer KV differences. Cache-to-cache comparison
requires the reference execution, so it is explicitly separate from candidate
runtime observations (current margin, block lengths, row and cache length).
Those observations are collected, not claimed to predict failures cheaply.

## Cost, output and expansion

Per case: two full target-only generations (one observed, one repeat), one
speculative generation stopped at the chosen point or first earlier mismatch,
one sequential cache reconstruction, and eight cell replays at repeats=2.
The diagnostic copies KV tensors to CPU for checks and can be slower than the
original benchmark. Both models load once; there are no timed pairs. Only JSON
is saved by default; full KV tensors and vocabulary vectors are not exported.
Results go to a new `results/raw/speculative_history_cases_*.json` without
overwriting the source report. Partial results are saved on interruption/error.

After reviewing the pilot, widen selection to six failures plus six controls
with `--failure-count 6 --control-count 6`. For the original FP16 target, use:

```bash
python benchmarks/speculative_history_cases.py \
  --report results/raw/speculative_mt_bench_20261106T232021872151Z.json
```

That plan includes failures 115/148/139 at tokens 15/41/58 and passing controls
113/146/135 at those respective positions. Both configurations are investigated
separately; numerical history labels must not be transferred between them.

Helper checks (manual tensors and mocks only):

```bash
python -m unittest discover -s tests -p test_speculative_history_cases.py -v
```
