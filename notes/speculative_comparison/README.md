# FP16 versus 4-bit speculative decoding on MPS

The quantized target benefited more from speculative decoding relative to its
own baseline, but its speculative inference still took **2.22x as long** as
FP16 speculative inference on the 46 prompts passing in both runs.

## Runs and setup

| Run | Target | Saved results |
| --- | --- | --- |
| FP16 | `Qwen/Qwen2.5-7B` | [Full 80-prompt JSON](../../results/raw/speculative_mt_bench_20261106T232021872151Z.json) |
| 4-bit NF4, FP16 compute | `unsloth/Qwen2.5-7B-bnb-4bit` | [Full 80-prompt JSON](../../results/raw/speculative_mt_bench_20261111T072715614993Z.json) |

Both used the unquantized FP16 `Qwen/Qwen2.5-Coder-1.5B-Instruct` draft on
Apple MPS, PyTorch 2.15.0, Transformers 5.22.1, gamma=3, up to 128 new tokens
with EOS stopping, and three timed ordinary/speculative pairs per passing
prompt. These are first-turn MT-Bench latency experiments, not official task
quality scores. The quantized run used bitsandbytes 0.50.2 and Accelerate 1.15.0.
The linked reports retain settings, checkpoint paths, validation outputs,
per-pair timings, failures and category summaries.

## Output agreement and investigated causes

Ordinary target decoding and gamma-token speculative verification sometimes
produced different token sequences in both configurations:

| Target | Exact agreement with its own ordinary baseline | Divergent prompts |
| --- | ---: | ---: |
| FP16 | 77/80 (96.25%) | 3/80 |
| 4-bit | 46/80 (57.5%) | 34/80 |

Ordinary baseline repeats were stable in both runs. Controlled investigations
of the FP16 failures identified two contributing mechanisms:

- **Current computation:** in case 148, identical starting caches and token
  inputs produced sequential `Install` versus block `Design`. Substituting eight
  sequential query-projection values into the block calculation restored
  `Install`. See the [captured-cache layer trace](../../results/raw/speculative_cached_layer_trace_148_20261107T191349588466Z.json).
- **Cached numerical history:** in case 115, sequential and block evaluation
  using the captured speculative cache both chose a space instead of ` half`.
  Rebuilding the preceding cache through reference sequential execution restored
  ` half`, even with current block evaluation. See the
  [failure replay report](../../results/raw/speculative_failure_trace_20261107T175433745245Z.json).

These findings implicate execution-dependent numerical differences; they do
not establish a PyTorch bug. The 34 quantized failures have not been individually
diagnosed, so the FP16 mechanisms must not be assumed to explain all of them.
See the [numerical-equivalence findings](../speculative_decoding.md).

## Timing on the same 46 passing prompts

To compare the configurations, restrict both saved runs to the intersection of
prompts passing exact-token checks in both. For each prompt and method, take the
median of its three total-time samples, then sum those medians:

| Target | Ordinary decoding | Speculative decoding | Ordinary / speculative |
| --- | ---: | ---: | ---: |
| FP16 | 305.0792 s | 214.4742 s | 1.4225x |
| 4-bit | 1103.9024 s | 476.0001 s | 2.3191x |

Thus, `476.0001 / 214.4742 = 2.2194`: quantized speculative execution took
approximately 2.22x as long, despite its larger speedup over its own ordinary
baseline. Failed prompts and untimed correctness checks are excluded from
these totals. Timing includes prefill, draft work and verification; model
loading and tokenization are excluded.

The FP16 report's top-level speedup is 1.3853x over all 77 passing prompts.
The 1.4225x above is recomputed on the common 46, so it differs intentionally.
All 46 quantized passing prompts also passed in the FP16 run.

## Interpretation and limits

These are separate runs on the same prompts, not identical-output workloads.
The FP16 target generated 4,620 tokens and the quantized target 4,412 on this
subset; only 5/46 prompts produced identical sequences across targets. Agreement
within each configuration does not require agreement between the two targets.
Neither run establishes exact correctness across all 80 prompts.

Smaller weight storage does not guarantee faster execution. Dequantization and
backend kernel selection are plausible contributors to the quantized slowdown,
but their costs have not been profiled in these runs. The measurements establish
a slowdown for this tested configuration, not an inherent disadvantage of 4-bit
inference. No new model execution was used to prepare this comparison.
