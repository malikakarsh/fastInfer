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

The subsequent [all-80 four-way replay](../speculative_history_cases.md#completed-all-80-result-november-13-2026)
reproduced all 34 quantized failures: 11 cache-history winner effects only,
5 current-execution effects only, and 18 mixed/interaction effects. All 46
selected passing decisions kept their winner. All replay controls passed with
four repeats per cell. Recorded logits were BF16 despite FP16 compute being
requested; this precision detail remains under investigation.

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

## Repeated NF4 reconstruction overhead in MPS projections

To investigate the slowdown, we isolated real layer-0 query, MLP up and MLP down
projections using the same controlled synthetic inputs for each implementation.
An untimed dispatch trace observed this path for both one-token and four-token
inputs in all six cases:

```text
Packed NF4 weights → reconstruct full weights → ordinary FP16 multiplication
```

The optional `kernels` distribution was absent and specialized Hub kernels were
unavailable in this diagnostic. We compared normal bitsandbytes execution with
explicit reconstruction on every call, and with the same quantized weights
reconstructed once before timing. Original FP16 checkpoint weights provided a
separate latency reference.

Expanded-run medians, in milliseconds per call:

| Projection | Token rows | Normal 4-bit | Reconstruct each call + linear | Predecoded FP16 | Original FP16 | Normal / predecoded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q_proj` | 1 | 0.5278 | 0.5256 | 0.1535 | 0.1438 | 3.44x |
| `q_proj` | 4 | 0.5618 | 0.5645 | 0.1743 | 0.1621 | 3.22x |
| `up_proj` | 1 | 2.3930 | 2.3847 | 0.5983 | 0.5974 | 4.00x |
| `up_proj` | 4 | 2.4130 | 2.4012 | 0.6078 | 0.6054 | 3.97x |
| `down_proj` | 1 | 2.3928 | 2.3805 | 0.5994 | 0.5954 | 3.99x |
| `down_proj` | 4 | 2.4202 | 2.4101 | 0.6089 | 0.6066 | 3.97x |

Normal quantized execution closely matched reconstructing on every call.
Moving reconstruction outside timing made the projections **3.22–4.00x faster**.
All six cases passed controls: normal quantized, explicitly reconstructed and
predecoded paths produced **exactly identical outputs**, with stable repeats and
no trace-induced changes. This controlled intervention establishes substantial
latency from repeated reconstruction and its associated temporary-memory work.
It does not isolate unpacking arithmetic from allocation, memory traffic or
other dispatch costs. Keeping predecoded FP16 matrices resident also gives up
the compressed weight-storage benefit; it is a diagnostic, not a production fix.

Evidence: [initial query-projection run](../../results/raw/mps_quantized_projection_20261113T051301196956Z.json)
and [expanded six-case run](../../results/raw/mps_quantized_projection_20261113T051721845333Z.json).
The [full test methodology and related research](quantized_projection_tests.md)
include commands, individual timing samples and interpretation limits.

These results strongly support reconstruction overhead as a contributor to the
model-level slowdown. They do not establish how much of the entire **2.22x**
inference slowdown it explains: these are isolated offline measurements, and
the earlier generation run's exact backend dispatch was not recorded.

## Interpretation and limits

These are separate runs on the same prompts, not identical-output workloads.
The FP16 target generated 4,620 tokens and the quantized target 4,412 on this
subset; only 5/46 prompts produced identical sequences across targets. Agreement
within each configuration does not require agreement between the two targets.
Neither run establishes exact correctness across all 80 prompts.

Smaller weight storage does not guarantee faster execution. The reconstruction
finding above is an engineering diagnosis of a known performance mechanism,
not a new quantization method or evidence that 4-bit inference is inherently
slower. See the [related papers and research significance](quantized_projection_tests.md#related-work-and-research-significance).
