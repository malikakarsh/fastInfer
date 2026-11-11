# Speculative decoding: numerical equivalence findings

## Scope

Greedy speculative decoding with target `Qwen/Qwen2.5-7B` and draft
`Qwen/Qwen2.5-Coder-1.5B-Instruct`, running on Apple MPS with PyTorch 2.15.0
and Transformers 5.22.1. The full experiment used FP16, gamma=3, 80 MT-Bench
first-turn prompts, up to 128 generated tokens, and three timed pairs per prompt.
These are user-run experiments; this note records existing results.

## Observations

- **Exact token agreement failed on 3/80 prompts.** Baseline repeats were stable.
  First mismatches: math 115 at generated token 15, STEM 148 at 41, and
  extraction 139 at 58 (reported mismatch indices).
- **Case 148 isolates execution-shape sensitivity.** From the same captured KV
  cache and token history, sequential evaluation chose ` Install`; block
  evaluation chose ` Design`. Sequential scores favored ` Install` by 0.015625;
  the block scores tied, so argmax selected the lower token ID, ` Design`.
  This was the target-generated bonus token after all three proposals matched.
- **The difference begins inside the model.** Layer 0's query projection,
  evaluated on identical input rows and weights, differed at eight FP16 outputs
  between shapes `[1, 1, 3584]` and `[1, 4, 3584]`. Maximum absolute difference:
  0.00048828125. Independent operator replay reproduced it; repeats were exact.
- **An intervention demonstrated a causal contribution.** Replacing those eight
  block projection values with their sequential counterparts restored
  ` Install`. The reverse replacement did not change the sequential winner.
  This does not establish that this projection is the sole cause.
- Cache lineage, positions, retained values after cropping, and replay controls
  passed in the investigated cases. Cases 115 and 139 also show sensitivity to
  how the cache was constructed; case 148's explanation does not cover them all.

## Interpretation and limits

The same dtype and model settings do not require the same arithmetic execution
across tensor shapes. Different accumulation patterns can round differently;
close token rankings can then flip. The precise internal kernel instructions
responsible have not been established.

This is a counterexample to **exact greedy-output equivalence in our tested
implementation**, not a refutation of the mathematical speculative algorithm,
a demonstrated quality loss, or a confirmed PyTorch bug. Our implementation is
greedy; it does not implement probability-based speculative rejection sampling.

Operator-only checks across three backend configurations passed **540/540
exact-arithmetic tests**, with zero conditional FP32 error-bound failures or
repeat instabilities. The original discrepancy remained reproducible. Finding:
`no_correctness_violation_demonstrated`; passing does not certify the backend.

Increasing only the final projection's precision cannot ensure agreement when
its input hidden vectors already differ. Higher precision and batch-invariant
kernels are candidates for further evaluation, not validated remedies here.

## Reporting the existing FP16 run

Exact baseline/speculative agreement was **77/80 prompts (96.25%)**. On those
77 matching prompts, aggregate speedup was **1.3853x**, computed as the sum of
baseline per-prompt median times divided by the sum of speculative per-prompt
median times. This is a valid descriptive result for the matching subset;
it is not the average of individual speedup ratios.

Keep all three failures visible in the full-dataset correctness count, and
explicitly exclude them from the timing aggregate. This does not establish
exact correctness or a correct speedup across all 80 prompts. Exclusions were
identified after generating both outputs; no cheap advance detector or fallback
has been validated. Successful-subset performance need not represent other
prompts or workloads.

Next, summarize the saved FP16 report without new inference: per-category
agreement counts, aggregate speedups on matching prompts, draft acceptance,
and matching prompts that slowed down. Use that analysis to select a targeted
follow-up such as changing gamma; FP32 generation is not required for this
reporting step. These category and slowdown summaries remain pending.

## Evidence and follow-up

- Full run: `results/raw/speculative_mt_bench_20261106T232021872151Z.json`.
- Failure replay: `results/raw/speculative_failure_trace_20261107T175433745245Z.json`.
- Causal trace: `results/raw/speculative_cached_layer_trace_148_20261107T191349588466Z.json`.
- Operator checks: `results/raw/mps_linear_bug_check_20261108T000221341332Z/summary.json`.

See [commands and detailed methodology](speculative_mt_bench.md). Any remedy
must preserve token agreement across cache construction, sequential decode and
block verification, then be measured for its performance cost.

Related reading: [finite-precision speculative losslessness](https://arxiv.org/abs/2609.15504),
[numerical reproducibility and LayerCast](https://arxiv.org/abs/2506.09501), and
[precision-sensitive greedy decoding](https://arxiv.org/abs/2609.26621).
