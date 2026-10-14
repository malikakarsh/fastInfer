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
