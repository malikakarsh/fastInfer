# Softmax findings

Implemented stable row-wise softmax for finite contiguous CUDA FP16/FP32 matrices, with FP32 arithmetic and widths up to8192. Performance conclusions below concern FP16 synthetic inputs on Tesla T4, torch2.5.1+cu121 and Triton3.1.0.

## Measurements

All15tests passed on Kaggle. The final selected implementation passed accuracy against native softmax and an FP64 oracle at rtol0.002/atol0.000002. Two frozen runs used seed2,30 balanced measurement rounds,100 operations per CUDA-graph sample,10 warmups and10 graph warmups. Compilation and correctness checks were excluded from timing.

| Shape | Native/Triton range | Compiled reference/Triton range | Triton wins against compiled (run1/run2) |
| --- | ---: | ---: | ---: |
| 1 × 65 | 1.253–1.258× | 1.145–1.149× | 30/30, 30/30 |
| 1 × 128 | 1.131–1.133× | 1.013–1.014× | 20/30, 19/30 |
| 16 × 128 | 1.319–1.353× | 1.020–1.020× | 23/30, 19/30 |
| 128 × 128 | 1.405–1.419× | 1.034–1.043× | 21/30, 21/30 |
| 16 × 1024 | 2.651–2.652× | 1.056–1.062× | 25/30, 26/30 |
| 128 × 1024 | 2.004–2.014× | 1.127–1.142× | 27/30, 28/30 |
| 1 × 1536 | 1.573–1.575× | 1.139–1.143× | 30/30, 30/30 |
| 128 × 1536 | 3.352–3.374× | 1.649–1.662× | 30/30, 30/30 |
| 1 × 8192 | 1.614–1.618× | 1.189–1.193× | 30/30, 30/30 |

Ranges span two run medians, not confidence intervals. Native lost all30paired rounds on every shape in both runs. Small128-column compiled comparisons remain near ties. Compiled means Inductor compiling the readable PyTorch reference, not compiling our custom Triton or the entire model.

## What changed and why

Each program handles one row, subtracts its maximum, exponentiates, sums, divides and stores probabilities. Padding uses negative infinity so padded positions contribute zero to the denominator. Shape-specific warp tuning changes the number of cooperating threads, not the equation.

Selection and new-seed confirmation supported one warp for1x65/128x128 and two for128x1536. Automatic choices apply only to those exact FP16 shapes on the tested hardware/software; other cases use four warps. Explicit settings preserve the original comparison baseline.

For width8192, one warp reported255registers/thread and76local-storage metadata units, versus79registers/thread and zero for four warps. Timing was16.079us versus3.504us. This supports avoiding the one-warp configuration for that wide row; it does not quantify how much slowdown comes from spilling alone. More threads can reduce per-thread storage pressure even while increasing the program total register allocation.

## Evidence and limits

The archive contains seven raw reports and four pre-existing console summaries. All members passed ZIP integrity/JSON parsing checks. Full reports were imported without overwriting different local data. Both final source hashes match current code; recorded medians and paired statistics were recomputed from samples. See [manifest](../results/raw/softmax_evidence_20261014T011658Z.manifest.json).

This benchmark does not establish model-level attention acceleration, causal or all-masked softmax support, exact output agreement, performance on other GPUs/dtypes, or superiority to a compiled full model. Native softmax is already a fused GPU operation, so its gap cannot be attributed simply to Python launching each mathematical step. Different scheduling and resource choices are hypotheses; causal attribution requires runtime profiling.

## Next task

The isolated softmax experiment is complete. Continue with the tractable elementwise SwiGLU stage: apply SiLU to a gate tensor and multiply by a second tensor. Teach its role and arithmetic before implementing or attempting to fuse matrix multiplications.
