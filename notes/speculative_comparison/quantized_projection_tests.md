# Investigating NF4 projection latency on MPS

User-run results are recorded below. This is an isolated-operator diagnostic,
not a full model benchmark or generation test.

## Question and controlled comparison

Does using packed NF4 weights introduce reconstruction/dispatch overhead that
outweighs their smaller storage? The script reads one projection from the cached
quantized checkpoint and the corresponding original checkpoint, not full models.
All variants receive identical deterministic synthetic FP16 input rows. These
are controlled inputs, not captured real-model activations.

| Report label | Timed work |
| --- | --- |
| `normal_4bit` | Real bitsandbytes Linear4bit call, using packed NF4 weights |
| `reconstruct_each_call` | Reconstruct weights to FP16 on every call, then multiply |
| `predecoded_fp16` | Multiply with the same NF4 weights reconstructed once before timing |
| `original_fp16` | Multiply with the original model's FP16 weights |
| `reconstruction_only` | Reconstruct NF4 weights and scales without multiplication |

The first three use the same quantized model weights and bias. The original
checkpoint has different numerical weights/bias and is a latency comparator,
not an accuracy reference for quantized outputs. Explicit reconstruction outputs
FP16 directly to avoid adding a BF16-to-FP16 intermediate rounding step.
Persistent predecoded weights intentionally consume more memory; this is a
diagnostic ablation rather than a proposed memory-efficient implementation.

An untimed call instruments the installed MPS backend and records specialized
Hub GEMV calls, full-weight reconstruction and ordinary linear calls. Wrappers
are removed before timing. Unknown backend layouts are reported as unresolved.
Trace output and repeats must match exactly; quantized-versus-explicit outputs
must be finite and within rtol=0.02, atol=0.02 before timing. These tolerances
gate this experiment, not exact model-level token correctness.

## Run

From the repository root, in the same environment as the earlier experiment:

```bash
python benchmarks/mps_quantized_projection.py \
  --rows 1 4 --warmup 3 --samples 15 --ops-per-sample 3
```

Default projection: `model.layers.0.self_attn.q_proj`. One row represents a
single-token projection; four rows represent the input shape of a typical
gamma=3 verification. These do not execute attention, a KV cache or generation.
Five variants each receive 3 warmup calls and 15 timed samples of 3 calls, for
480 operation calls across the two shapes, plus setup and untimed controls.
The console reports observed paths and median milliseconds per call. Full
samples and controls go to a fresh `results/raw/mps_quantized_projection_*.json`.

If the first run succeeds, broaden the projection/shape coverage:

```bash
python benchmarks/mps_quantized_projection.py \
  --module model.layers.0.self_attn.q_proj \
  --module model.layers.0.mlp.up_proj \
  --module model.layers.0.mlp.down_proj \
  --rows 1 2 4 8
```

No downloads are requested: checkpoints are local-only and Hub access is forced
offline. Optional Metal kernels must already be cached and loadable. This may
differ from the earlier benchmark's environment; the script records backend
source hash, versions, offline setting and observed dispatch. Do not interpret
a new fallback trace as proof that the original run used the same path.

Timing uses synchronized MPS wall time and rotates variant order. It includes
launch, temporary-allocation and synchronization overhead, excludes loading and
one-time reconstruction, and leaves all comparison weights resident. Caching,
thermal state and allocator reuse affect results. Raw samples are retained.

## Interpretation

- Normal execution observing reconstruction plus linear establishes that path
  for the diagnostic call. A large gap versus predecoded multiplication supports
  reconstruction/dispatch overhead as a contributor for this operator.
- A specialized GEMV trace for one row but reconstruction for four rows shows
  shape-dependent dispatch; compare both timing results before drawing conclusions.
- Reconstruction-only time is a separate measurement, not an additive accounting
  of the normal call. Its ratio is a proxy; allocations, fusion and cache state
  differ, so it is not a precise percentage attribution.
- Small or inconsistent gaps weaken the reconstruction hypothesis and motivate
  profiling other operations and controlled full-model fixed-token workloads.
- Even a clear projection result does not establish the cause of the entire
  model's measured 2.22x slowdown. No automatic whole-model finding is emitted.

Helper tests (no model, quantized operation or GPU timing):

```bash
python -m unittest discover -s tests -p test_mps_quantized_projection.py -v
```


## Observed results: November 13, 2026

Evidence:

- [Initial query-projection run: two cases](../../results/raw/mps_quantized_projection_20261113T051301196956Z.json).
- [Expanded query/up/down run: six cases](../../results/raw/mps_quantized_projection_20261113T051721845333Z.json).

Both reports completed successfully on MPS with PyTorch 2.15.0 and bitsandbytes
0.50.2. The optional `kernels` distribution was absent, and the instrumented
backend reported Hub kernels unavailable. Every tested shape followed
`weight_reconstruction_then_linear_observed`.

Expanded-run medians, in milliseconds per call:

| Projection | Token rows | Normal 4-bit | Reconstruct each call + linear | Predecoded FP16 | Original FP16 | Reconstruction only | Normal / predecoded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `q_proj` | 1 | 0.5278 | 0.5256 | 0.1535 | 0.1438 | 0.4178 | 3.44x |
| `q_proj` | 4 | 0.5618 | 0.5645 | 0.1743 | 0.1621 | 0.4212 | 3.22x |
| `up_proj` | 1 | 2.3930 | 2.3847 | 0.5983 | 0.5974 | 1.8743 | 4.00x |
| `up_proj` | 4 | 2.4130 | 2.4012 | 0.6078 | 0.6054 | 1.8720 | 3.97x |
| `down_proj` | 1 | 2.3928 | 2.3805 | 0.5994 | 0.5954 | 1.8710 | 3.99x |
| `down_proj` | 4 | 2.4202 | 2.4101 | 0.6089 | 0.6066 | 1.8775 | 3.97x |

All six expanded-run cases passed the numerical and repeat controls with zero
value differences among normal quantized, explicitly reconstructed and predecoded
outputs, and zero trace-induced changes. Normal quantized timing closely matched
explicit reconstruction on every call. Moving reconstruction outside timing
reduced projection latency by a factor of **3.22–4.00**. The initial query run
independently showed the same path and pattern (3.24–3.49x).

### Established finding and limits

The controlled intervention shows that repeated NF4 reconstruction and its
associated temporary-memory work add substantial latency for these tested
projections. Predecoding changes when that work occurs while preserving the
observed output exactly. It also keeps a full FP16 matrix resident, giving up
its weight-storage saving; it is not a production quantization fix.

This experiment does not separately attribute time to unpacking arithmetic,
allocation, memory traffic or dispatch. Nor does it account quantitatively for
the full-model 2.22x slowdown: the earlier generation run's kernel dispatch was
not recorded, and this diagnostic is offline with controlled synthetic inputs.
The result strongly supports reconstruction overhead as a contributor in this
configuration, rather than establishing that all quantized inference is slower.

### Related work and research significance

- [Profiling Large Language Model Inference on Apple Silicon: A Quantization Perspective](https://arxiv.org/html/2508.08531v1)
  studies dequantization cost versus bandwidth savings on Apple hardware using
  llama.cpp. Its runtime and formats differ from this bitsandbytes NF4 test.
- [FLUTE: Fast Matrix Multiplications for Lookup Table-Quantized LLMs](https://aclanthology.org/2024.findings-emnlp.724/)
  addresses efficient lookup/unpacking and fused dequantization with matrix
  multiplication for lookup-table quantization on GPUs.
- [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://spcl.inf.ethz.ch/Publications/.pdf/2025_marlin.pdf)
  integrates efficient low-bit conversion and multiplication on NVIDIA GPUs;
  its INT4 implementation is not a direct NF4/MPS replacement.

The general mechanism is established prior work. Our result is a reproducible
engineering diagnosis of this execution path, not a novel quantization algorithm
or a claim of research novelty. Extending the same measurements alone is unlikely
to provide a substantial new contribution. A new kernel, a validated execution
policy, or a previously uncharacterized numerical interaction would require its
own prior-art check and broader evaluation. Adaptive repair based on cache history
remains an untested research hypothesis, not an outcome of these latency tests.
