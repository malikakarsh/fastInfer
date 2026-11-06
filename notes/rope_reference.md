# RoPE reference: first milestone

RoPE rotates pairs of features in each token's query and key vectors before
attention compares them. It does not combine adjacent tokens or change values V.
Start with the readable PyTorch reference; no custom GPU kernel, model replacement
or performance claim is part of this milestone.

## Pair layout

Qwen's pinned Transformers4.46.3 implementation pairs the two halves of a vector.
For `[a,b,c,d]`, the pairs are `(a,c)` and `(b,d)`. A90degree rotation of every
pair changes `[5,2,1,3]` into `[-1,-3,5,2]`. This is the same two-number rotation
learned earlier, simply applied to different feature indices.

For each pair `(left,right)`:

```text
new_left  = left * cos(angle) - right * sin(angle)
new_right = left * sin(angle) + right * cos(angle)
```

Source checked: [pinned Qwen rotation and application](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/qwen2/modeling_qwen2.py#L159).

## Functions

`rope_cos_sin(position_ids, head_dim, theta=10000, dtype=...)` builds angle tables.
`theta` here names the frequency base, NOT the angle of a particular rotation.
For pair index j and vector width D:

```text
rate[j] = 1 / theta ** (2*j/D)
angle[position,j] = position * rate[j]
```

Angles use radians (pi radians equals180degrees). The numerical default is a
reference convenience: pass the actual model's configured base for integration.
No dynamic or other scaled RoPE variants are implemented. Frequencies and trig
useFP32, then coefficients are cast to the requested FP16/FP32dtype.

`reference_rope(q,k,cos,sin)` applies the tables using separate multiply/add
operations in the input dtype. Tables have `[batch,tokens,head_dim]` shape;
Q/K have `[batch,heads,tokens,head_dim]` shape and may have different head counts.
One head is a separate attention calculation with its own Q/K vectors. All heads
share the same position tables. This reference permits noncontiguous tensors
and has no custom backward restriction. It requires finite inputs; positions
are supplied as nonnegative integers by the caller.

## Validation and next step

Five reference tests cover exact known rotations/split-half indexing, position0,
different frequencies, independent scalar rotations on FP16/FP32, unequal Q/K
head counts, unchanged inputs, norm preservation, shared shifts of Q/K positions,
and rejected shapes/dtypes/base values. Syntax passed locally; numerical tests
are pending in the existing Kaggle interpreter (no local PyTorch installation).
Run `scripts/kaggle_rope_check.py` with notebook Python to repair NumPy/project
imports and run only these tests. It does not download Qwen or run benchmarks.
Next: inspect/understand the reference output and array layout, then implement
and benchmark a bounded Triton rotation using the same precomputed tables.

## Reference confirmed; first Triton rotation prepared

User's Kaggle run passed all five reference tests in0.020s. This is numerical
reference validation, not a performance result or a real-Qwen comparison.

`kernels/triton_rope.py` adds `triton_rope(q,k,cos,sin)`. It does not call or
replace `rope_cos_sin`. The complete calling flow is explicit:

```python
from fastinfer.kernels.rope import rope_cos_sin, reference_rope
from fastinfer.kernels.triton_rope import triton_rope

# q and k already exist on CUDA, with [batch,heads,tokens,head_dim] shapes.
# position_ids has [batch,tokens] shape on the same device.
cos, sin = rope_cos_sin(position_ids, q.shape[-1], dtype=q.dtype)
expected_q, expected_k = reference_rope(q, k, cos, sin)
actual_q, actual_k = triton_rope(q, k, cos, sin)
```

Both rotations receive exactly the same Q/K and coefficient arrays. There is
one GPU launch covering both Q and K, one program per head/token vector, four
warps and next-power-of-two width masking. Initial contract: contiguous CUDA
FP16/FP32, positive even head widths2..256, separate outputs, inference only.
It reads each feature's split-half partner and applies its own cosine/sine
coefficients. Each multiply is rounded to input dtype before addition, matching
the eager reference; multiply-add fusion is disabled. Coefficient generation
and Q/K projection are unchanged, and no model replacement is installed.

Four GPU tests cover known90degree rotations/identity, batches/unequal heads,
multiple seeds and widths2/6/64/66/128/256, FP16/FP32, masked tails, offset positions,
input preservation, asymmetric per-feature coefficients, supported layouts and
no-backward validation. They require exact equality with the eager reference
on these tested inputs. CUDA compilation/numerics remain untested locally;
syntax checks passed. Run `scripts/kaggle_rope_check.py --gpu` with notebook
Python to restore imports, check CUDA and run all nine RoPE tests. No timing,
model download or table-generation Triton kernel is part of this check.
