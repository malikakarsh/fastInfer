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
