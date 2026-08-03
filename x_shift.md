# X-Shift: Explanation-Oriented Adversarial Attack

## Overview

X-Shift is an iterative gradient-based adversarial attack designed to manipulate explanation maps while preserving the original model prediction.

Unlike conventional adversarial attacks that optimize the classification loss, X-Shift directly optimizes the similarity between image patch embeddings and a target text embedding. Several auxiliary objectives are introduced to maintain prediction consistency and improve optimization stability.

The overall optimization objective is

\[
\mathcal{L}
=
\mathcal{L}_{xai}
+
\lambda_{pred}\mathcal{L}_{pred}
+
\lambda_{patch}\mathcal{L}_{patch}
+
\lambda_{ent}\mathcal{L}_{entropy}.
\]

The attack iteratively updates the image using projected gradient ascent under an L0 sparsity constraint.

---

# Pipeline

```
Clean Image
      │
      ▼
CLIP Vision Encoder
      │
      ├──────── CLS embedding
      │
      └──────── Patch embeddings
                    │
                    ▼
      Compute similarities with
      target text embedding
                    │
                    ▼
      Compute four losses

      Lxai
      Lpred
      Lpatch
      Lentropy

                    │
                    ▼
      Total Loss

                    │
                    ▼
 Gradient Ascent

                    │
                    ▼
 Top-K Projection (L0)

                    │
                    ▼
      Clip Image

                    │
                    ▼
     Next Iteration
```

---

# Step 1. Forward Pass

Given an image

\[
x \in [0,1]^{H\times W\times3},
\]

the CLIP image encoder outputs

- CLS embedding

\[
z_{cls}
\]

- Patch embeddings

\[
\{z_p\}_{p=1}^{P}
\]

where

P = number of image patches.

All embeddings are normalized.

---

# Step 2. Compute Patch Similarities

Let

\[
z_t
\]

be the normalized target text embedding.

For every patch

\[
p,
\]

compute

\[
s_{p,t}
=
z_p^T z_t.
\]

Implementation

```python
patch_features = F.normalize(patch_features, dim=-1)
text_feature = F.normalize(text_feature, dim=-1)

similarity = patch_features @ text_feature.T
```

shape

```
[B, P]
```

---

# Step 3. Explanation Manipulation Loss

Goal

Move the explanation toward the target class.

Only the Top-K patches are optimized.

Loss

\[
\mathcal{L}_{xai}
=
-\frac1K
\sum_{i\in TopK}
s_{i,t}
+
\alpha
\cdot
\frac1{P-K}
\sum_{i\notin TopK}
s_{i,t}.
\]

Interpretation

maximize similarity of important patches

while suppressing all remaining patches.

Implementation

```python
topk = similarity.topk(K, dim=1)

mask = torch.zeros_like(similarity)
mask.scatter_(1, topk.indices, 1)

loss_top = -(similarity * mask).sum() / K

loss_other = (
    similarity * (1-mask)
).sum() / (P-K)

loss_xai = loss_top + alpha * loss_other
```

---

# Step 4. Prediction Preservation

We do not want to fool CLIP.

Instead,

the original prediction should remain unchanged.

Using CLS embedding,

\[
\mathcal{L}_{pred}
=
-\log
\frac
{\exp(z_{cls}^Tt_y)}
{\sum_c\exp(z_{cls}^Tt_c)}.
\]

Implementation

```python
logits = cls_feature @ text_features.T

loss_pred = F.cross_entropy(
    logits,
    original_label
)
```

---

# Step 5. Patch Margin Loss

The target similarity should dominate every competing class.

For every patch

\[
\max(0,
s_{p,c}-s_{p,t}+m)
\]

Implementation

```python
target_score = patch_sim[:, :, target]

other_score = patch_sim.clone()

other_score[:, :, target] = -1e9

max_other = other_score.max(dim=-1).values

loss_patch = F.relu(
    max_other
    - target_score
    + margin
).mean()
```

This encourages

```
target similarity
>
other similarities
```

for every patch.

---

# Step 6. Entropy Loss

The attack should create concentrated explanations instead of diffuse heatmaps.

Define

\[
m_p
=
\frac
{\exp(s_p)}
{\sum_q\exp(s_q)}.
\]

Entropy

\[
\mathcal{L}_{entropy}
=
\sum_p
m_p
\log m_p.
\]

Since Shannon entropy is

\[
-\sum p\log p,
\]

this objective minimizes entropy.

Implementation

```python
prob = F.softmax(similarity, dim=1)

loss_entropy = (
    prob * torch.log(prob + 1e-8)
).sum(dim=1).mean()
```

---

# Step 7. Total Loss

```python
loss = (
    loss_xai
    + lambda_pred * loss_pred
    + lambda_patch * loss_patch
    + lambda_ent * loss_entropy
)
```

---

# Step 8. Gradient Update

Gradient ascent

```python
loss.backward()

delta += step_size * delta.grad.sign()
```

or

```python
x_adv = x_adv + eta * grad.sign()
```

---

# Step 9. Sparsity Projection

Only k pixels are allowed to change.

Compute

```
delta = x_adv - x
```

Keep only the largest

```
k
```

entries

```python
delta = topk_projection(delta, k)

x_adv = x + delta
```

Equivalent to

\[
\delta
\leftarrow
TopK(\delta,k).
\]

---

# Step 10. Clamp

```python
x_adv.clamp_(0,1)
```

---

# Complete Algorithm

```python
x_adv = x.clone()

for step in range(T):

    patch_feature, cls_feature = model(x_adv)

    similarity = patch_feature @ target_embedding

    loss_xai = ...

    loss_pred = ...

    loss_patch = ...

    loss_entropy = ...

    loss = (
        loss_xai
        + lambda_pred * loss_pred
        + lambda_patch * loss_patch
        + lambda_ent * loss_entropy
    )

    grad = autograd(loss)

    x_adv += eta * sign(grad)

    x_adv = topk_projection(x_adv, k)

    x_adv.clamp_(0,1)

return x_adv
```

---

# Summary of Each Loss

| Loss | Purpose |
|------|---------|
| **Lxai** | Shift explanation toward target text |
| **Lpred** | Preserve original prediction |
| **Lpatch** | Increase target dominance for every patch |
| **Lentropy** | Produce compact explanation maps |
| **Top-K projection** | Enforce sparse perturbations |