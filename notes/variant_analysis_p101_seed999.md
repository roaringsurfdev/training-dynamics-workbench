# Variant Analysis: p=101/seed=999 Anomalous Grokking

**Date:** 2026-02-10
**Control:** p=113, seed=999 (Nanda paper model)
**Subject:** p=101, seed=999 (extended to 35K epochs)

## Context

p=101/999 shows anomalous grokking behavior. The model groks, but uncharacteristically.
Training was extended from 25K to 35K epochs after an uptick in parameter velocity
was observed at the end of the original training run.

Working hypothesis: the model settled on a sub-optimal but near-optimal manifold.

---

## p=113/999 (Control) - Baseline Behavior

Classic, textbook grokking:
- Sharp test loss phase transition around epoch 9-13K
- Fourier spectrum cleans up dramatically: from broad/noisy at epoch 9000 to a sparse set of dominant frequencies at 13500
- Neuron specialization: ~340 total, well-distributed across high (~200), low (~70), and mid (~70) frequency bands
- All 4 attention heads specialize sharply (3 of 4 reach ~0.95+ variance fraction)
- Parameter trajectory shows clean U-shaped arc with decisive phase transition
- W_in PR converges to 19.4 with a sharp SVD cutoff after ~20 singular values
- Landscape flatness: perturbation distribution tightens to ~600p (10^-10) scale post-grok

---

## p=101/999 (Anomalous) - Key Observations

### 1. Gradual grokking instead of sharp phase transition
Test loss declines slowly over thousands of epochs rather than the crisp drop seen in
p=113. Even extended to 35K epochs, the test loss curve has a lingering, gradual
character. Defining a "post-grok epoch" is difficult; attention head specialization
was used as a proxy for when the model committed to its solution.

### 2. Pathologically skewed neuron frequency allocation
Most striking anomaly. At convergence:
- **p=113**: ~200 high, ~70 mid, ~70 low (balanced)
- **p=101**: ~420 high, ~0 mid, ~0 low (almost entirely one band)

The low-freq neurons briefly rose to ~60 around epoch 15-20K then **regressed back
to near zero** by ~25K. The model tried a more balanced allocation and abandoned it.

The bump in parameter velocity around 25K may correspond to the epoch where the
model dropped its low-frequency specialization.

### 3. Incomplete head specialization
Head 2 only reaches ~0.7 variance fraction by 35K and is still rising. In p=113, all
heads locked in by ~15K. The gradual, sigmoid-shaped specialization curves (vs sharp
transitions) mirror the gradual grokking.

### 4. Loss landscape is ~1000x less flat
Possibly the strongest quantitative evidence for the sub-optimal manifold hypothesis:
- **p=113** post-grok perturbation range: ~600p (10^-10 scale)
- **p=101** post-grok perturbation range: ~1u (10^-6 scale)

Three orders of magnitude difference in perturbation sensitivity, despite both models
achieving near-zero loss.

### 5. Fourier spectrum remains noisier
At epoch 20000, p=101 retains more above-threshold frequencies than p=113 at
convergence. The solution is less sparse -- more frequencies are being used rather
than the model committing to a minimal set.

### 6. Effective dimensionality converges similarly
W_in PR converges to ~19.7 for p=101 vs ~19.4 for p=113. Same rank, different
structure. The representational capacity is similar; the allocation across frequencies
is what differs.

---

## Assessment of Sub-Optimal Manifold Hypothesis

The evidence is consistent with a model that:
- **Found a working solution** (loss does go to zero)
- **Uses a less structured representation** (more frequencies, skewed allocation, noisier spectrum)
- **Sits in a less flat minimum** (1000x more perturbation-sensitive)
- **Hasn't fully committed** (heads still specializing at 35K, low-freq neurons regressed)

The low-freq neuron regression is particularly notable -- the model briefly explored a
more balanced (p=113-like) solution and was pulled back. This suggests the
high-freq-dominated manifold is a local attractor for this variant.

Extended training to 35K did not trigger a transition to a qualitatively different
solution. The model stayed on the same manifold.

---

## Cross-Variant Context

- **p=109/999**: Weak low-frequency specialization
- **p=113/485**: No low-frequency specialization
- **p=101/485**: More canonical grokking presentation with characteristic test loss drop (~4K epoch grokking window)

The absence of low-frequency specialization is not unique to p=101/999 and appears
across other variants. It may not be causal for the anomalous behavior. The
**regression** of low-freq specialization (having it then losing it) may be more
distinctive and warrants tracking across additional variants.

The fact that p=101/485 shows canonical grokking suggests the anomalous behavior is
initialization-dependent (seed-specific) rather than a structural property of p=101.
The loss landscape likely has multiple viable minima, and seed 999 routed the model
toward a less clean one.

---

## Open Questions

- Does the low-freq regression pattern appear in any other variants?
- What is the typical grokking window across canonical variants? (Needed to calibrate how extreme p=101/999's protracted transition is)
- Would a learning rate perturbation or other intervention push p=101/999 off its current manifold?
- Is there a number-theoretic relationship between the modulus and which frequency bands dominate, or is allocation purely initialization-dependent?
