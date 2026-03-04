# Thoughts

Unstructured parking lot for ideas and observations. See Claude.md for context.

---

## 2026-03-02: DMD Residual — Grokking Dynamics Across Variants

### Validated on 113/999 and 101/999 (log scale)

The DMD residual norm (REQ_051), viewed on log scale, is a viable per-site grokking onset signal with distinct signatures across variant archetypes.

**Site hierarchy is consistent**: Post-Embed < Attn Out < MLP Out < Resid Post throughout training. Residuals scale with depth and computation — the embedding layer is nearly always the most linearly predictable, and the residual stream the least. This is a cross-variant invariant.

### 113/999 — Broad Plateau During Grokking Window

Resid Post shows a broad elevated plateau from ~5k–12k epochs — diffuse rather than a sharp spike. After the plateau resolves, residuals drop slightly (~12–13k) corresponding to post-grokking stabilization. The geometry story: the model found its circle early and maintained it, then reorganized during grokking. The DMD sees this as a sustained departure from linearity during the reorganization phase, not a sharp onset event.

### 101/999 — Sharp Spike at ~13k; Site Dissociation

The spike at ~13k–14k is a full order of magnitude above the pre-spike baseline for Resid Post and MLP Out. This is the most prominent feature in the 35K training history. Timing: right at the labeled grokking onset.

**New observation on log scale**: Post-Embed has a dramatic *minimum* at exactly the spike moment — dropping to near 0.05 while MLP Out and Resid Post are maximal. The embedding layer briefly becomes ultra-predictable precisely when downstream computation is most nonlinear. This dissociation between sites was invisible on linear scale.

**Hypothesis (to verify with 109/485)**: The 101/999 spike is more dramatic than 113/999's because the model *did not have an organized circle entering the grokking window* — the Circularity/Fourier Alignment graph shows 101/999 finds a circle only transiently and weakly, right before what the user calls the "second search phase." Without pre-organized geometry, the grokking attempt is a sudden departure from established dynamics rather than a smooth reorganization.

### Post-spike structure for 101/999

After the 13k spike, residuals remain elevated and structured (second bump ~17–25k) before declining. This second bump aligns with the parameter velocity spike at ~24k that prompted extended training. The model didn't recover — it lost a lower-frequency MLP band instead. The DMD residual captures this second dynamics excursion as its own event.

### Connection to Basin Landing Hypothesis

The Post-Embed minimum at the spike moment for 101/999 may be the embedding layer briefly settling into a regular configuration — a momentary alignment — just before the downstream computation attempts (and fails to sustain) the grokking transition. This is the weight-space analog of the transient circle seen in the representational geometry: the model *approaches* the basin briefly (Post-Embed settles, geometry almost organizes) but the substrate (degenerate cos/sin ratio, premature neuron commitment) can't hold it.

### 109/485 — Scaffolding Hypothesis in Direct Form

**Residual:** No sharp spike at grokking onset (~5k). The broad elevated region from 1k–5k simply resolves around grokking — the model had its geometry ready, so the test-loss transition was not a sudden departure from established dynamics. This directly confirms the hypothesis that spike magnitude inversely correlates with pre-grokking geometric organization.

**Reconstruction:** At epoch 3100 (labeled pre-grokking), the ring is already fully and cleanly formed. Grokking occurs at ~5k. The geometry preceded the test-loss drop by at least 1500–1900 epochs. This is the scaffolding hypothesis in its most direct form: representational structure was in place well before generalization performance improved.

**Global PCA:** 72.3% variance in PC1+PC2 (vs 41.3% for 113/999). PC3 is only 9.8% vs 18% for 113/999. The fastest clean grokker has the most compressed representation — class structure concentrated in two dimensions.

**Post-Embed minimum at 1200 (train loss):** Fires at memorization completion, not at grokking. This separates the Post-Embed minimum from the 101/999 case, where it fired at the grokking *attempt*. The phenomenon appears to be a "stable regime achieved" marker: embeddings briefly align whenever the model enters a stable attractor, whether that's memorization or grokking. In 101/999, the failed grokking attempt produces a transient stability that the embedding layer registers, even though the downstream substrate can't hold it.

### 59/485 — Oscillatory Waves; Post-Embed Fires at Second Search Phase

**Residual:** Broad oscillatory waves throughout training — structured chaos, not random noise. A trough is visible at ~13-15k (first attractor approach) before the dynamics resume their oscillatory character. No clean spike; instead the entire training history is elevated and wave-like. Confirms the prediction from the 101/999 vs 109/485 comparison: variants without pre-organized geometry show more dramatic DMD departure events, and the most chaotic geometry produces the most chaotic residual profile.

**Post-Embed minimum at ~24-25k:** Fires at the parameter velocity spike (the "second search phase"). User confirmed: the model begins locking in at exactly this moment. This is consistent with the "stable regime" interpretation: the embedding layer registers whenever the model enters a stable attractor. For 59/485, the first search phase (~13-15k) was a transient, and the real commitment starts at 24-25k.

**Global PCA:** Asymmetric ring at enormous scale (±100), heart-shaped 3D structure rather than torus. PC1 35.3% / PC2 26.2% / PC3 18.5%. The 3D structure shows diagonal scatter in PC2 vs PC3 rather than organized Lissajous — consistent with a two-band-only solution, different from clean grokkers' clean toroidal structure.

**Reconstruction:** DMD dashed lines predict an oscillatory rise-peak-fall trajectory (what most of training looks like), while actual solid lines show sustained growth into the final ring. DMD correctly characterized the oscillatory regime but the final phase transition to ring structure was a dynamics event outside the linear fit. This is expected: approximate DMD captures the dominant long-term linear structure, and the dominant structure for 59/485 is the oscillatory search phase.

**Global PCA compression:** 61.5% in PC1+PC2 — less compressed than 109/485 (72.3%) but more than 101/999 (~27%). The overshooter achieves partial organization; the anomalous variant achieves almost none.

### Cross-Variant Summary

| Variant | Pre-grokking geometry | Residual signature | Post-Embed event |
|---------|----------------------|-------------------|-----------------|
| 109/485 (fast) | Ring fully formed at 3100, grokking at 5k | Broad plateau resolves at grokking | Minimum at train loss (~1200) |
| 113/999 (canonical) | Ring building during grokking | Broad diffuse plateau 5k–12k | Not yet characterized |
| 59/485 (overshooter) | Oscillatory search; ring during second phase | Broad oscillatory waves throughout; trough ~13-15k | Minimum at second search phase (~24-25k) |
| 101/999 (anomalous) | Transient ring at ~18k, collapses | Sharp spike at grokking attempt (~13k) | Minimum at grokking attempt spike |

The pattern: better pre-organized geometry → smoother residual transition → Post-Embed alignment tied to an earlier stable event. 59/485 sits between 113/999 and 101/999 in organization quality: it eventually locks in, but through a chaotic oscillatory search rather than a clean reorganization.

---

*Hypothesis confirmed: spike magnitude inversely correlates with pre-grokking geometric organization. The ring preceding grokking onset in 109/485 is direct scaffolding evidence. Post-Embed minimum is a "stable regime" marker, not a grokking-specific signal — fires at memorization (109/485), grokking transition (113/999 TBD), grokking attempt (101/999), and second search phase commitment (59/485).*

### Geometry-Debt Hypothesis — Evidence from Reconstruction Right Panel

Reading the PC1 trajectory panel across variants reveals a two-sided version of the scaffolding story. The positive case (109/485) was already established; the reconstruction view makes the negative case explicit.

**The distinction:** Early class separation is not inherently good. What matters is whether the model separated *toward the right positions*. 109/485 fanned out directly to its eventual ring positions in a single monotonic motion. 101/999 fanned out to somewhere, overcommitted, and then contracted sharply at the grokking-attempt spike — the contraction is the model trying to undo its prior commitment before attempting reorganization. The second expansion is smaller and doesn't produce a clean ring: the model carried the cost of the bad first geometry through the rest of training.

**Two-sided scaffolding:**
- Correct early geometry → scaffolding accelerates grokking, smooth residual transition, monotonic PC1 fan-out
- Wrong early geometry → scaffolding becomes debt; contraction event required to revise the commitment; grokking delayed or degraded

**Contraction magnitude as a debt proxy:** The size of the contraction reflects how wrong the prior commitment was. 101/999 = large contraction, failed revision, degraded ring. 107/485 = smaller contraction bump, lumpy but more complete ring. 109/485 = no contraction, largest spread, cleanest ring.

**Geometry-debt shows up in multiple metrics — evidence trail:**
- Circularity / Fourier Alignment: 101/999 finds a circle transiently and weakly before the second search phase
- DMD residual (log scale): sharp spike at 13k = moment of geometry revision attempt; second bump at 24k = continued search
- DMD reconstruction right panel: expansion-contraction-partial-reexpansion visible directly as a PC1 trajectory
- Global PCA: 101/999 has lowest PC1+PC2 compression (~27%) — debt left in the representation

The hypothesis is recurring across lenses. It's not a DMD-specific artifact; DMD is just the view that makes the commitment-and-revision sequence most visually explicit.

---

## Future Requirement: Windowed DMD Eigenvalue Spectrum

**Motivation:** The current `centroid_dmd_eigenvalues` view shows global eigenvalues (DMD fit over the entire training trajectory). The epoch cursor does nothing — eigenvalues don't change because they're a property of the global fit. This is correct behavior but not the most interesting view.

**What we want:** Refit DMD over a sliding window centered on each selected epoch, then plot the eigenvalue spectrum at that moment. This would let you watch the eigenvalue spectrum shift as grokking happens — e.g., seeing whether complex conjugate pairs appear (rotational modes) as the ring forms, or whether real eigenvalues near 1 give way to oscillatory modes post-grokking.

**Why it's interesting:** The global DMD sees only monotonic drift (all-real eigenvalues near (1,0)) because the dominant long-run signal is centroids converging to their ring positions. A windowed fit would capture the *local* dynamics — what the trajectory looks like over a 500–2000 epoch window — which may reveal oscillatory structure around grokking that the global model washes out.

**Scope note:** This is a new analyzer (windowed DMD) + a new view, not a modification to CentroidDMD. The window width is a meaningful hyperparameter (too narrow = noisy, too wide = washes out the transition). Eigenvalue stability as a function of window position is itself a grokking onset signal.

---

## 2026-03-02: Circularity Handoff Hypothesis — Ruled Out

**Hypothesis:** During grokking, Attention Circularity rises above MLP/Resid Circularity (handoff), then falls back below (reclaim). These crossover epochs would mark grokking onset and resolution.

**Test:** Implemented `find_circularity_crossovers()` to detect all crossover events across `mlp_out` and `resid_post` reference sites. Marked events on loss curves across variants.

**Result:** The pattern appeared in a handful of models but did not generalize. Not a reliable grokking signal.

**Code left in place:** `find_circularity_crossovers()` in `miscope.analysis.library.geometry` and `event_epochs` param in `render_loss_curves_with_indicator()` — both are generically useful for future event-annotation work.
