# REQ_053: Frequency Quality and Per-Class Error Analysis

**Status:** Active
**Priority:** High
**Related:** REQ_052 (Fourier Frequency Quality Scoring — required dependency), REQ_045 (Fisher Minimum Pair)
**Last Updated:** 2026-03-03

## Problem Statement

Even with a frequency quality score (REQ_052) and overall accuracy curves, there is no way to see *which specific classes* the model is failing on, or to relate class-level failure patterns to the quality of the model's frequency selection.

Two research questions motivate this requirement:

1. **When a model has a poor frequency subset, which residue classes does it fail on?** A model with low frequency quality might fail uniformly across all classes, or it might fail specifically on classes whose signal depends heavily on the missing frequencies. Distinguishing these tells us something about how the model's learned representation breaks down.

2. **Does frequency quality predict per-class failure over training?** If a model's quality score is low early in training, is there a corresponding pattern in which classes are poorly predicted, and does recovery in quality correlate with accuracy recovery by class?

The Fisher minimum (REQ_045) already measures the worst-case class separation in representational geometry. This requirement closes the loop by connecting frequency quality → class accuracy → representational separation in a single set of views.

## Conditions of Satisfaction

### Per-Class Accuracy Artifact

- [ ] A new analyzer `class_accuracy` computes per-epoch per-class accuracy for modulo addition
- [ ] "Class" is the output residue `(a + b) % p` — accuracy is measured per target residue (p classes total)
- [ ] Per-epoch artifact contains:
  - `accuracy_by_residue` — float array of shape `(p,)`, accuracy for each output residue class
  - `accuracy_by_pair` — float array of shape `(p, p)`, accuracy for each (a, b) input pair; stored in artifact (not computed on demand)
  - `overall_accuracy` — scalar
- [ ] Artifacts follow the standard `artifacts/class_accuracy/epoch_{NNNNN}.npz` pattern
- [ ] Accuracy is computed over the full evaluation set (all p² input pairs), not a sample

### Views

- [ ] A per-epoch view `class_accuracy_heatmap` shows a p×p grid of (a, b) pairs colored by accuracy
  - Color scale: 0 (red) to 1 (green), or equivalent diverging scale
  - Title includes epoch and overall accuracy
- [ ] A per-epoch view `class_accuracy_by_residue` shows a bar chart of accuracy per output residue (0..p-1)
  - Bars sorted by residue value (0 to p-1), not by accuracy
- [ ] A cross-epoch view `frequency_quality_vs_accuracy` overlays two traces on a single axis:
  - Overall accuracy over training epochs
  - Frequency quality score over training epochs (from REQ_052 artifacts)
  - Fisher minimum (from REQ_045 artifacts) shown as an optional third trace, toggled via a parameter
- [ ] All views are accessible via `variant.at(epoch).view(name)` through the catalog

### Validation

- [ ] A notebook cell shows the `class_accuracy_heatmap` for at least two distinct epochs of a converging variant — demonstrating that the pattern of per-class accuracy changes visibly over training
- [ ] A notebook cell shows `frequency_quality_vs_accuracy` for two variants with different convergence profiles, demonstrating that quality and accuracy trajectories are meaningfully different between them

## Constraints

**Must:**
- The `class_accuracy` analyzer is a family-contributed concept — it requires a ground-truth evaluation set that the modulo addition family provides
- Views that render class accuracy data are universal instruments — they accept the artifact regardless of which family produced it
- This requirement depends on REQ_052 for the quality trajectory overlay; the per-class accuracy artifact and its views can be implemented independently of REQ_052

**Must not:**
- Implement per-input trace analysis (probing through the network) — that is the scope of a future requirement (REQ_054 or similar)
- Require cross-variant comparison infrastructure — all views operate on a single selected variant

**Decision authority:**
- **Resolved:** `accuracy_by_pair` is stored in the artifact. At p=113 this is ~12,800 floats per epoch (~100KB uncompressed) — manageable, and avoids requiring model checkpoints at render time.
- **Claude decides:** Color scales, axis formatting, and whether to normalize the Fisher minimum to the [0, 1] range for co-plotting

## Context & Assumptions

Per-class accuracy for modulo addition is computed by running the trained model on all p² input pairs, taking the argmax of the output logits, and comparing to the ground-truth residue. This requires the model weights (from the checkpoint) and the input pairs (generated from task parameters).

The relationship between frequency quality and per-class accuracy is the central empirical question. If a model lacks frequency k, then residues whose representation depends heavily on frequency k's components should be predicted poorly. This is a testable structural claim — this requirement creates the tools to test it.

The Fisher minimum (REQ_045) measures the worst-case class pair separation in the representation geometry. Including it in the overlay view allows visual inspection of whether geometric separation failure co-occurs with accuracy failure, and whether both correlate with frequency quality.

## Notes

- The `accuracy_by_pair` artifact may be large (p² floats per epoch × number of epochs). For p=113, that is ~12,800 floats per epoch — manageable but worth noting. If storage becomes a concern, this could be stored at coarser epoch resolution.
- The notebook proof-of-concept mentioned in the roadmap ("understanding failure modes across the network") is explicitly deferred. This requirement stops at per-class accuracy and its relationship to frequency quality. Per-input network tracing is a larger infrastructure effort.
- The research question "do models that start with high-quality frequency subsets converge faster?" is best answered by combining REQ_052's quality trajectory with the cross-variant analysis infrastructure (a future requirement). This requirement lays the groundwork within single-variant analysis.
