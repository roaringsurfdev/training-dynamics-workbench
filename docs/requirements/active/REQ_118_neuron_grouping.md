# REQ_118: Neuron Grouping Primitive

**Status:** Draft
**Priority:** High — on the critical path for REQ_117's parameter track. The DMD backbone narrative depends on a clean grouping primitive being in place before parameter DMD can ship in the architecturally correct shape.
**Branch:** TBD
**Supersedes:** None. Net-new capability.
**Dependencies:**
- REQ_109 (measurement primitives) — grouping outputs follow REQ_109's pure-input pattern; clustering metrics already in `library/clustering.py` are reused for group-summary computation.
- REQ_106 (analysis layering) — grouping is a transform step, consumed by downstream analyzers; it does not embed extract or load logic.
**Downstream consumers:**
- REQ_117 (DMD reorganization) — parameter track. Hard dependency.
- Atlas: `group_geometry` (planned-consolidation absorbing `neuron_group_pca`, `freq_group_weight_geometry`, `intragroup_manifold`) — operates on whatever grouping this primitive produces.
- Atlas: `neuron_dynamics` (rename pending to `group_membership_dynamics`) — once grouping is basis-independent, the dynamics analyzer tracks how neurons move between groups.
**Attribution:** Engineering Claude (under user direction)

---

## Problem Statement

Today the project's notion of a "neuron group" is implicitly Fourier-derived. `neuron_freq_norm` from `neuron_dynamics` assigns each neuron a dominant frequency; downstream analyzers (`neuron_group_pca`, `freq_group_weight_geometry`, `intragroup_manifold`, the existing transient-frequency work) treat that frequency assignment as the group label. This works for modadd because Fourier is the right basis for that task. It does not work as a universal capability:

1. **The grouping is task-specific.** A non-Fourier task — a future induction-head model, a popcount classifier, a 2L MLP variant whose neurons partition by some other learned structure — has no Fourier basis to derive grouping from. The current implicit grouping cannot run.

2. **The grouping is invisible.** It lives inside the analyzers that consume it. There is no single artifact that says "for variant X at epoch Y, here is the group assignment for every neuron." Cross-analyzer comparison ("does the PCA group structure match the weight-geometry group structure?") is not directly possible because each analyzer derives its own grouping internally.

3. **The grouping is conflated with measurement.** `neuron_dynamics` simultaneously computes "what frequency does each neuron prefer" and "how does that preference change over training." These are two different operations: a grouping (assignment) and a dynamics analysis (movement of assignments over time). Conflating them prevents either from being reused independently.

4. **REQ_117 needs grouping that the modadd family supplies but the analyzer does not encode.** Parameter DMD must operate per-group on weight slices. The grouping must come from outside the analyzer — supplied by the family for tasks that have a natural basis, supplied by a data-driven clustering for tasks that do not. The analyzer does not know which.

This REQ extracts grouping into a single primitive with a family-override mechanism. The primitive operates on activations or weight signatures; the modadd family overrides with its Fourier-derived grouping; downstream analyzers consume group assignments without knowing or caring how they were produced.

The architectural rule from PROJECT.md applies directly: families are context providers; analytical instruments are universal. Grouping today violates the rule by being implicitly Fourier-locked inside universal analyzers. This REQ corrects the violation.

---

## Conditions of Satisfaction

### Primitive

- [ ] **`miscope/analysis/library/grouping.py`** exposes pure-input forms following REQ_109's convention:
  - `group_neurons(features, n_groups=None, method='kmeans') -> GroupAssignment` — base function. Input is an `(n_neurons, n_features)` matrix; output is the canonical result type.
  - `group_neurons_summary(group_assignment, features) -> GroupSummary` — derives per-group centroids, dispersion, separation metrics. Reuses `library/clustering.py` primitives.
- [ ] **`miscope/core/grouping.py::GroupAssignment`** — canonical return type. Fields: `assignments` (length `n_neurons` integer array), `n_groups`, `method`, `feature_basis_name` (string identifying what the features were — `'activation_profile'`, `'fourier_w_in'`, `'weight_signature'`, etc.), `confidence` (per-neuron, optional — supports soft assignments where the method produces them).
- [ ] **`miscope/core/grouping.py::GroupSummary`** — per-group statistics. Fields: `centroids`, `radii`, `n_per_group`, `fisher_min`, `fisher_mean`, `dispersion`. Computed by composing `library/clustering.py` primitives.
- [ ] **Method registry.** v1 supports `'kmeans'` (data-driven) and `'argmax_by_basis'` (assignment-by-dominant-component, used when features are a basis projection). Adding a method is local: register a function with the canonical input/output signature.
- [ ] **No analyzer-internal clustering.** A grep audit after this REQ lands: no analyzer in `analyzers/` imports `sklearn.cluster` or implements ad-hoc clustering. All grouping flows through this primitive.

### Analyzer wrapper

- [ ] **`miscope/analysis/analyzers/neuron_grouping.py`** — a per-epoch (or cross-epoch — decide during implementation; per-epoch with stable group identity across epochs is the harder shape and the more useful one) analyzer that pulls the appropriate features from a variant, calls the primitive, and writes a `GroupAssignment` artifact.
- [ ] **Feature source is configurable.** Two modes initially:
  - *Activation features* — per-neuron mean activation profile across the dataset at the analyzed epoch (or a stack of profiles across epochs for cross-epoch grouping). Universal — works on any classifier.
  - *Weight features* — per-neuron concatenated `W_in[:, neuron]` and `W_out[neuron, :]`. Universal — works on any architecture with an MLP block.
- [ ] **Group identity stability across epochs.** When the analyzer runs at multiple epochs, group identity is preserved (group 3 at epoch 1000 corresponds to group 3 at epoch 5000). v1: align via centroid distance after clustering each epoch independently. Document the alignment method; accept that early-epoch instability may produce identity flips that need post-hoc correction.
- [ ] **Family override mechanism.** A family can override the analyzer's default feature-source-and-method by registering a grouping function on the family object. The override receives the variant and epoch; it returns a `GroupAssignment` with the same canonical type as the universal path. `ModularAdditionFamily` registers a Fourier-based override that returns argmax-by-frequency assignments.
- [ ] **Artifact contract.** The analyzer writes `GroupAssignment` per epoch (or cross-epoch — match REQ_117's parameter-track expectation when finalizing). The artifact is the single source of truth; downstream consumers do not re-derive grouping.

### Downstream consumer migration

- [ ] **REQ_117 parameter track consumes `neuron_grouping` artifacts directly.** No fallback path that bypasses the artifact for the modadd family. The Fourier grouping is supplied through the family-override mechanism, not through a parallel code path.
- [ ] **Atlas's `group_geometry` planned-consolidation cites this primitive** as the grouping input. The geometry analyzer accepts a `GroupAssignment` artifact and computes its outputs against whatever grouping was supplied. This REQ does not migrate the existing group-geometry analyzers — that migration is part of the `group_geometry` consolidation and is out of scope here. The contract this REQ establishes is what `group_geometry` consumes when its consolidation lands.
- [ ] **`neuron_dynamics` is not modified by this REQ.** The Atlas notes the eventual rename to `group_membership_dynamics` once basis-independent grouping is in place; that rename is deferred. This REQ provides the primitive that makes the rename possible, but does not perform it.

### Validation

- [ ] **Reference variant set:** p109/s485/ds598, p113/s999/ds598, p101/s999/ds598 (REQ_109's reference set).
- [ ] **Family-override parity.** On modadd canon variants, the family-supplied Fourier grouping returned through this primitive matches the implicit grouping currently used by `freq_group_weight_geometry` and `neuron_group_pca`. This is parity validation in REQ_111's sense — same conceptual grouping, surfaced through a clean interface.
- [ ] **Universal-path sanity check.** On modadd canon, the universal data-driven grouping (k-means on weight signatures) produces groups that correlate substantially with the Fourier grouping. They will not be identical (the universal path has no privileged Fourier basis); the question is whether the data-driven shape recovers something close to the family-supplied shape on a task where the right answer is known. Document the agreement quantitatively.
- [ ] **Group identity stability check.** Run the analyzer at every checkpoint on a canon variant. Group identity should remain stable post-grokking; pre-grokking instability is expected and acceptable. Document the cross-epoch alignment quality.

---

## Constraints

**Must:**
- Grouping is a transform-only primitive. Extract is delegated to the analyzer wrapper; load (artifact write) is delegated to the wrapper.
- Family-supplied groupings flow through the family-override mechanism. There is no parallel code path for "family knows the grouping" that bypasses the universal interface.
- The primitive is pure-input: ndarray in, typed result out. No `Variant` import in the pure form. The wrapper is the variant-coupled layer.
- Group summary metrics reuse `library/clustering.py`. No re-implementation of centroid / Fisher / dispersion computation.

**Must avoid:**
- **Coupling clustering to a specific feature source.** The primitive accepts feature matrices, not "activations" or "weights." The wrapper decides which features to pull.
- **Hardcoding `n_groups`.** Either auto-determined by the method (silhouette / elbow heuristic, optional) or caller-supplied. Hardcoding kills the universality.
- **Embedding family logic in the primitive.** The primitive does not know about Fourier. Family overrides live at the wrapper layer.
- **Carrying the `neuron_dynamics` rename forward in this REQ.** The rename is deferred per the Atlas note; this REQ provides the prerequisite but does not perform the rename.
- **Migrating existing group-geometry analyzers.** Out of scope. This REQ defines the contract those analyzers will consume when their consolidation lands.

**Flexible:**
- Whether the analyzer is per-epoch or cross-epoch. Per-epoch with cross-epoch identity alignment is the natural fit for REQ_117's windowed parameter DMD; cross-epoch as a single artifact may be simpler. Decide during implementation; document the choice.
- v1 method set. `kmeans` and `argmax_by_basis` are the documented minimum. Spectral clustering, hierarchical, GMM — all are fine additions if a research need surfaces. Keep the method registry open.
- Group identity alignment algorithm. Centroid distance is the v1 default; Hungarian assignment is a possible upgrade.
- Soft vs. hard assignments. v1 is hard; soft assignments are supported by the canonical type's `confidence` field for methods that produce them.

---

## Architecture Notes

### Why a primitive, not a built-in to each analyzer

Today, every analyzer that needs grouping derives its own. The result is a hidden contract: the geometry analyzer and the dynamics analyzer happen to agree about what the groups are because they both pull from `neuron_freq_norm`, which happens to be deterministic. This contract is not enforced anywhere; if either analyzer's grouping logic drifts, downstream comparisons silently become incoherent.

A single primitive with a single artifact makes the contract explicit. Every analyzer that needs grouping reads the same artifact. Cross-analyzer comparison is meaningful by construction.

### Why the family-override mechanism is the right shape

The architectural rule from PROJECT.md: families are context providers, views and analyzers are universal instruments. Grouping today violates the rule. The fix is not to remove Fourier grouping — it is the right grouping for modadd — but to express it through the rule's intended mechanism.

`ModularAdditionFamily` is the natural home for "I know what the right grouping is for this task." The override registration says, "for this family, use this grouping function." The analyzer's universal path runs unchanged for any family that does not override. The architectural constraint is honored; the practical functionality is preserved.

This is the same shape as `weight_basis_projection` (planned-consolidation in the Atlas): the basis projection is universal; the basis is family-supplied. Grouping has the same structure.

### Why this REQ is small but on the critical path

Most of this REQ's surface area is contract definition — the canonical result type, the family override mechanism, the primitive signature. The implementation cost of v1 is modest: k-means with sklearn or a hand-rolled equivalent, a thin wrapper, a Fourier override. The cost lives in getting the contract right.

Getting the contract right matters because everything downstream of grouping depends on the contract being stable. REQ_117's parameter track will read `GroupAssignment` artifacts; the future `group_geometry` consolidation will read them; the `neuron_dynamics` rename will read them. The contract is load-bearing.

### Identity stability as a research observable

Group identity stability is itself a meaningful dynamical observation. A neuron that switches group membership during training is a neuron whose learned function reorganized. The `transient_frequency` work in the project's notes is essentially an observation about identity instability under the Fourier grouping. Once `neuron_grouping` lands as a universal primitive, "transient groups" generalizes naturally — the question becomes whether neurons leave a group temporarily before returning, or leave permanently.

The Atlas's note that `transient_frequency` may generalize to "transient groups" depends on this primitive. v1 of `neuron_grouping` does not need to solve the transience question; it needs to provide the assignments that future analyzers can compute transience over.

---

## Notes

### Open questions

- **Per-epoch vs. cross-epoch artifact.** REQ_117's parameter track will run windowed DMD per group across all checkpoints. The grouping it consumes can be per-epoch (each window uses that window's grouping) or cross-epoch (a single grouping holds across all windows). The cross-epoch shape is simpler and matches the project's stable post-grokking group identity; the per-epoch shape captures pre-grokking reorganization. The right answer is probably both, with the per-epoch artifact aggregable into a cross-epoch one. Decide during implementation.
- **Default `n_groups`.** Modadd's canonical grouping has a clear answer (one group per dominant frequency); a universal default does not. Auto-detection via silhouette score is the obvious answer; whether it works across the variant set is empirically unknown.
- **Soft assignments for transient neurons.** A neuron whose dominant frequency contribution is split across two frequencies has a meaningful soft assignment. v1 supports it via the canonical type's `confidence` field, but v1's default methods produce hard assignments. Whether to default to soft on the universal path is an empirical question.
- **`neuron_dynamics` migration timing.** The Atlas says "defer the rename until the grouping primitive is in place." Once this REQ lands, the rename is unblocked, but it is deferred to whatever future REQ picks it up.

### Cross-references

- [REQ_117](REQ_117_dmd_reorganization.md) — primary downstream consumer; parameter track is hard-blocked on this REQ.
- [REQ_109](../staging/REQ_109_measurement_primitives.md) — clustering primitives reused for group summary.
- [REQ_111](REQ_111_parallel_analyzer_buildout.md) — parallel construction philosophy. The new `neuron_grouping` analyzer ships alongside the implicit Fourier-grouping logic in `neuron_dynamics`; that logic is not modified by this REQ.
- [Analysis Atlas](../../analysis_atlas.md) — `neuron_grouping` entry (planned-new) updated to point at this REQ. `group_geometry`, `neuron_dynamics`, and `transient_frequency` entries are downstream consumers when their respective consolidations land.
- [PROJECT.md](../../../PROJECT.md) — the architectural rule (universal instruments, family context providers) that motivates the family-override mechanism.
