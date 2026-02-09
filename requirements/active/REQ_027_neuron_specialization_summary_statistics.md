# REQ_027: Neuron Frequency Specialization Summary Statistics

**Status:** Draft
**Priority:** High (enables key grokking hypotheses about neuron behavior)
**Dependencies:** REQ_022 (Summary Statistics), REQ_021b (Analysis Library)
**Last Updated:** 2026-02-08

## Problem Statement

The `neuron_freq_clusters` analyzer (name: `neuron_freq_norm`) already produces a per-epoch frequency decomposition matrix of shape `(n_freq, d_mlp)` — the fraction of each neuron's variance explained by each frequency. This data exists for all trained variants. However, there are no summary statistics that distill this matrix into the metrics needed to test core grokking hypotheses:

**Hypothesis 1 — Quality:** p=101, seed=999 never groks because its neurons never specialize in a low frequency above the 0.9 threshold. Frequency quality matters and is predictive.

**Hypothesis 2 — Timing:** The earlier a model's neurons activate on lower frequencies (above threshold), the earlier the model will grok.

**Hypothesis 3 — Saturation:** The more neurons that lock in on lower frequencies, the sooner the model will grok.

**Hypothesis 4 — Frac explained:** The more neurons locked into frequencies above the 0.9 threshold, the lower the test error.

Testing these hypotheses requires per-epoch counts of:
- How many neurons are above the specialization threshold for each frequency
- How many neurons are specialized in low, mid, and high frequency ranges
- The total count of specialized neurons (above threshold, any frequency)

These are small values per epoch — exactly the access pattern that summary statistics (REQ_022) are designed for. Currently, a researcher would need to load every per-epoch `neuron_freq_norm` artifact, extract the matrix, apply thresholds, and count — an expensive and manual process.

## Design

### Extend Existing Analyzer

Add `get_summary_keys()` and `compute_summary()` to the existing `NeuronFreqClustersAnalyzer`. This follows the pattern established by `CoarsenessAnalyzer` in REQ_023. The existing `analyze()` method and its `norm_matrix` output are unchanged.

### Summary Statistics

Each statistic is computed from the `norm_matrix: (n_freq, d_mlp)` returned by `analyze()`.

**Specialization threshold:** A neuron is considered "specialized" for a frequency if its variance fraction for that frequency exceeds a threshold (default: 0.9). This is the `frac_explained` threshold from the research notes.

#### Per-Frequency Counts

| Key | Shape | Description |
|-----|-------|-------------|
| `specialized_count_per_freq` | `(n_freq,)` | Number of neurons above threshold for each frequency |

This is the core metric: for each epoch, how many neurons have locked in to each frequency. Plotting this across epochs shows which frequencies attract neurons and when.

#### Frequency Range Buckets

Frequencies are bucketed into three ranges based on index relative to `n_freq`:
- **Low:** frequencies 1 through `n_low` (first third of non-DC frequencies)
- **Mid:** frequencies `n_low + 1` through `n_mid` (middle third)
- **High:** frequencies `n_mid + 1` through `n_freq` (final third)

The exact boundaries are parameterizable. Thirds is a reasonable starting point.

| Key | Shape | Description |
|-----|-------|-------------|
| `specialized_count_low` | scalar | Neurons specialized in low-frequency range |
| `specialized_count_mid` | scalar | Neurons specialized in mid-frequency range |
| `specialized_count_high` | scalar | Neurons specialized in high-frequency range |
| `specialized_count_total` | scalar | Total neurons specialized in any frequency |

A neuron is counted in a range if its dominant frequency (highest variance fraction) falls in that range **and** its variance fraction exceeds the threshold.

#### Specialization Strength

| Key | Shape | Description |
|-----|-------|-------------|
| `mean_max_frac` | scalar | Mean of each neuron's maximum variance fraction |
| `median_max_frac` | scalar | Median of each neuron's maximum variance fraction |

These measure the overall degree of specialization across all neurons, regardless of which frequency they specialize in. Rising values indicate the neuron population is becoming more frequency-selective.

### Summary Computation

```python
def compute_summary(self, result, context):
    norm_matrix = result["norm_matrix"]  # (n_freq, d_mlp)

    # Per-neuron max and dominant frequency
    max_frac_per_neuron = norm_matrix.max(axis=0)       # (d_mlp,)
    dominant_freq_per_neuron = norm_matrix.argmax(axis=0) # (d_mlp,)

    # Specialization mask: neurons above threshold
    specialized_mask = max_frac_per_neuron >= threshold   # (d_mlp,)

    # Per-frequency counts (only counting specialized neurons)
    specialized_count_per_freq = ...  # count neurons whose dominant freq is f AND above threshold

    # Range buckets
    specialized_count_low = ...   # dominant freq in low range AND specialized
    specialized_count_mid = ...
    specialized_count_high = ...
    specialized_count_total = specialized_mask.sum()

    # Strength
    mean_max_frac = max_frac_per_neuron.mean()
    median_max_frac = np.median(max_frac_per_neuron)
```

### Parameters

- **`specialization_threshold`** (default: 0.9): The variance fraction above which a neuron is considered specialized. This matches the 0.9 threshold referenced in the research hypotheses.
- **`n_range_buckets`** (default: 3): Number of frequency range buckets. Default of 3 gives low/mid/high.

### Renderers

#### 1. Specialization Count Trajectory (Cross-Epoch, Dashboard)

Line plot showing the count of specialized neurons per frequency range over training, with epoch indicator.

```python
render_specialization_trajectory(
    summary_data: dict[str, np.ndarray],
    current_epoch: int,
    title: str | None = None,
) -> go.Figure
```

**Plot elements:**
- Three lines: low, mid, high frequency specialized neuron counts
- Optional fourth line: total specialized count
- X-axis: epoch
- Y-axis: neuron count
- Vertical indicator at current epoch
- Legend identifying each range

**Data source:** `summary.npz` via `ArtifactLoader.load_summary("neuron_freq_norm")`

#### 2. Per-Frequency Specialization Heatmap (Cross-Epoch, Notebook)

Heatmap showing `specialized_count_per_freq` across epochs — frequencies on Y-axis, epochs on X-axis, color = neuron count.

```python
render_specialization_by_frequency(
    summary_data: dict[str, np.ndarray],
    current_epoch: int | None = None,
    title: str | None = None,
) -> go.Figure
```

This gives a detailed view of which frequencies attract neurons and when. Useful for identifying the specific frequencies that matter during grokking.

### Dashboard Integration

Add a specialization count trajectory panel to the Analysis tab, conditional on `neuron_freq_norm` summary data existing. This sits naturally alongside the existing frequency clusters heatmap.

**Update behavior:** Summary data loaded once on variant selection. Only the indicator line updates on slider change.

### Visualization Registration

Add `"specialization_trajectory"` to the Modulo Addition 1-Layer family's `visualizations` list in `family.json`.

## Scope

This requirement covers:
1. Extending `NeuronFreqClustersAnalyzer` with `get_summary_keys()` and `compute_summary()`
2. Summary statistics: per-frequency counts, range bucket counts, specialization strength
3. Renderer: `render_specialization_trajectory()` (cross-epoch range counts, dashboard)
4. Renderer: `render_specialization_by_frequency()` (cross-epoch per-frequency heatmap, notebook)
5. Dashboard integration: conditional trajectory panel
6. Visualization registration in `family.json`
7. Export from `visualization/__init__.py`

This requirement does **not** cover:
- Cross-epoch tracking of individual neuron frequency changes (deferred)
- Cross-variant specialization comparison
- Changes to the existing `analyze()` method or `norm_matrix` artifact
- New analyzer — this extends the existing one

## Conditions of Satisfaction

### Analyzer Extension
- [ ] `get_summary_keys()` added to `NeuronFreqClustersAnalyzer`, returning all summary key names
- [ ] `compute_summary()` added, returning all summary statistics from `norm_matrix`
- [ ] Existing `analyze()` method and `norm_matrix` artifact unchanged
- [ ] `specialized_count_per_freq` correctly counts neurons above threshold per frequency
- [ ] Range bucket counts correctly partition frequencies into low/mid/high
- [ ] `specialized_count_total` equals sum of range bucket counts
- [ ] `mean_max_frac` and `median_max_frac` computed from per-neuron maximum variance fractions

### Renderers
- [ ] `render_specialization_trajectory()` produces a multi-line Plotly figure with epoch indicator
- [ ] `render_specialization_by_frequency()` produces a Plotly heatmap
- [ ] Both accept `title` override parameter
- [ ] Both return `plotly.graph_objects.Figure`
- [ ] Exported from `visualization/__init__.py`

### Dashboard
- [ ] Specialization trajectory panel appears when `neuron_freq_norm` summary data exists
- [ ] Panel absent when summary does not exist
- [ ] Summary data loaded once on variant selection
- [ ] Epoch indicator updates on slider change

### Family Registration
- [ ] `"specialization_trajectory"` added to visualizations list in `family.json`

### Tests
- [ ] Summary keys match `compute_summary()` output keys
- [ ] `specialized_count_per_freq` has shape `(n_freq,)` with non-negative integer values
- [ ] Range counts are non-negative and sum to `specialized_count_total`
- [ ] `mean_max_frac` and `median_max_frac` are in [0, 1]
- [ ] Known input: all-specialized matrix → counts match expectations
- [ ] Known input: no-specialized matrix → all counts are 0
- [ ] Renderers produce valid Plotly figures

## Constraints

**Must have:**
- Extends existing analyzer (no new analyzer class)
- Uses REQ_022 summary statistics infrastructure (inline computation, summary.npz)
- Default threshold of 0.9 matches research hypotheses
- Per-frequency counts are for specialized neurons only (above threshold)

**Must avoid:**
- Modifying the existing `analyze()` return value
- Breaking existing neuron_freq_clusters renderers or dashboard integration
- Hardcoding frequency count or neuron count (derive from norm_matrix shape)

**Flexible:**
- Exact frequency range boundaries for low/mid/high buckets
- Whether range boundaries are defined by index thirds or by domain-meaningful cutoffs
- Additional summary statistics beyond the core set
- Whether the trajectory shows total count as a separate line or not

## Notes

**Re-running analysis required:** Adding summary statistics to an existing analyzer means existing variants will need their analysis re-run (or at minimum, summary-only recomputation) to generate the `summary.npz`. The gap-filling pattern from REQ_022 should handle this: the per-epoch artifacts already exist, so only summary computation is needed. However, the current pipeline computes summaries inline during the epoch loop, not as a post-processing step on existing artifacts. This means a re-run of the analysis pipeline is needed, though it will skip already-computed epochs for per-epoch artifacts.

**Threshold sensitivity:** The 0.9 threshold is a research parameter, not a physical constant. It's possible that different thresholds reveal different dynamics. The implementation should make the threshold easy to adjust, but the default of 0.9 is the starting point based on the researcher's observations.
