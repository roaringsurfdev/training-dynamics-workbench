# REQ_026: Attention Head Frequency Specialization

**Status:** Draft
**Priority:** High (tests hypothesis about head-level frequency specialization during grokking)
**Dependencies:** REQ_025 (Attention Head Visualization), REQ_021b (Analysis Library), REQ_022 (Summary Statistics)
**Last Updated:** 2026-02-08

## Problem Statement

Preliminary analysis of the Modulo Addition 1-Layer model (p=113, seed=999) suggests that attention heads become frequency specialists over the course of training: two heads became low-frequency specialists, one became a medium-high frequency specialist, and one was mixed. This specialization may represent a phase change associated with grokking.

Currently, there is no systematic way to quantify this. The `neuron_freq_clusters` analyzer performs Fourier decomposition of MLP neuron activations, yielding a per-neuron frequency profile. The analogous operation for attention heads would decompose each head's attention pattern into frequency components, revealing which frequencies each head is tuned to.

### Hypothesis

Attention heads, like neurons, become frequency specialists over the course of training. Specifically:
- Each head develops a dominant frequency (or frequency range) during grokking
- This specialization is a phase change marker — it emerges during the grokking transition
- Non-grokking models (e.g., p=101, seed=999) may fail to develop clear head specialization
- The timing and pattern of head specialization may correlate with neuron specialization

### What This Enables

- Quantify per-head frequency tuning at each checkpoint
- Track the emergence of head specialization across training
- Compare head specialization patterns between grokking and non-grokking variants
- Correlate head specialization with existing neuron frequency data

## Design

### Approach: Fourier Decomposition of Attention Patterns

For MLP neurons, the `neuron_freq_clusters` analyzer:
1. Extracts neuron activations, reshapes to `(d_mlp, p, p)` grid
2. Computes 2D Fourier transform using the modular Fourier basis
3. Computes per-frequency variance fractions: `(n_freq, d_mlp)`

The analogous operation for attention heads:
1. Extracts attention patterns for a given position pair, reshapes to `(n_heads, p, p)` grid
2. Computes 2D Fourier transform using the same modular Fourier basis
3. Computes per-frequency variance fractions: `(n_freq, n_heads)`

This reuses the same library functions (`compute_2d_fourier_transform`, `compute_frequency_variance_fractions`). The attention pattern `(n_heads, p, p)` has the same structure as neuron activations `(d_mlp, p, p)` — a batch of 2D grids — so the Fourier pipeline applies directly.

### Which Position Pair?

The primary attention relationship for prediction is `= → a` (the equals token attending to the first operand). This is the pattern analyzed in the Nanda walkthrough and the pattern most likely to show frequency structure.

The analyzer should compute frequency decomposition for the primary position pair (`= → a`). If REQ_025 stores all position pairs, additional decompositions can be added later without changing the analyzer interface.

### Analyzer: `AttentionFreqAnalyzer`

A new analyzer that computes Fourier frequency decomposition of attention patterns per head.

**Input:** Requires the activation cache (for attention patterns) and the analysis context (for `fourier_basis`).

**Composition from library:**
```
ActivationCache
  → extract_attention_patterns()           # (batch, n_heads, seq_to, seq_from)
  → select position pair, reshape to grid  # (n_heads, p, p)
  → compute_2d_fourier_transform()         # (n_heads, n_comp, n_comp)
  → compute_frequency_variance_fractions() # (n_freq, n_heads)
```

All functions except `extract_attention_patterns()` (added in REQ_025) already exist in `analysis/library/`.

**Per-epoch artifact:**
```
attention_freq/epoch_{NNNNN}.npz
  freq_matrix: shape (n_freq, n_heads)  — variance fraction per (frequency, head)
```

This mirrors `neuron_freq_norm` output shape `(n_freq, d_mlp)`, with heads replacing neurons.

### Summary Statistics

Using the REQ_022 infrastructure, the analyzer should produce summary statistics per epoch:

| Key | Shape | Description |
|-----|-------|-------------|
| `dominant_freq_per_head` | `(n_heads,)` | Index of dominant frequency for each head |
| `max_frac_per_head` | `(n_heads,)` | Maximum variance fraction (specialization strength) for each head |
| `mean_specialization` | scalar | Mean of `max_frac_per_head` across all heads |

These enable cross-epoch tracking of which frequency each head specializes in and how strongly.

### Renderers

#### 1. Attention Frequency Heatmap (Per-Epoch, Dashboard)

A heatmap showing frequency decomposition per head, analogous to `render_freq_clusters()` for neurons.

```python
render_attention_freq_heatmap(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    title: str | None = None,
) -> go.Figure
```

**Plot elements:**
- Heatmap: frequencies on Y-axis, heads on X-axis
- Color: variance fraction (0 to 1)
- Head labels on X-axis (Head 0, Head 1, etc.)
- Frequency labels on Y-axis (matching neuron_freq_clusters convention)
- Title includes epoch number

With only 4 heads, this is a narrow heatmap — consider adding per-head annotation showing the dominant frequency.

#### 2. Head Specialization Trajectory (Cross-Epoch, Dashboard)

A line plot showing each head's specialization strength (max variance fraction) over training, with an epoch indicator.

```python
render_attention_specialization_trajectory(
    summary_data: dict[str, np.ndarray],
    current_epoch: int,
    title: str | None = None,
) -> go.Figure
```

**Plot elements:**
- One line per head (4 lines, color-coded)
- Y-axis: max variance fraction (0 to 1) — higher means more specialized
- X-axis: epoch
- Vertical indicator at current epoch
- Legend identifying each head

**Data source:** `summary.npz` via `ArtifactLoader.load_summary("attention_freq")`

**Update behavior:** Summary data loaded once on variant selection. Only the indicator line updates on slider change.

#### 3. Dominant Frequency per Head (Cross-Epoch, Notebook)

A plot showing which frequency each head is tuned to over training — revealing when heads "lock in" to their frequencies.

```python
render_attention_dominant_frequencies(
    summary_data: dict[str, np.ndarray],
    current_epoch: int | None = None,
    title: str | None = None,
) -> go.Figure
```

**Plot elements:**
- One line or scatter per head showing dominant frequency index over epochs
- Color-coded by head
- Y-axis: frequency index (discrete)
- X-axis: epoch

This directly tests the hypothesis: do heads transition from mixed to specialized at a particular training moment?

### Dashboard Integration

Add two attention frequency panels to the Analysis tab:
1. **Attention Frequency Heatmap** — per-epoch, driven by slider
2. **Head Specialization Trajectory** — cross-epoch with indicator

Both conditional on `attention_freq` artifacts existing.

### Family Registration

Add `"attention_freq"` to the Modulo Addition 1-Layer family's `analyzers` list in `family.json`. Add `"attention_freq_heatmap"` and `"attention_specialization_trajectory"` to the `visualizations` list.

## Scope

This requirement covers:
1. Analyzer: `AttentionFreqAnalyzer` with per-epoch artifacts and summary statistics
2. Renderer: `render_attention_freq_heatmap()` (per-epoch frequency decomposition, dashboard)
3. Renderer: `render_attention_specialization_trajectory()` (cross-epoch specialization strength, dashboard)
4. Renderer: `render_attention_dominant_frequencies()` (cross-epoch dominant frequency per head, notebook)
5. Dashboard integration: two conditional panels with epoch slider and variant selection wiring
6. Family registration: analyzer and visualization identifiers in `family.json`
7. Export from `visualization/__init__.py`

This requirement does **not** cover:
- Attention pattern capture (REQ_025 — prerequisite)
- Frequency decomposition for position pairs other than `= → a`
- Cross-variant head specialization comparison
- Correlation analysis between head specialization and neuron specialization (future research)
- Coarseness-style reduction of head frequency data (future, if warranted)

## Conditions of Satisfaction

### Analyzer
- [ ] `AttentionFreqAnalyzer` class in `analysis/analyzers/attention_freq.py`
- [ ] Composes existing library functions (Fourier transform, variance fractions)
- [ ] Uses `extract_attention_patterns()` from REQ_025
- [ ] Per-epoch artifact: `{"freq_matrix": ndarray(n_freq, n_heads)}`
- [ ] Implements `get_summary_keys()` returning summary stat keys
- [ ] Implements `compute_summary()` returning dominant frequencies and specialization strength
- [ ] Registered in `AnalyzerRegistry`

### Renderers
- [ ] `render_attention_freq_heatmap()` produces a Plotly heatmap with heads on X, frequencies on Y
- [ ] `render_attention_specialization_trajectory()` produces a multi-line Plotly figure with epoch indicator
- [ ] `render_attention_dominant_frequencies()` produces a per-head dominant frequency plot
- [ ] All renderers accept `title` override parameter
- [ ] All renderers return `plotly.graph_objects.Figure`
- [ ] Exported from `visualization/__init__.py`

### Dashboard
- [ ] Attention frequency heatmap panel appears when `attention_freq` artifacts exist
- [ ] Head specialization trajectory panel appears when `attention_freq` summary exists
- [ ] Both panels absent when artifacts do not exist
- [ ] Epoch slider updates the frequency heatmap and trajectory indicator
- [ ] Summary data loaded once on variant selection

### Family Registration
- [ ] `"attention_freq"` added to analyzers list in `family.json`
- [ ] Visualization identifiers added to visualizations list in `family.json`

### Tests
- [ ] Analyzer: conforms to Analyzer protocol
- [ ] Analyzer: produces correct artifact key and shape `(n_freq, n_heads)`
- [ ] Analyzer: summary keys match `compute_summary()` output keys
- [ ] Analyzer: dominant frequency indices are valid (within n_freq range)
- [ ] Renderers: produce valid Plotly figures for valid inputs
- [ ] Integration: pipeline produces both per-epoch artifacts and summary.npz

## Constraints

**Must have:**
- Uses modular Fourier approach consistent with neuron_freq_clusters
- Composes existing library functions (no duplicated Fourier transform logic)
- Per-epoch artifact mirrors neuron_freq_norm structure (frequencies x units)
- Summary statistics computed inline via REQ_022 infrastructure

**Must avoid:**
- Reimplementing Fourier transform or variance fraction computation
- Hardcoding number of heads (derive from cache dimensions)
- Hardcoding the position pair in the analyzer (parameterize, default to `= → a`)

**Flexible:**
- Exact set of summary statistics (core set above, can expand)
- Whether the frequency heatmap includes annotations for dominant frequency
- Layout of attention panels relative to existing dashboard panels
- Whether dominant frequency trajectory uses lines, scatter, or step plot

## Decision Log

| Date | Question | Decision | Rationale |
|------|----------|----------|-----------|
| 2026-02-08 | Separate analyzer or extend REQ_025 analyzer? | Separate analyzer | Different output semantics: patterns are raw attention, freq is a decomposition. Mirrors the neuron_activations / neuron_freq_norm separation |
| 2026-02-08 | Which position pair for frequency decomposition? | `= → a` (default, parameterized) | Primary prediction relationship; matches walkthrough analysis; can extend later |
| 2026-02-08 | Modular Fourier vs standard FFT? | Modular Fourier | Consistent with neuron analysis; same basis functions; allows direct comparison |

## Notes

**Prior analysis context:** Earlier analysis of p=113, seed=999 showed 2 low-frequency heads, 1 medium-high head, 1 mixed head. This observation motivates the hypothesis but was done with ad-hoc notebook code. This requirement formalizes the analysis for systematic comparison across variants and training epochs.

**Relationship to neuron frequency analysis:** The output shape `(n_freq, n_heads)` directly parallels `neuron_freq_norm` output `(n_freq, d_mlp)`. This means existing visualization patterns (heatmap rendering, frequency labels) can be reused with minimal adaptation. The key difference is scale: 4 heads vs 512 neurons, so the heatmap will be narrow and individual head trajectories are tractable to plot individually.

**Dependency on REQ_025:** This analyzer needs `extract_attention_patterns()` from REQ_025's library addition. However, if the two requirements are implemented in the same milestone, the library function can be developed once and used by both analyzers.
