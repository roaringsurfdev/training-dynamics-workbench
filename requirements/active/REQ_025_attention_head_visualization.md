# REQ_025: Attention Head Visualization

**Status:** Draft
**Priority:** High (reproduces foundational analysis from Nanda walkthrough)
**Dependencies:** REQ_021b (Analysis Library), REQ_021f (Per-Epoch Artifacts), REQ_008 (Dashboard)
**Last Updated:** 2026-02-08

## Problem Statement

The original Nanda grokking walkthrough (ModuloAdditionRefactored.py, lines 131–145) includes attention pattern visualizations that reveal how each of the model's 4 attention heads attends across input positions during the modular addition task. These visualizations are critical for understanding the model's learned algorithm — they show which input tokens each head attends to when making predictions.

Currently, there is no analyzer to capture attention patterns, no renderer to display them, and no dashboard integration. The only way to view attention patterns is to run ad-hoc notebook code per variant and checkpoint. This prevents systematic comparison of attention behavior across training.

### What This Enables

An attention pattern analyzer and visualization allows the researcher to:
- Watch how attention heads develop their attending behavior across training
- Compare attention patterns between grokking and non-grokking variants at the same epoch
- Identify when heads "lock in" to consistent attention patterns (potential phase change marker)
- Establish baseline attention head understanding before deeper frequency specialization analysis (REQ_026)

## Design

### Analyzer: `AttentionPatternsAnalyzer`

A new analyzer that captures attention patterns from the activation cache. For the Modulo Addition 1-Layer model, the sequence has 3 tokens: `[a, b, =]`. The model has 1 layer with 4 heads.

**Cache access:** `cache["pattern", 0]` returns a tensor of shape `(batch, n_heads, seq_to, seq_from)` where `batch = p * p` (all input pairs). The pattern tensor values represent attention weights (softmax outputs), where each row sums to 1.

**What to store:** For each (head, to_position, from_position) combination, the attention weights are reshaped from `(p*p,)` to `(p, p)` to form a 2D grid indexed by input values `(a, b)`. This captures how attention varies across all input pairs.

Storing all position pairs is cheap. For the Modulo Addition 1-Layer model: 4 heads x 3 to-positions x 3 from-positions = 36 patterns, each `(p, p)`. At p=113, that's 36 x 113 x 113 x 4 bytes ~ 1.8MB per epoch — comparable to existing neuron activation artifacts.

**Per-epoch artifact:**
```
attention_patterns/epoch_{NNNNN}.npz
  patterns: shape (n_heads, n_positions, n_positions, p, p)
```

For the 1-layer, 3-token model: `(4, 3, 3, p, p)`.

**Composition from library:** This analyzer requires a new library function to extract attention patterns from the cache. Existing library functions cover MLP activations and embeddings, but not attention patterns.

### Library Addition

A new function in `analysis/library/activations.py`:

```python
def extract_attention_patterns(
    cache: ActivationCache,
    layer: int = 0,
) -> torch.Tensor:
    """Extract attention patterns from cache.

    Args:
        cache: Activation cache from forward pass
        layer: Transformer layer index

    Returns:
        Tensor of shape (batch, n_heads, seq_to, seq_from)
    """
```

### Renderers

#### 1. Attention Head Grid (Per-Epoch, Dashboard)

A faceted heatmap showing all 4 heads' attention patterns for a selected position pair, faithful to the Nanda walkthrough visualization (lines 138–145).

```python
render_attention_heads(
    epoch_data: dict[str, np.ndarray],  # from load_epoch("attention_patterns", epoch)
    epoch: int,
    to_position: int = -1,   # default: = token
    from_position: int = 0,  # default: a token
    position_labels: list[str] | None = None,  # e.g., ["a", "b", "="]
    title: str | None = None,
) -> go.Figure
```

**Plot elements:**
- Faceted heatmap: one subplot per head (2x2 or 1x4 layout)
- X-axis: `b` values (0 to p-1)
- Y-axis: `a` values (0 to p-1)
- Color: attention weight intensity
- Title includes epoch number and position pair description (e.g., "= attending to a")

This directly reproduces the walkthrough's `facet_col=0` visualization.

#### 2. Single Head Attention (Per-Epoch, Notebook)

A single heatmap for one specific head and position pair.

```python
render_attention_single_head(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    head_idx: int = 0,
    to_position: int = -1,
    from_position: int = 0,
    position_labels: list[str] | None = None,
    title: str | None = None,
) -> go.Figure
```

For detailed examination of individual head behavior.

### Dashboard Integration

Add an attention pattern visualization panel to the Analysis tab, driven by the epoch slider. The default view shows all 4 heads' attention from `= → a` (the primary pattern from the walkthrough).

**Position pair selection:** The dashboard should allow the researcher to choose which position pair to display. This can be a simple dropdown with human-readable labels (e.g., "= attending to a", "= attending to b", "b attending to a"). Default: `= → a`.

**Update behavior:** Loads per-epoch artifact on each slider change. Artifact size is moderate (~1.8MB for p=113) — comparable to neuron activations.

**Conditional rendering:** Attention panels appear only when `attention_patterns` artifacts exist for the selected variant.

### Family Registration

Add `"attention_patterns"` to the Modulo Addition 1-Layer family's `analyzers` list in `family.json`. Add `"attention_heads_grid"` to the `visualizations` list.

## Scope

This requirement covers:
1. Library function: `extract_attention_patterns()` in `analysis/library/activations.py`
2. Analyzer: `AttentionPatternsAnalyzer` with per-epoch artifacts
3. Renderer: `render_attention_heads()` (faceted per-head heatmap, dashboard)
4. Renderer: `render_attention_single_head()` (single-head heatmap, notebook)
5. Dashboard integration: attention panel with epoch slider and position pair selector
6. Family registration: analyzer and visualization identifiers in `family.json`
7. Export from `analysis/library/__init__.py` and `visualization/__init__.py`

This requirement does **not** cover:
- Fourier decomposition of attention patterns (REQ_026)
- Cross-epoch attention pattern comparison or animation
- Summary statistics for attention patterns
- Attention pattern analysis for models with more than 1 layer

## Conditions of Satisfaction

### Library
- [ ] `extract_attention_patterns()` added to `analysis/library/activations.py`
- [ ] Function returns tensor of shape `(batch, n_heads, seq_to, seq_from)` from cache
- [ ] Exported from `analysis/library/__init__.py`

### Analyzer
- [ ] `AttentionPatternsAnalyzer` class in `analysis/analyzers/attention_patterns.py`
- [ ] Per-epoch artifact: `{"patterns": ndarray(n_heads, n_positions, n_positions, p, p)}`
- [ ] Uses `extract_attention_patterns()` library function
- [ ] Reshapes attention from flat batch to `(p, p)` grid per (head, to_pos, from_pos)
- [ ] Registered in `AnalyzerRegistry`

### Renderers
- [ ] `render_attention_heads()` produces a faceted Plotly heatmap (one subplot per head)
- [ ] `render_attention_single_head()` produces a single Plotly heatmap
- [ ] Both accept `to_position` and `from_position` parameters
- [ ] Both accept `position_labels` for human-readable axis annotation
- [ ] Both accept `title` override parameter
- [ ] Both return `plotly.graph_objects.Figure`
- [ ] Exported from `visualization/__init__.py`

### Dashboard
- [ ] Attention heads panel appears when `attention_patterns` artifacts exist
- [ ] Panel absent when artifacts do not exist
- [ ] Epoch slider updates the attention head visualization
- [ ] Position pair selector allows choosing which attention relationship to display
- [ ] Default view: `= → a` (to_position=-1, from_position=0)

### Family Registration
- [ ] `"attention_patterns"` added to analyzers list in `family.json`
- [ ] `"attention_heads_grid"` added to visualizations list in `family.json`

### Tests
- [ ] Library function: returns correct shape for known cache structure
- [ ] Analyzer: conforms to Analyzer protocol
- [ ] Analyzer: produces correct artifact key and shape
- [ ] Renderers: produce valid Plotly figures for valid inputs
- [ ] Renderers: position parameters select correct data slice

## Constraints

**Must have:**
- Faithful to Nanda walkthrough visualization (faceted per-head heatmaps)
- Stores all position pair combinations (not just `= → a`)
- Uses library function for cache access (not inline cache indexing in analyzer)
- Per-epoch artifact storage following REQ_021f pattern

**Must avoid:**
- Hardcoding number of heads or sequence length (use model/cache dimensions)
- Hardcoding position labels (accept as parameter, derive from family if possible)
- Computing Fourier transforms of attention patterns (that's REQ_026)

**Flexible:**
- Exact layout of faceted heatmap (2x2 vs 1x4)
- Color scale choice
- Whether position pair selector is a dropdown or radio buttons
- Dashboard panel placement relative to existing visualizations

## Notes

**Relationship to ModuloAdditionRefactored.py:** Lines 131–145 show two visualizations: (1) a single head's attention pattern reshaped to `(p, p)`, and (2) all 4 heads faceted via `einops.rearrange`. This requirement formalizes both patterns. The analyzer stores complete data; the renderers reproduce both views.

**Position indexing convention:** Negative indices (e.g., -1 for `=`) should be resolved to absolute indices during artifact creation, so the stored artifact uses absolute position indices. Renderers accept either convention for API ergonomics.
