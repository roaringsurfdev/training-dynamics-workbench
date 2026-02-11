# Thoughts

Unstructured parking lot for ideas and observations. See Claude.md for context.

---

# Analysis Findings

Insights from exploratory analysis that may inform interpretation or future investigation.

---

## 2026-02-02: Multi-Scale Structure in Activations and Attention

### Discovery

When viewing neuron activation heatmaps at reduced resolution (via bilinear interpolation to 28×28), distinct "blob" regions emerge that aren't visible at full resolution (113×113). This suggests activation patterns have structure at multiple spatial scales.

### Key Observations

**Neuron activations** (first 10 neurons analyzed):
- Most neurons (1-7, 9) have energy concentrated in high frequencies (FFT no-DC ~0.005)
- Neuron 8 is an outlier with genuinely coarse structure (FFT no-DC = 0.962)
- Neuron 0 has moderate low-freq content after DC removal (0.226)

**Attention heads** show distinct specialization patterns:

| Head | FFT no-DC | Gradient | Structure Type |
|------|-----------|----------|----------------|
| 0 | 0.014 | 0.324 | High-freq content, variable intensity |
| 1 | 0.349 | 0.241 | Mixed frequencies |
| 2 | 0.969 | 0.094 | Low-freq, smooth uniform regions |
| 3 | 0.013 | 0.334 | High-freq content, most edge variation |

### Nuanced Interpretation

Initial framing of "blob vs grid" was imprecise. The metrics reveal different structural properties:

- **FFT (no DC)**: Captures spatial frequency of the dominant pattern
  - High = large-scale/low-frequency structure (Head 2's big uniform squares)
  - Low = fine-grained/high-frequency structure

- **Gradient magnitude**: Captures local intensity variation / edge density
  - High = more edges, more intensity variation (Heads 0, 3)
  - Low = smoother, more uniform regions (Head 2)

Head 2's "checkerboard" pattern is actually large flat regions (low gradient) at low spatial frequency (high FFT). Heads 0 and 3 have more continuous intensity variation creating higher gradients despite lower FFT scores.

### Open Questions

1. Does this multi-scale structure emerge during training, or is it present from initialization?
2. Do low-frequency patterns (like Head 2) converge before or after high-frequency patterns?
3. What is the functional role of heads with different scale characteristics?
4. Are there neuron clusters that share similar scale profiles?

### Experimental Validation

Working prototype cells in `ModuloAdditionRefactored.py`:
- Bilinear downsampling experiments (lines 295-334)
- Coarseness metrics: FFT energy ratio, variance preservation, gradient magnitude (lines 340-529)

Screenshots archived in `temp_artifacts/` (session 2026-02-02).

---

## 2026-02-02: Quantitative Metrics for Structural Analysis

### Metrics Tested

**1. FFT Low-Frequency Energy Ratio**
```python
fft_2d = torch.fft.fft2(activation_map)
power = torch.abs(torch.fft.fftshift(fft_2d)) ** 2
# With circular mask for low frequencies (radius = 12.5% of spectrum)
coarseness = power[low_freq_mask].sum() / power.sum()
```
- **Issue**: DC component (mean) dominates the ratio
- **Solution**: Zero out center pixel before computing ratio

**2. Variance Preservation**
```python
downsampled = F.interpolate(map, size=(14, 14), mode='bilinear')
upsampled = F.interpolate(downsampled, size=(p, p), mode='bilinear')
score = upsampled.var() / original_var
```
- Captures how much structure survives aggressive downsampling
- Correlates with FFT metric but provides different perspective

**3. Gradient Magnitude**
```python
dx = activation_map[:, 1:] - activation_map[:, :-1]
dy = activation_map[1:, :] - activation_map[:-1, :]
grad = (dx.abs().mean() + dy.abs().mean()) / 2
```
- Best at distinguishing "smooth uniform regions" vs "variable intensity"
- Complements FFT metric - captures edge density rather than frequency content

### Recommended Combination

For training dynamics analysis, track both:
- **FFT no-DC**: Spatial frequency of dominant patterns
- **Gradient magnitude**: Local intensity variation / smoothness

These capture orthogonal properties and together give a richer picture of structural evolution.

---

# Workbench Improvement Ideas

Suggestions for future enhancements to the Training Dynamics Workbench.

---

## 2026-02-02: Architectural Pressures from Multi-Scale Visualization (REQ_019)

### 1. Data Reuse Between Analyzers

**Observation**: Multiple analyzers need the same underlying data (neuron activations, attention patterns) but currently each analyzer independently loads checkpoints and runs forward passes.

**Specific case**: Both `NeuronActivationsAnalyzer` and a future `DownsampledActivationsAnalyzer` need `probe_cache["post", 0, "mlp"][:, -1, :]`. The downsampled version is just a transform of the same data.

**Potential approaches**:
- Shared "activation extraction" step that multiple analyzers consume
- Hierarchical analyzer pattern where transforms (like downsampling) operate on base artifacts
- Lazy evaluation / caching at the probe_cache level

**Trade-off**: More complexity vs. avoiding redundant computation. May not matter for single-checkpoint analysis but becomes relevant for full training run analysis across many checkpoints.

### 2. Viewing More Neurons

**Observation**: Showing only 5 neurons is arbitrary and limiting. The multi-scale visualization revealed that different neurons have different scale characteristics. Seeing more neurons would help identify patterns.

**Potential approaches**:
- **Paginated/scrollable**: Show N neurons at a time with navigation controls
- **Summary view**: Cluster neurons by scale characteristics and show representatives
- **Configurable N**: Let user choose how many to display (with performance warnings for large N)
- **Small multiples**: Dense grid of tiny heatmaps showing all 512 neurons (patterns visible even if individual details aren't)
- **Filtering**: Show only neurons matching certain criteria (e.g., "neurons where blob structure dominates")

**Related**: The "Neuron Frac Explained by Freq" heatmap already shows all neurons on one axis. Could use this as a selection/filtering mechanism.

### 3. Attention Head Specialization Labels

**Observation**: Different attention heads appear to specialize at different scales. This could be surfaced in the UI.

**Potential approaches**:
- Auto-compute scale metrics and display as labels/badges
- Group heads by scale characteristics in visualizations
- Color-code heads by their structural type

---

*These are observations for future consideration, not immediate action items.*

---

## 2026-02-03: Analysis Report Expansion Ideas

Brainstormed directions for expanding analysis report functionality:

### 1. Report Generation
Structured output summarizing analysis findings in shareable formats (markdown, HTML, PDF). Could include:
- Summary statistics per checkpoint
- Key observations (e.g., "grokking detected at epoch X")
- Embedded visualizations or links to interactive versions

### 2. Comparison Reports
Diff analysis between variants with different parameters. Examples:
- Compare p=113 vs p=97 modulus variants
- Compare different random seeds to identify consistent vs seed-dependent behaviors
- Side-by-side visualization grids

### 3. Aggregate Metrics
Summary statistics or derived insights across checkpoints:
- When does grokking occur? (test loss inflection point)
- Which frequencies dominate at convergence?
- Neuron specialization timeline

### 4. Export Formats
Packaging analysis artifacts in shareable formats:
- Bundled artifact archives for reproducibility
- Standalone HTML reports with embedded data
- Integration with external tools (notebooks, papers)

---

*These ideas emerged from discussion about analysis report requirements.*

---

## 2026-02-10: Post-v0.3.1 Discussion — Future Directions

Three threads emerged from discussion after the v0.3.1 release. Ordered by estimated readiness.

### 1. Turn Detection (Near Requirement-Ready)

**Problem**: Grokking onset ("the turn") is currently identified by visual inspection. Quantitative detection would enable:
- Automated identification of when grokking begins
- Dynamic checkpoint scheduling (uniform spacing by default, then retroactively densify around the turn)
- Downstream analytics that key off the turn epoch

**Observations**:
- The turn appears to precede all other meaningful structural events (frequency specialization, attention head differentiation, trajectory curvature)
- The existing system already handles non-uniform checkpoint schedules gracefully — adding dense checkpoints to a variant and re-running analysis "just worked"
- Candidate signals: test loss inflection, parameter velocity spike, trajectory curvature change

**Scoping notes**:
- User suggests keeping initial scope isolated: detect the turn, report the epoch, done
- Dynamic checkpoint scheduling and downstream analytics would be separate follow-on requirements
- Could start with test loss curvature (simplest signal, doesn't require new analyzers)

### 2. Dashboard UI Redesign — Observed Friction Points

**Context**: Dashboard was designed for 4-5 visualizations, now has 18. User is not frustrated but recognizes cognitive load is growing and the UI is approaching its limits. "I get *lost* in analysis. I love it."

#### Observed Workflows

**A. Epoch discovery loop** (high friction):
1. Scroll down to a summary visualization
2. Mouse over an interesting area to find a good epoch
3. Scroll back to the top
4. Mouse over train/test curves to find the right slider index
5. Type in the index
6. Scroll back down to see the result

User has memorized some indices but can't always remember which variant they belong to (e.g., is index 71 important for 101/999 or 113/999?).

**B. Neuron discovery** (moderate friction):
1. Look at Neuron Frequency Specialization (clusters) plot
2. Mouse over a yellow block to find the neuron ID
3. Navigate to Neuron Index slider input box
4. Enter the neuron ID
5. Scroll to neuron heatmap

**C. Two modes of use**:
- **Summary mode**: Scrolling the full page to find and compare summary information across visualizations
- **Deep dive mode**: Focused on one visualization (especially neuron heatmaps), using frequency clusters to identify neurons of interest

#### Specific Pain Points

1. **Top controls steal vertical space**: Variant selector, epoch slider, neuron index, and other inputs at the top push all content down, requiring constant scrolling
2. **Epoch cross-referencing**: Summary plots show the interesting epoch but don't connect to the epoch selector
3. **Neuron cross-referencing**: Cluster plots show the interesting neuron but don't connect to the neuron selector
4. **Mixed layout**: Summary and detail visualizations interleaved in one long scroll — serves neither mode well

#### Design Patterns Implied

- **Click-to-navigate**: Click a point on any summary plot → epoch slider updates. Click a neuron in clusters → neuron index updates. Eliminates the discovery-to-input indirection.
- **Collapsible left sidebar**: Global controls (variant, epoch, neuron index) in a toggleable left nav instead of top controls. Always accessible without stealing plot space.
- **Summary vs detail modes**: Summary dashboard (compact, all trajectories) vs detail view (one visualization large with controls). Match the two actual usage patterns.
- **Drill from summary** as general principle: Every summary view supports drilling into detail. This is the unifying interaction model.

#### Key Constraints (from user feedback on initial design ideas)

1. **No category tabs for summary scanning.** Dividing visualizations into category tabs and requiring click-through would feel like regression. The current "everything on one page" has value for summary scanning — tabs would fragment that workflow. Any solution must preserve the ability to scan summaries without tab-switching.

2. **Click-to-navigate has dual intent.** Sometimes a click means "navigate to this epoch" (temporal navigation). Sometimes it means "navigate to a deep dive" (structural navigation — e.g., go to neuron heatmap). The interaction model must handle both, and it won't always be obvious which the user wants. This may resolve naturally by visualization type (trajectory click = epoch; cluster click = neuron), but it's a real design tension to watch.

3. **Some summary plots are navigation elements for their adjacent details.** Example: Neuron Frequency Specialization clusters is used to find interesting neurons, then the user clicks through several neurons from the same band to compare their heatmaps. Separating the cluster plot from the heatmap would break this workflow. Summary and detail are sometimes *paired*, not separate modes. This argues against a clean summary/detail split and toward *groups* where each group has its own summary-to-detail drill-down.

#### Performance

- All 18 plots re-render on every epoch slider change
- Lazy rendering (only render visible/active plots) would address this
- Tabbed or collapsible sections would naturally limit what renders

#### Framework Decision: Gradio → Dash

Assessed Gradio's ceiling against emerging requirements:
- **No sidebar layout**: Linear rows/columns, no fixed-position panels during scroll
- **No figure patching**: Every update serializes entire figure to browser
- **Limited click data**: `.select()` returns basic coords, no trace/point identity
- **No client-side callbacks**: Every interaction round-trips to server

Dash provides native solutions for all four. Migration cost is bounded — only `dashboard/app.py` and `dashboard/state.py` are framework-specific. Renderers return `go.Figure` objects that work identically in both.

**Strategy**: Freeze Gradio dashboard (stays available for ongoing analysis), build Dash app in `dashboard_v2/` incrementally. Spike first (5-6 visualizations to prove patterns), then full migration. Training tab stays in Gradio (no interaction problems there).

### 3. Cross-Variant Comparison (Biggest Gap)

**Problem**: The workbench currently shows one variant at a time. Comparing variants requires:
- Manually switching between variants in the dashboard
- Holding differences in memory
- Or exporting images and comparing side by side

**Why it matters**:
- Core scientific question is "how do different parameters affect training dynamics?"
- 10 model variants exist (different primes, seeds) and comparing them is the whole point
- The p=101/seed=999 anomaly was identified through laborious manual comparison

**User's preferred approach**: Start with summary statistics comparison across variants, then drill into details from there. The "drill from summary" pattern applies here too — it's how they'd want to work generally.

**Potential approaches** (not yet scoped):
- Side-by-side dashboard panels (two variants, same visualization)
- Overlay mode for trajectory plots (multiple variants on same axes)
- Diff/delta views showing quantitative differences between variants
- Summary tables aggregating key metrics across all variants

**Dependencies**: May benefit from turn detection first (align comparisons to training phase rather than raw epoch number).

### 4. Notebook Prototyping Workflow

**Observation**: The `export_variant_visualization()` function and renderer library enable a new workflow:
- Prototype new visualizations in notebooks
- User reviews rendered output
- Iterate on design before committing to dashboard integration

This is a useful side effect of REQ_033, not a separate requirement. Worth noting as a development pattern.

---

*These threads are candidates for future requirements. Turn detection is closest to ready.*
