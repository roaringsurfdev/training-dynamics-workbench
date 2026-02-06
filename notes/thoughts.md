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
