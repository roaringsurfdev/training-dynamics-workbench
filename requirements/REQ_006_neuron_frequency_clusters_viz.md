# REQ_006: Neuron Frequency Cluster Visualization

## Problem Statement
To understand which neurons specialize in detecting which Fourier frequencies, we need to visualize the fraction of each neuron's activation variance explained by each frequency component.

This analysis performs 2D Fourier transform on neuron activation patterns and computes how much each frequency contributes to each neuron's response. The result is a heatmap showing which neurons "belong" to which frequency clusters.

The current visualization (from ModuloAdditionRefactored.py line 202) has a problematic legend that obscures the data. The visualization needs cleaner presentation.

## Conditions of Satisfaction
- [ ] Analysis computation generates artifact: neuron_freq_norm matrix per checkpoint
- [ ] Artifact saved to disk (e.g., `neuron_freq_norm.npz`)
- [ ] Artifact structure: shape (n_checkpoints, p//2, d_mlp)
- [ ] Visualization loads from artifact (no recomputation of Fourier transforms)
- [ ] Heatmap showing neuron_freq_norm: neurons (x-axis) vs frequencies (y-axis)
- [ ] Color intensity shows fraction of variance explained (0 to 1)
- [ ] Legend minimal or removed to avoid obscuring data
- [ ] Y-axis labels readable (sparse labeling or hover tooltips)
- [ ] Interactive slider to view evolution across checkpoints
- [ ] Clear identification of neuron clusters for each key frequency

## Constraints
**Must have:**
- Computation follows existing algorithm from ModuloAdditionRefactored.py (lines 190-202)
- 2D Fourier transform using fourier_basis (done during analysis, saved to artifact)
- Fraction normalized by total power across all frequencies
- Artifact persisted to disk for reuse
- Plotly format for interactivity
- Generated as artifact consumable by dashboard

**Must avoid:**
- Cluttered legend that blocks data (current issue)
- Unreadable y-axis with 50+ frequency labels
- Misleading color scales (should clearly show 0-1 range)
- Recomputing Fourier transforms during visualization rendering

**Flexible:**
- Y-axis labeling strategy (sparse labels, hover tooltips, or remove entirely)
- Color scheme (as long as it's clear)
- Whether to filter to only show neurons with strong frequency specialization
- Whether to annotate dominant frequency clusters

## Context & Assumptions
- Algorithm: 2D Fourier transform of neuron activations, compute power in specific frequency terms
- For each frequency k, sum power from terms (0,0), (2k-1, 2k-1), (2k, 2k) (constant, sin, cos)
- Normalize by total power to get fraction explained
- Original uses imshow with frequency indices on y-axis
- 2D Fourier transform is computationally expensive (should be computed once, cached in artifact)
- Requires neuron activations (from REQ_005 artifact) as input
- Artifact enables fast iteration on visualization parameters (color scales, labels, etc.)
- Assumption: Clear clusters emerge where neurons specialize
- p=113 means 56 frequencies on y-axis (too dense for readable labels)

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Run analysis on trained model
- Artifact file created and persisted to disk
- Visualization loads from artifact without recomputing Fourier transforms
- Heatmap clearly shows neuron frequency specialization
- No legend obscuring the visualization
- Y-axis labels are readable (sparse or via tooltips)
- Interactive slider allows viewing cluster evolution across checkpoints
- Can identify which neurons respond to which frequencies
- Hover tooltips provide exact values
- Changing visualization parameters (colors, labels) updates immediately without recomputation
- Artifact loads in Gradio dashboard with full interactivity

---
## Notes
[Claude adds implementation notes, alternatives considered, things to revisit]
