# REQ_005: Activation Heat Maps Visualization

## Problem Statement
To understand what computational patterns neurons learn during training, we need to visualize neuron activation patterns as heatmaps over the input space (a, b).

For a modulo addition task with inputs (a, b, =), each neuron's activation can be visualized as a 2D heatmap where position (a, b) shows the activation strength. This reveals whether neurons are learning frequency-based patterns, positional patterns, or other structures.

The model has 512 MLP neurons, which is too many to show individually. We need a strategy to select representative or interesting neurons.

## Conditions of Satisfaction
- [ ] Heatmap visualization of neuron activations with (a, b) as axes
- [ ] Activation values shown as color intensity
- [ ] Animation or selection mechanism to view evolution across checkpoints
- [ ] Strategy for selecting which neurons to visualize (first N, most active, most varied, etc.)
- [ ] Generated as artifact consumable by dashboard

## Constraints
**Must have:**
- Activations computed from model forward pass on full dataset
- Heatmap dimensions match input space (p x p)
- Plotly format for interactivity
- Works for arbitrary model size (not hardcoded to 512 neurons)

**Must avoid:**
- Trying to visualize all 512 neurons at once (overwhelming)
- Expensive recomputation of activations for each visualization

**Flexible:**
- Which neurons to show (first 5, random sample, highest variance, etc.)
- Layout (single large heatmap, faceted grid, selector dropdown)
- Color scale and normalization
- Whether to show multiple neurons simultaneously or one at a time

## Context & Assumptions
- Neuron activations available via cache: `cache["post", 0, "mlp"][:, -1, :]`
- Shape: (p², d_mlp) needs to be reshaped to (d_mlp, p, p) for visualization
- Original notebook shows first 5 neurons using faceted heatmaps
- Assumption: Interesting neurons show clear patterns (not just noise)
- Assumption: Showing 3-5 representative neurons is sufficient for MVP

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Run analysis on trained model
- Heatmap shows clear activation patterns (e.g., frequency-based stripes)
- Can view activation patterns for multiple checkpoints
- Can identify when neurons develop structured responses
- Artifact loads in Gradio dashboard

---
## Notes
[Claude adds implementation notes, alternatives considered, things to revisit]
