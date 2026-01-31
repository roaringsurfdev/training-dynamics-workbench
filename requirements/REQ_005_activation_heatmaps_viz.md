# REQ_005: Activation Heat Maps Visualization

## Problem Statement
To understand what computational patterns neurons learn during training, we need to visualize neuron activation patterns as heatmaps over the input space (a, b).

For a modulo addition task with inputs (a, b, =), each neuron's activation can be visualized as a 2D heatmap where position (a, b) shows the activation strength. This reveals whether neurons are learning frequency-based patterns, positional patterns, or other structures.

The model has 512 MLP neurons, which is too many to show individually. We need a two-stage workflow:
1. **Explore final trained state**: Browse all neurons in the trained model to identify interesting patterns
2. **Historical deep-dive**: Select specific neurons of interest, view their evolution across all checkpoints

Computing activations requires expensive forward passes, so we need to generate activation artifacts once and reuse them for visualization iteration.

## Conditions of Satisfaction
- [ ] Analysis computation generates artifact: neuron activations for all checkpoints
- [ ] Artifact saved to disk (e.g., `neuron_activations.npz`)
- [ ] Artifact structure: shape (n_checkpoints, d_mlp, p, p)
- [ ] **Stage 1 - Explore trained model**: Browse all 512 neurons in final checkpoint
- [ ] **Stage 2 - Historical view**: Select 3-5 neurons, view evolution across checkpoints
- [ ] Heatmap visualization of neuron activations with (a, b) as axes
- [ ] Activation values shown as color intensity
- [ ] Interactive slider for checkpoint selection (historical view)
- [ ] Neuron selector (dropdown or similar) for choosing which neurons to display
- [ ] Visualization loads from artifact (no recomputation)

## Constraints
**Must have:**
- Activations computed from model forward pass on full dataset (done once, saved to artifact)
- Artifact persisted to disk for reuse
- Heatmap dimensions match input space (p x p)
- Plotly format for interactivity
- Works for arbitrary model size (not hardcoded to 512 neurons)
- Two-stage workflow: explore final state, then historical deep-dive

**Must avoid:**
- Showing all 512 neurons × N checkpoints simultaneously (overwhelming)
- Recomputing activations when changing visualization parameters
- Running forward passes during visualization rendering

**Flexible:**
- Neuron selection UI (dropdown, slider, grid of thumbnails)
- Layout (single heatmap, faceted grid, side-by-side comparison)
- Color scale and normalization strategy
- Whether to show multiple neurons simultaneously or one at a time
- Default neuron selection for initial view (first N, highest variance, etc.)

## Context & Assumptions
- Neuron activations available via cache: `cache["post", 0, "mlp"][:, -1, :]`
- Shape: (p², d_mlp) needs to be reshaped to (d_mlp, p, p) for visualization
- Original notebook shows first 5 neurons using faceted heatmaps
- Forward passes are expensive (run once per checkpoint during analysis, not during visualization)
- Activation artifacts enable fast iteration on visualization parameters
- Two-stage workflow: user explores trained model to find interesting neurons, then examines their training history
- Assumption: Interesting neurons show clear patterns (not just noise)
- Assumption: Viewing 3-5 neurons historically is sufficient for MVP
- Assumption: Final trained state provides good signal for identifying interesting neurons

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Run analysis on trained model
- Artifact file created and persisted to disk
- **Stage 1**: Can browse all 512 neurons in final trained checkpoint
- **Stage 1**: Identify 3-5 neurons with interesting patterns
- **Stage 2**: Select specific neurons for historical view
- **Stage 2**: Interactive slider shows neuron evolution across checkpoints
- Heatmap shows clear activation patterns (e.g., frequency-based stripes)
- Changing neuron selection or checkpoint doesn't trigger recomputation (loads from artifact)
- Can identify when selected neurons develop structured responses
- Artifact loads in Gradio dashboard with full interactivity

---
## Notes
[Claude adds implementation notes, alternatives considered, things to revisit]
