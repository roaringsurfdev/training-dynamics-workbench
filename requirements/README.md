# Requirements

This directory contains the structured requirements for the Training Dynamics Workbench MVP.

## Requirements Overview

### Phase 1: Training Runner Enhancements
**Goal:** Enhance existing training infrastructure for flexible checkpointing and modern persistence

- [REQ_001: Configurable Checkpoint Epochs](REQ_001_configurable_checkpoint_epochs.md)
  - Enable fine-grained control over checkpoint timing via integer list
  - Supports densifying checkpoints around critical phases like grokking

- [REQ_002: Safetensors Persistence](REQ_002_safetensors_persistence.md)
  - Migrate from pickle to safetensors format for model weights
  - Maintain backward compatibility with existing checkpoints

### Phase 2: Analysis Engine
**Goal:** Modular analysis pipeline for generating visualizations from checkpoints

- [REQ_003: Analysis Pipeline Architecture](REQ_003_analysis_pipeline_architecture.md)
  - Orchestration layer for loading checkpoints and running analysis
  - Extensible framework for adding new analysis types

- [REQ_004: Dominant Embedding Frequencies Visualization](REQ_004_dominant_embedding_frequencies_viz.md)
  - Fourier analysis of embedding space evolution
  - Identify which frequencies emerge during training

- [REQ_005: Activation Heat Maps Visualization](REQ_005_activation_heatmaps_viz.md)
  - 2D heatmaps of neuron activations over input space
  - Reveal computational patterns learned by neurons

- [REQ_006: Neuron Frequency Cluster Visualization](REQ_006_neuron_frequency_clusters_viz.md)
  - Show which neurons specialize in which frequencies
  - Improved presentation (minimal legend to avoid obscuring data)

### Phase 3: Workbench
**Goal:** Gradio-based UI for training and analysis

- [REQ_007: Gradio Dashboard with Training Controls](REQ_007_gradio_dashboard_training_controls.md)
  - Configure and launch training runs through web interface
  - Control modulus, seeds, checkpoints, and other parameters

- [REQ_008: Analysis Execution and Visualization Display](REQ_008_analysis_visualization_display.md)
  - Trigger analysis on completed training runs
  - Display all three visualizations in dashboard
  - Global checkpoint slider synchronizing all visualizations

- [REQ_009: Loss Curves with Epoch Indicator](REQ_009_loss_curves_epoch_indicator.md)
  - Train/test loss curves with synchronized vertical line indicator
  - Visual correlation between slider position and learning curve phase

## Status

All requirements defined. Ready for implementation planning.

## Working with Requirements

To work on a specific requirement:
```
"Work on REQ_003"
```

See `/policies/requirements/templates/requirements_README.md` for collaboration workflow.
