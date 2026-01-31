# REQ_003: Analysis Pipeline Architecture

## Problem Statement
Current analysis code exists as a script (ModuloAdditionRefactored.py) that runs ad-hoc visualizations. For the workbench, we need a modular analysis pipeline that can:
- Load model checkpoints systematically
- Run analysis computations on each checkpoint (forward passes, Fourier transforms, etc.)
- Generate and persist analysis artifacts to disk
- Enable visualization rendering from artifacts without recomputation
- Be extensible for future analysis types

The analysis engine should separate concerns:
1. **Analysis computation** (expensive): Load checkpoints, run forward passes, compute features → save artifacts
2. **Visualization rendering** (cheap): Load artifacts, apply visual parameters, display → iterate quickly

This separation enables fast iteration on visualizations without re-running expensive computations.

## Conditions of Satisfaction
- [ ] AnalysisPipeline class or module that orchestrates analysis
- [ ] Can load checkpoints from a training run by epoch number
- [ ] Executes analysis computations across specified checkpoints
- [ ] Saves analysis artifacts to disk (persistent, reusable)
- [ ] Artifacts stored in organized directory structure alongside checkpoints
- [ ] Visualization components can load artifacts independently without recomputation
- [ ] Analysis functions are modular and can be composed
- [ ] Progress indication during analysis (simple logging acceptable for MVP)
- [ ] Can resume/skip analysis if artifacts already exist (avoid redundant computation)

## Constraints
**Must have:**
- Reuses existing FourierEvaluation utilities
- Works with safetensors checkpoint format (from REQ_002)
- Separates analysis computation from visualization rendering
- Artifacts persisted to disk (not memory-only)
- Artifacts stored in structured directory alongside training results
- Suggested artifact formats: NumPy `.npz`, HDF5, or similar efficient formats
- Directory structure:
  ```
  results/
    model_p{prime}_seed{seed}/
      checkpoints/
        checkpoint_epoch_{epoch:05d}.safetensors
      artifacts/
        neuron_activations.npz
        fourier_coefficients.npz
        neuron_freq_norm.npz
      metadata.json
      config.json
  ```

**Must avoid:**
- Tight coupling between analysis computations and visualization rendering
- Re-running expensive computations when only changing visual parameters
- Storing Plotly figures as artifacts (store data, render figures on demand)
- Hard-coded visualization logic mixed with analysis logic

**Flexible:**
- Exact artifact file format (NumPy, HDF5, Parquet all acceptable)
- Whether to store one combined artifact or separate files per analysis type
- Granularity of artifact caching

## Context & Assumptions
- Three analysis types needed for MVP: dominant frequencies, activation heatmaps, neuron clusters
- Each analysis requires forward passes on the full dataset (expensive operation)
- Forward passes need to be run once per checkpoint, results cached as artifacts
- FourierEvaluation.py already provides key utility functions
- Disk storage is cheap, recomputation is expensive (favor caching)
- Assumption: Analysis runs after training completes (not real-time)
- Assumption: Full dataset fits in memory for forward pass computation
- Assumption: Artifacts are significantly smaller than full checkpoint files

## Decision Authority
- [x] Propose options for review

## Success Validation
- Can instantiate AnalysisPipeline with a trained model directory
- Can specify which analyses to run
- Analysis generates and saves artifacts to disk for all checkpoints
- Artifacts persist between sessions
- Can load artifacts directly without re-running analysis
- Changing visualization parameters (colors, thresholds) doesn't trigger recomputation
- Re-running analysis on same model skips checkpoints with existing artifacts
- Adding a new analysis type requires minimal changes to pipeline
- Artifact files are inspectable (can load in notebook for debugging)

---
## Notes
[Claude adds implementation notes, alternatives considered, things to revisit]
