# REQ_003: Analysis Pipeline Architecture

## Problem Statement
Current analysis code exists as a script (ModuloAdditionRefactored.py) that runs ad-hoc visualizations. For the workbench, we need a modular analysis pipeline that can:
- Load model checkpoints systematically
- Run analysis on each checkpoint
- Generate visualization artifacts that can be displayed in the dashboard
- Be extensible for future analysis types

The analysis engine should separate the concerns of checkpoint loading, computation, and artifact generation.

## Conditions of Satisfaction
- [ ] AnalysisPipeline class or module that orchestrates analysis
- [ ] Can load checkpoints from a training run by epoch number
- [ ] Executes analysis for specified visualizations across all checkpoints
- [ ] Generates artifacts (files or objects) that can be consumed by dashboard
- [ ] Analysis functions are modular and can be composed
- [ ] Progress indication during analysis (simple logging acceptable for MVP)

## Constraints
**Must have:**
- Reuses existing FourierEvaluation utilities
- Works with safetensors checkpoint format (from REQ_002)
- Separates analysis logic from visualization rendering
- Generates artifacts suitable for Gradio display

**Must avoid:**
- Tight coupling between analysis computations and visualization rendering
- Re-running expensive computations if analysis is interrupted
- Hard-coded visualization logic mixed with analysis logic

**Flexible:**
- Artifact storage format (Plotly JSON, pickled figures, raw data arrays)
- Whether artifacts are stored on disk or kept in memory for MVP
- Caching strategy for intermediate computations

## Context & Assumptions
- Three analysis types needed for MVP: dominant frequencies, activation heatmaps, neuron clusters
- Each analysis may need to run forward passes on the dataset
- FourierEvaluation.py already provides key utility functions
- Assumption: Analysis runs after training completes (not real-time)
- Assumption: Full dataset fits in memory for analysis

## Decision Authority
- [x] Propose options for review

## Success Validation
- Can instantiate AnalysisPipeline with a trained model path
- Can specify which analyses to run
- Analysis generates artifacts for all checkpoints
- Artifacts can be loaded and displayed independently
- Adding a new analysis type requires minimal changes to pipeline

---
## Notes
[Claude adds implementation notes, alternatives considered, things to revisit]
