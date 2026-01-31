# REQ_008: Analysis Execution and Visualization Display

## Problem Statement
After a training run completes, we need to trigger analysis and display the resulting visualizations in the dashboard. Currently, analysis requires running separate scripts manually.

The workbench should allow selecting a trained model, running analysis, and viewing the three MVP visualizations in one location.

## Conditions of Satisfaction
- [ ] Interface to select a trained model (path or dropdown)
- [ ] Button to trigger analysis execution
- [ ] Progress indication during analysis
- [ ] Display area for three visualizations:
  - Dominant embedding frequencies
  - Activation heatmaps
  - Neuron frequency clusters
- [ ] Visualizations are interactive (Plotly features work)
- [ ] Can re-run analysis or switch to different trained model

## Constraints
**Must have:**
- Integrates with AnalysisPipeline (REQ_003)
- Displays artifacts from REQ_004, REQ_005, REQ_006
- Gradio framework for UI
- All three visualizations visible without excessive scrolling

**Must avoid:**
- Requiring manual file path entry (should be user-friendly)
- Losing visualization state when switching between models
- Unresponsive UI during analysis execution

**Flexible:**
- Layout of visualizations (tabs, accordion, grid, single column)
- Whether visualizations update live during analysis or appear when complete
- Caching of analysis results to avoid recomputation
- Controls for visualization parameters (color scales, neuron selection, etc.)

## Context & Assumptions
- Analysis runs after training completes (not real-time with training)
- Visualizations are Plotly figures that can be embedded in Gradio
- Analysis may take time depending on number of checkpoints
- User may want to compare visualizations from different training runs
- Assumption: Viewing one training run's analysis at a time is acceptable for MVP

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Select a completed training run from dashboard
- Trigger analysis
- Three visualizations appear in dashboard
- Visualizations are interactive (zoom, hover, etc.)
- Can interpret dominant frequencies, activation patterns, and neuron clusters
- Switching to different trained model updates visualizations correctly

---
## Notes
[Claude adds implementation notes, alternatives considered, things to revisit]
