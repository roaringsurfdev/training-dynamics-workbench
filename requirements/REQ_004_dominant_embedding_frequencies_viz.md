# REQ_004: Dominant Embedding Frequencies Visualization

## Problem Statement
To understand how the model learns to represent modular arithmetic during training, we need to visualize which Fourier frequency components become dominant in the embedding space over time.

The visualization should show how the norm of Fourier basis coefficients evolves across training checkpoints, helping identify when specific frequencies emerge as important features.

## Conditions of Satisfaction
- [ ] Line graph showing `(fourier_bases @ W_E).norm(dim=-1)` for each checkpoint
- [ ] X-axis: Fourier basis components (frequency indices or labels)
- [ ] Y-axis: Normed coefficients
- [ ] Animation or multi-line plot showing evolution across checkpoints
- [ ] Uses FourierEvaluation utilities for computation
- [ ] Visualization clearly shows dominant frequencies (consider threshold highlighting)

## Constraints
**Must have:**
- Fourier bases defined by modulus (period = p)
- W_E is embedding weights (excluding the equals token)
- Plotly format for interactivity
- Generated as artifact consumable by dashboard

**Must avoid:**
- Hardcoded frequency indices (must work for arbitrary p)
- Cluttered visualization (too many frequencies obscure the signal)

**Flexible:**
- Animation vs. interactive slider vs. multi-line plot
- Color scheme for highlighting dominant frequencies
- Whether to show all frequencies or filter to dominant ones
- Exact threshold for "dominance"

## Context & Assumptions
- FourierEvaluation.get_fourier_bases() provides basis vectors
- FourierEvaluation.get_basis_coefficients() computes norms
- Current dominant basis threshold is 1.0 (configurable)
- Visualization should help identify key frequencies like those in key_freqs from original notebook
- Assumption: Dominant frequencies emerge clearly and aren't buried in noise

## Decision Authority
- [x] Make reasonable decisions and flag for review

## Success Validation
- Run analysis on trained modulo addition model (p=113)
- Visualization shows clear emergence of specific frequencies during grokking
- Can identify dominant frequencies visually
- Changing p to different prime produces different but still interpretable visualization
- Artifact loads in Gradio dashboard

---
## Notes
[Claude adds implementation notes, alternatives considered, things to revisit]
