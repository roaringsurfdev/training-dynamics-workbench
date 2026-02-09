REQ_???
### Summary
In examining the results of 4 model variants across two random seeds, one model does not grok: p=101, seed=999. Using a different seed (485), however, results in all models grokking faster, and p=101, seed=485 groks, albeit later than all the other models.

**Hypothesis**: p=101, seed=999 never groks because it never specializes in a low frequency. Neurons that do end up activating on Frequency 6 do so below the .9 threshold. (Frequency quality matters and is predictive)

**Hypothesis**: The earlier a model learns lower frequencies (its neurons activate on lower frequencies), the earlier the model will grok. (Timing matters and is predictive)

**Hypothesis**: The more neurons lock in on lower frequencies, the sooner the model will grok (Saturation matters and is predictive)

**Hypothesis**: The more neurons there are that are locked into frequencies above the .9 threshold, the lower the test error. (Frac explained matters)

### Add Summary statistic(s) for Neuron Frequency Specialization:
For each epoch, how many neurons are above .9 frac explained for a given frequency. This will show how many neurons have "locked in" to a frequency.

It may also be useful to see how many neurons have locked into frequencies in low, middle, and high range.

I would really love to see how many neurons change frequency (and maybe even which ones).

REQ_???
### Summary: Recompute summary statistics from existing artifacts
When an analyzer gains new `compute_summary()` / `get_summary_keys()` methods (e.g., REQ_027 added summary stats to `NeuronFreqClustersAnalyzer`), the pipeline skips epochs that already have artifact files. This means summary.npz is never generated for previously-analyzed variants.

**Workaround**: Delete the analyzer's artifact directory and re-run analysis.

**Desired behavior**: A "recompute summaries" mode that reads existing per-epoch `.npz` artifacts and calls `compute_summary()` on each, without re-running the full analysis (model loading, forward pass, etc.). This should be accessible from both the CLI and the dashboard "Run Analysis" button.

REQ_???
### Summary
Add Attention Head visualization from original Nanda notebook.

On lines 138-145 of ModuloAdditionRefactored.py, there is a visualization of all 4 heads of the Modulo Addition 1-Layer model. (The title on this visualization is misleading)

I would like to add this to the analysis pipeline and dashboard visualization.

REQ_???
### Summary
I would like to see the interplay between frequency specialization in the Attention Heads and grokking.

**Hypothesis**:
Attention Heads, like neurons, become frequency specialists over the course of training.

### Add analysis for Attention Heads
I would like to add visualization that would support analysis of frequency specialization for each of Attention Head over the course of training.