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
