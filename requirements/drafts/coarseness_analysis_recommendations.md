# Coarseness Analysis Recommendations for Grokking Research

## Background Context
First Note: I'm using "blob" and "plaid" to loosely describe what appear to be two different types of neuron activation heatmaps. "Blob" heatmaps show periodic solid circles, squares, ovals surrounded by whitespace. "Plaid" seems like a possible inverse, where the heatmap is filled with lines, and the circles, squares, and ovals are voids or whitespace.

Second Note: I have trained 8 models using this workbench platform:
seed=999, p=97, 101, 109, 113
seed=485, p=97, 101, 109, 113
Only one model fails to grok: seed=999, p=101
All models with seed=485 grok, and they grok earlier than for seed=999

Third Note: Frequencies learned in the embedding weights do not appear to correspond to special frequencies for the modulus.

Fourth Note: Frequencies in the input embeddings (W_E) seem to be selected by the model early on and show stability over training. This seems to contradict the idea that the frequencies are suddenly discovered, leading to a grokking transition period. Evidence of "grokking" might be happening elsewhere in the network. Hence this dive into perceived types of neuron heat maps.

This document contains recommendations for analyzing the "coarseness" (blob vs plaid) metric in modular addition grokking experiments. The key observation is that **grokking models develop "blob" neurons (high coarseness, low-frequency patterns) while non-grokking models remain dominated by "plaid" neurons (low coarseness, high-frequency checkerboard patterns)**.

The particularly interesting case is **p=101, seed=999**, which fails to grok and appears to lack blob neurons, while the same prime with seed=485 does grok (albeit slower than other primes).

## Coarseness Metric Definition

**Coarseness (Low-Freq Energy Ratio)**: The ratio of low-frequency to total power in a neuron's activation pattern.

- **High coarseness (≥0.7)**: "Blob" neurons with large, coherent activation regions
- **Low coarseness (<0.5)**: "Plaid" neurons with fine-grained checkerboard patterns
- **Mid coarseness (0.5-0.7)**: Transitional or mixed patterns

---

## Top 5 Recommended Analyses (Priority Order)

### 1. ⭐ Coarseness Distribution Over Time (HIGHEST PRIORITY)

**Visualization Type**: Animated histogram or heatmap

**What to Plot**:
- **X-axis**: Coarseness bins (0 to 1, suggest 20 bins)
- **Y-axis**: Count of neurons in each bin
- **Animation/Facets**: Different epochs (suggested: 0, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000)
- **Color/Separate plots**: By model (prime, seed) combination

**Why This Matters**: 
Directly tests the hypothesis that grokking models develop more high-coarseness neurons over time, while non-grokking models stay stuck with low-coarseness distributions.

**Expected Results**:
- p=101 seed=999: Distribution remains concentrated at low coarseness (<0.5) throughout training
- Grokking models: Distribution shifts rightward (toward higher coarseness) around grokking time
- Fastest grokking models (p=109 seed=485): Should show earliest rightward shift

**Implementation Notes**:
```python
# For each epoch, compute histogram
coarseness_values = [compute_coarseness(neuron_activation) for neuron in all_neurons]
hist, bins = np.histogram(coarseness_values, bins=20, range=(0, 1))
```

---

### 2. Mean Coarseness vs Epoch (Summary Statistic)

**Visualization Type**: Line plot

**What to Plot**:
- **X-axis**: Epoch
- **Y-axis**: Mean coarseness across all neurons
- **Multiple lines**: One per (prime, seed) combination
- **Optional enhancements**: 
  - Shaded region for standard deviation
  - Show 25th, 50th, 75th percentiles
  - Overlay with test loss (dual y-axis) to show temporal relationship

**Why This Matters**: 
Provides a single trajectory that should clearly separate grokking from non-grokking models. Can directly compare timing of coarseness increase vs test loss drop.

**Expected Results**:
- Grokking models: Mean coarseness increases sharply during or slightly before the grokking transition
- p=101 seed=999: Mean coarseness remains flat and low throughout training
- Correlation: Higher mean coarseness should correlate with lower test loss

**Implementation Notes**:
```python
def compute_mean_coarseness(epoch):
    coarseness_values = [compute_coarseness(neuron) for neuron in all_neurons]
    return {
        'mean': np.mean(coarseness_values),
        'std': np.std(coarseness_values),
        'median': np.median(coarseness_values),
        'p25': np.percentile(coarseness_values, 25),
        'p75': np.percentile(coarseness_values, 75)
    }
```

**Easiest to Implement**: This can be added as a second y-axis on existing training progress plots!

---

### 3. Coarseness vs Neuron Index (Spatial Pattern)

**Visualization Type**: Heatmap

**What to Plot**:
- **X-axis**: Neuron index (0-511)
- **Y-axis**: Epoch (sample key epochs: 0, 1000, 5000, 10000, 15000, 20000, 25000)
- **Color**: Coarseness value (0-1 scale)
- **Facets**: Different (prime, seed) combinations

**Why This Matters**: 
Tests whether specific neuron positions consistently become blob-like, or if it's spatially random. Shows if there are "clusters" of blob neurons that emerge together.

**Expected Results**:
- Grokking models: May show clusters of high-coarseness neurons emerging together
- Early neurons (0-50): Might show earlier blob formation based on previous observations
- p=101 seed=999: Should show uniformly low coarseness across all neuron indices

**Implementation Notes**:
```python
# Create matrix: epochs × neurons
coarseness_matrix = np.zeros((len(epochs), num_neurons))
for epoch_idx, epoch in enumerate(epochs):
    for neuron_idx in range(num_neurons):
        coarseness_matrix[epoch_idx, neuron_idx] = compute_coarseness(neuron_activation)
        
# Plot as heatmap with diverging colormap (e.g., 'RdBu_r')
```

---

### 4. Early Coarseness Predicts Grokking Time

**Visualization Type**: Scatter plot

**What to Plot**:
- **X-axis**: Mean coarseness at epoch 1000 (or 500, tune based on when structure first appears)
- **Y-axis**: Epoch when test loss drops below threshold (grokking time)
  - Suggested threshold: Test loss < 0.01 or 99% test accuracy
  - For non-grokking models, use max epoch or mark as "no grok"
- **Points**: One per model
- **Color/Shape**: By prime or seed
- **Annotations**: Label points with (prime, seed)

**Why This Matters**: 
Tests causality direction - does early blob formation *predict* later grokking? If yes, this supports blob neurons as a necessary precondition for grokking.

**Expected Results**:
- Negative correlation: Higher early coarseness → faster grokking
- p=101 seed=999: Outlier with low coarseness and no grokking (or grokking at max epoch)
- Possible threshold effect: Minimum coarseness needed for grokking to occur

**Implementation Notes**:
```python
# For each model
early_coarseness = mean_coarseness_at_epoch[1000]
grokking_epoch = first_epoch_where(test_loss < 0.01)

# Plot with regression line
plt.scatter(early_coarseness, grokking_epoch)
# Add trend line excluding non-grokking points
```

---

### 5. High-Coarseness Neuron Count Over Time

**Visualization Type**: Line plot

**What to Plot**:
- **X-axis**: Epoch
- **Y-axis**: Count of neurons with coarseness > threshold
- **Multiple lines**: One per (prime, seed) combination
- **Thresholds to try**: 0.6, 0.7, 0.8
- **Optional enhancement**: Add horizontal line showing "minimum viable blob count" if pattern emerges

**Why This Matters**: 
More interpretable than mean coarseness - gives concrete number of blob neurons. Allows statements like "model needs at least 50 blob neurons to grok."

**Expected Results**:
- Grokking models: Increasing blob neuron count over time, with possible sharp increase during grokking
- p=101 seed=999: Blob neuron count stays near zero throughout training
- Possible threshold effect: Minimum number of blob neurons required for grokking

**Implementation Notes**:
```python
def count_blob_neurons(epoch, threshold=0.7):
    coarseness_values = [compute_coarseness(neuron) for neuron in all_neurons]
    return (np.array(coarseness_values) > threshold).sum()

# Track over epochs for multiple thresholds
blob_counts = {
    'threshold_0.6': [],
    'threshold_0.7': [],
    'threshold_0.8': []
}
```

---

## Summary Statistics Infrastructure

To support these analyses, implement a summary statistics computation function:

```python
def compute_summary_stats(model, epoch, cache):
    """
    Compute summary statistics for an epoch.
    Returns dict of statistics that can be logged/plotted over time.
    """
    
    stats = {}
    
    # Get all neuron activations for this epoch
    neuron_activations = get_all_neuron_activations(cache)
    
    # Compute coarseness for each neuron
    coarseness_values = []
    for neuron_idx, activation_map in enumerate(neuron_activations):
        coarseness = compute_coarseness(activation_map)
        coarseness_values.append(coarseness)
    
    coarseness_array = np.array(coarseness_values)
    
    # Basic statistics
    stats['mean_coarseness'] = np.mean(coarseness_array)
    stats['std_coarseness'] = np.std(coarseness_array)
    stats['median_coarseness'] = np.median(coarseness_array)
    stats['min_coarseness'] = np.min(coarseness_array)
    stats['max_coarseness'] = np.max(coarseness_array)
    
    # Percentiles
    stats['p25_coarseness'] = np.percentile(coarseness_array, 25)
    stats['p75_coarseness'] = np.percentile(coarseness_array, 75)
    
    # Blob neuron counts at different thresholds
    stats['num_blob_neurons_0.5'] = (coarseness_array > 0.5).sum()
    stats['num_blob_neurons_0.6'] = (coarseness_array > 0.6).sum()
    stats['num_blob_neurons_0.7'] = (coarseness_array > 0.7).sum()
    stats['num_blob_neurons_0.8'] = (coarseness_array > 0.8).sum()
    
    # Distribution (for histogram visualization)
    hist, bins = np.histogram(coarseness_array, bins=20, range=(0, 1))
    stats['coarseness_histogram'] = hist
    stats['coarseness_bins'] = bins
    
    # Fraction in different coarseness ranges
    stats['frac_low_coarseness'] = (coarseness_array < 0.5).sum() / len(coarseness_array)
    stats['frac_mid_coarseness'] = ((coarseness_array >= 0.5) & (coarseness_array < 0.7)).sum() / len(coarseness_array)
    stats['frac_high_coarseness'] = (coarseness_array >= 0.7).sum() / len(coarseness_array)
    
    return stats
```

---

## Quick Implementation Priority

If you can only implement **one visualization immediately**, choose:

**#2 - Mean Coarseness vs Epoch**

**Reasons**:
1. Easiest to implement (single summary statistic per epoch)
2. Directly comparable to existing loss curves
3. Can overlay with test loss on same plot for immediate visual correlation
4. Should immediately show if coarseness increase correlates with grokking transition

**Quick Implementation**: Add it as a second y-axis on your existing training progress plot!

---

## Key Hypotheses to Test

1. **Blob neurons are necessary for grokking**: Grokking models develop high-coarseness neurons; non-grokking models don't
2. **Early blob emergence predicts grokking speed**: Models with more blob neurons at epoch 1000 grok faster
3. **Threshold effect**: There may be a minimum number/fraction of blob neurons needed for grokking
4. **p=101 is intrinsically harder**: Even seed=485 groks slower on p=101, suggesting this prime has some special property
5. **Seed×Prime interaction**: The combination of p=101 and seed=999 creates a pathological initialization that prevents blob formation

---

## Expected Outcomes

### For Grokking Models (e.g., p=109 seed=485, p=101 seed=485)
- Coarseness distribution shifts rightward over time
- Mean coarseness increases sharply during/before grokking transition
- Blob neuron count increases, especially around grokking time
- Early (epoch ~1000) coarseness predicts later grokking speed

### For Non-Grokking Model (p=101 seed=999)
- Coarseness distribution remains concentrated at low values (<0.5)
- Mean coarseness stays flat and low throughout training
- Blob neuron count remains near zero
- Outlier in early coarseness vs grokking time scatter plot

### Cross-Model Patterns
- Seed=485 consistently produces higher coarseness than seed=999 across all primes
- p=101 shows lower coarseness than other primes for both seeds
- Strong negative correlation between early coarseness and grokking time

---

## Additional Analyses (Future Work)

If the above analyses confirm the blob neuron hypothesis, consider:

1. **Coarseness vs Frequency Alignment**: Do high-coarseness neurons encode specific frequencies (e.g., 1, 2, 3)?
2. **Attention Head Analysis**: How do attention heads combine blob vs plaid neurons?
3. **Gradient Flow to Blob Neurons**: Do blob neurons receive larger gradients during grokking?
4. **Intervention Experiments**: What happens if you manually initialize neurons to have blob-like patterns?
5. **Theoretical Analysis**: Why does seed=999 + p=101 specifically prevent blob formation?

---

## Notes on p=101 Seed=999 Anomaly

**Key Observation**: This is the only model (among p=97, 101, 109, 113 with seeds 485, 999) that fails to grok.

**Possible Explanations**:
1. **Initialization far from blob-favorable regions**: Seed=999 initializes weights such that gradient descent can't reach blob neuron configurations for p=101 specifically
2. **Frequency mismatch**: The frequencies seed=999 naturally favors are particularly poorly suited for p=101's group structure
3. **Critical period failure**: There may be a critical early period where blob neurons must form; p=101 seed=999 misses this window
4. **Local minimum**: The combination creates a particularly deep local minimum in memorization that's hard to escape

**To Investigate**: 
- Compare epoch 0 Fourier spectra across all models - is p=101 seed=999's initialization qualitatively different?
- Analyze loss landscape curvature around initialization for different (prime, seed) pairs
- Track when first blob neurons appear in successful models vs when they should appear but don't in p=101 seed=999

---

## Implementation Tips

1. **Compute once, visualize many**: Calculate coarseness for all neurons at all checkpointed epochs, save to disk, then generate multiple visualizations from saved data
2. **Threshold tuning**: The blob/plaid boundary might not be exactly 0.7 - try multiple thresholds and see which gives cleanest separation
3. **Epoch sampling**: Don't need all 25,000 epochs for most visualizations - sample logarithmically or focus on key regions (early training, around grokking, late training)
4. **Comparison baseline**: Include theoretical coarseness for pure frequency components (cos, sin at different k) as reference

---

## Questions to Answer

1. Is there a quantitative threshold of coarseness (or blob neuron count) that predicts grokking?
2. Does coarseness increase precede, coincide with, or follow the test loss drop?
3. Are there "critical neurons" (specific indices) that must become blobs for grokking?
4. What is special about p=101 that makes it harder to grok?
5. Can we predict grokking failure from epoch 1000 coarseness alone?

---

*Document created: February 6, 2026*
*For: Modular Addition Grokking Research Project*
*Context: Analysis of blob vs plaid neuron patterns and their relationship to grokking dynamics*
