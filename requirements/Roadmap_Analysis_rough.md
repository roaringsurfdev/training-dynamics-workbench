## Roadmap - Analysis Rough Draft
*This is a landing place for a high-level collection of future **analysis** work. This is not meant to store future infrastructure ideas. It is not in any particular order and it is not meant to be authorative. As items are completed, they may be removed from this document.*

**Analysis Catalog**
With the addition of the view/catalog framework in the middle tier (miscope), it may be time to start formalizing and organizing the existing analysis. To support future use on new models (and to keep track of analysis on existing models), it would be useful to track the following fields for each view:
Name:
Category: [Summary, Parameter Space, Activation Space, Learned Representation] - can be multiple
Weight/Activation Spaces: [What portions of the network are being analyzed]
Purpose: [What does this analysis attempt to address]
Sources/Citations: [Where was this analysis used]
Math/Computation: [Any associated math, with formulas]
Description: [If someone else were to use the platform, this would allow them to gain a better understanding of this technique to determine whether they want to include it for a model or on a dashboard]
Implementation Details: [Completion status, Generic Analyzer->Renderer map, Possible Filters]

*Note: A markdown document (under articles/techniques) was created by Research Claude based on 10 techniques they are aware of and that are presented in a paper included in articles/releated_research*

**Animations**
At a high level, many of my requests are coming from the motivation to build animations for points of interest over training. I may also want to the ability to have a job queue for creating animations.

*Note: I can currently create animations using a purpose-built notebook.*

**Dominant Frequencies - All Weights**
We currently have an Analyzer called "Dominant Frequencies" that provides analysis on embedding frequencies. I would like to create a new analyzer that provides data on all weight spaces. At the very least, this should be expanded to include Output weights. We can leave the existing Analyzer as-is for now and create a new one if that's best. *Note: This is currently in progress*

**Cross-Variant Analysis**
Going to start needing some cross-variant comparison. This seems like it might be easiest for time/epoch series visualizations.

There are at least two ways of solving this, and both might be needed:
* Choose from list of variants for a model family to show on single epoch-series graph
* Side-by-side selection comparison where 2-4 models are selected, epoch-series data is combined where possible, then side-by-side visualizations where not. This would follow a product-comparison model selection UX.

**Mean IPR Analysis**
Add Mean IPR Analysis from notbooks/neuron_fourier_poc.py to analysis pipeline and renders.

**Fourier Frequency Prediction**
I would like to be able to evaluate how the top frequencies impact model learning over training - both in terms of frequency changes and initial frequency dominance. One approach: Take the perfect mod-p addition table (as a p × p × p tensor of logits), compute its full Fourier decomposition, then project down to only the components corresponding to your k frequencies. The reconstruction accuracy tells you how much of the "signal" those frequencies can capture.
This might allow:
Score the initial dominant frequencies at early checkpoints — do models that happen to initialize with high-quality frequency subsets converge faster or more reliably?
Track the quality score over training — when a model starts with a poor subset, can you see it trying to swap frequencies, and does it succeed?

**Support for Probe Traces**
I would like to be able to start evaluating probes through the network based on equivalence classes and residues. Coupled with Fourier Frequency Prediction, I'd like to be able to track failure modes across the network to see what it looks like when a class is poorly separated in the representation geometry. What are the interactions between the Attention Heads and the MLPs?

**Error predictions based on learned frequencies**
It would be great to be able to compare model accuracy within equivalence classes within the context of information about the quality of frequencies selected by the model and errors in output. This could likely be compared with the Fisher min.

**Per-frequency gradient energy per-checkpoint**
To be able to watch the accumulation trajectory from epoch 0 to cascade and see whether the winning frequencies had higher gradient energy from the start or whether they diverged later.

**Experiments for testing periodicity as natural partitioning mechanism for neural networks**
p109/s485/ds598 shows geometric structure as early as epoch 500, which is during first descent and while test loss is still climbing. This really pushes against the theory that geometric formation is what the model is learning during generalization.
p101/s999/ds598 shows poor organization *and* poor MLP/Resid_Post Fourier Alignment throughout training.
Many models show early MLP/Resid_Post Fourier Alignment - as early as first descent.
*If* the models are naturally partitioning the neurons using frequency alignment (and generalization is about coordination between neural populations), we might expect p101 models to perform better by changing the width of the 1-Layer to a size more friendly to carving up neural populations to represent 100 classes.

We might create a new model family that adds M as one of the domain parameters. Early signal that a model width trains well for the given prime would be early Fourier MLP alignment, although p101/s485/ds42 and p101/s999/ds999 challenge this thinking.

Another question might be: how early do neurons show periodicity within their activations?
Another question might be: does Fourier Alignment above threshold predict first mover emergence?
