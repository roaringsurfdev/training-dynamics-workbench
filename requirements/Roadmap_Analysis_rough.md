## Roadmap - Analysis Rough Draft
*This is a landing place for a high-level collection of future **analysis** work. This is not meant to store future infrastructure ideas. It is not in any particular order and it is not meant to be authorative. As items are completed, they may be removed from this document.*

**Animations**
At a high level, many of my requests are coming from the motivation to build animations for points of interest over training. I may also want to the ability to have a job queue for creating animations.

**Dominant Frequencies - All Weights**
We currently have an Analyzer called "Dominant Frequencies" that provides analysis on embedding frequencies. I would like to create a new analyzer that provides data on all weight spaces. We can leave the existing Analyzer as-is for now.

**Parameter Space PCA per epoch**
We currently show Parameter Space PCA for the entire training run. I'm wondering if it would make sense to create visualizations per epoch for side-by-side comparison with the Representation Space Centroid PCA diagrams.

**Centroid Class PCA Distribution**
The PCA redistributes across PC 1-3 over training. It would be nice to see this in a histogram or similar chart.

**Visibility into Neurons that Thrash**
It would be great to have a list of neurons that hop frequency specialization along with their respective concentrations of specialization.

### Notes
- Need to re-run models with dense checkpoint schedules around test loss curves where test loss curves are outside the standard set by 113/999

- Hypothesis: The model has achieved at least one circle by the end of training loss and this might be the first bottom of the curve I'm seeing in the Parameter Trajectory PCA. Work after this is about tuning the circle, and the final dip is when the model has reached optimal tuning.

-- The models seem to be moving through Lissajous curves. Is the Parameter Trajectory PCA plot showing that loop closing in PC2vPC3? How do we calculate the ratio? "The Lissajous figure closes when the frequency ratio is rational. It stays open, forever tracing without repeating, when it isn't."

-- Would a different seed on the training data help 101/999 or 59/485?
-- What interventions might help outlier models?