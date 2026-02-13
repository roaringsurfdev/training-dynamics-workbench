REQ_??? - Support for large cross-epoch summary results

As a researcher, I may want to analyze models across epochs and that analysis might create artifacts that are large. An example of this is the PCA of the parameter trajectories across epochs for a given variant.

Without getting into too much detail on implementation, we might be able to store summary tensors in a summary subfolder under each analyzer. It may also be necessary to create an additional analyzer type that runs at the end of an analysis run to summarize epoch data.