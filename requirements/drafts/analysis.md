REQ_???
Priority: LOW. As the project grows, this might become more of a real issue, but for now, this is easy to address.

### Summary: Recompute summary statistics from existing artifacts
When an analyzer gains new `compute_summary()` / `get_summary_keys()` methods (e.g., REQ_027 added summary stats to `NeuronFreqClustersAnalyzer`), the pipeline skips epochs that already have artifact files. This means summary.npz is never generated for previously-analyzed variants.

**Workaround**: Delete the analyzer's artifact directory and re-run analysis.

**Desired behavior**: A "recompute summaries" mode that reads existing per-epoch `.npz` artifacts and calls `compute_summary()` on each, without re-running the full analysis (model loading, forward pass, etc.). This should be accessible from both the CLI and the dashboard "Run Analysis" button.

REQ_??? - Compute effective dimensionality
For each checkpoint, compute the Hessian and examine the eigenspectrum.

REQ_??? - Parameter Space Trajectory Projections
For each epoch, use PCA to project parameter vectors onto 2D/3D space. This might be better as a separate requirement, but I might want to look at UMAP, too.

REQ_??? - Compute the local loss landscape flatness
Calculate the flatness radius based on a random sample of how far you can move before loss increases significantly.