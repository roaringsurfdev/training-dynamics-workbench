## Roadmap Rough Draft
*This is a landing place for a high-level collection of future **infrastructure** work. This is not meant to store future research ideas. It is not in any particular order and it is not meant to be authorative. As items are completed, they may be removed from this document.*

### Asynchronous Job Support
As we add more Variants and Analyzers, it would be nice to kick of Analysis runs as asynchronous jobs so that (1) the user can continue using the application and (2) the user can return to the Analysis Job Manager to check on the status of runs.

Once this capability is created for Analysis, it would be ideal to add Asynchronous Job Support for Training models. A use case for kicking off batches of Training runs would be if we wanted to add a new random seed across multiple Variants.

*Note: This behavior, along with general job management support will likely be built in a separate front-end from the analysis. The Dash framework may be ideal for rendering visualizations, but it might not be the right fit for job management. For ease of use, it may be that links are provided between front-ends.*

### Support for Choosing Subset for Analysis Runs
The Representational Geometry Analyzer is incredibly powerful and useful. However, it is a long-running Analyzer compared to the others. We may have other Analyzers that are similarly long-running. This might create friction for creating and exploring new Variants, since all Analyzers must run in order to Analyze a Variant. It would be nice to be able to choose which Analyzers to run.

### Add support for Loading and Editing Variant Checkpoint Data
The current Training interface assumes I'm only creating new variants. I would like to be able edit an existing variant's checkpoints - for example, adding checkpoint for models whose grokking window falls outside the default grokking window. It would also be nice to be able to delete unecessary checkpoints and their artifacts through a single interface to make sure everything remains in sync. This requirement is coming from a need to rerun 107/485 with additional checkpoints. There were mistakes made on my end, and there's no way to gracefully recover without deleting and retraining/reanalyzing the entire model.

### Adding checkpoints should not need to retrain the whole model
If checkpoints are added to a model, the platform should be able to pick up with the last checkpoint before the newly added ones. If inserting checkpoints (adding density), it should only be necessary to train the model between known good checkpoints.