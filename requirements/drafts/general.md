REQ_???
Model Family Summary Statistics

Some analysis types will require a single file per epoch due to the size of the analysis results.

Some analysis types will only create one datapoint per epoch and should be treated as summary statistics over the course of training.

Instead of including them in the metadata.json file that contains generic summary statistics, keeping a separate file for ModelFamily-specific summary statistics might be cleaner.

May need to differentiate analyzers between summary and per-epoch

REQ_???
Parallelize analysis. Allow the analysis engine to process multiple epochs at a time. Not sure how planning GPU usage works here.

REQ_???
Add Attention Head visualizations (heat maps) to examine any relationships to neuron frequency specialization behavior. First visualizations should likely be taken from the original ModuloAdditionRefactored.py notebook (lines 131-146) as these are brought over from Nanda's original walkthrough notebook.

REQ_???
Sort variants by alpha in variant drop-down list on Analysis tab.