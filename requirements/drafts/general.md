REQ_???
Parallelize analysis. Allow the analysis engine to process multiple epochs at a time. Not sure how planning GPU usage works here.

REQ_???
Add Attention Head visualizations (heat maps) to examine any relationships to neuron frequency specialization behavior. First visualizations should likely be taken from the original ModuloAdditionRefactored.py notebook (lines 131-146) as these are brought over from Nanda's original walkthrough notebook.

REQ_???
Sort variants by alpha in variant drop-down list on Analysis tab.

REQ_???
Add a default state to Select Variant drop-down that indicates no variant is selected. Currently, the default option in the drop-down is the first variant, but the variant doesn't load until the change event is fired. So if I want to see the first variant in the list, I need to choose a different variant and then choose the first variant in the list.

It would be ideal to tie this to the requirement to sort the variant list by alphanumeric.