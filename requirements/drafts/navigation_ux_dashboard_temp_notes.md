Key points about dashboard_temp:
* Does not wire up any visualizations with data.
* Shows a proof-of-concept of behavior for empty vs populated visualizations.
* Refactors the layout for analysis pages
* Wires up proof-of-concept of expected behavior for navigating between analysis pages 
* Encapsulates the main Variant + Epoch selection logic in a single component
* Encapsulates leftnav logic into a single component, including adding support for page-specific navigation options
* Encapsulate sitenav logic into a single component

Still to be done:
* Encapsulate epoch-series logic (visualizations that allow selection of epoch and that render an epoch marker (vertical line or star) for the currently selected epoch)
* Evolve current callbacks do differentiate between graphs that have different update triggers (selected variant change only, selected epoch change, selected neuron changed)
