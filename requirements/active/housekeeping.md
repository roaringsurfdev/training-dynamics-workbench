### Housekeeping tasks and requirements
*This is a space for tracking tasks and requirements that help keep the codebase clean and healthy*

#### Scheduled for the next Release
The following requirements are completed on `feature/architecture-adapter` and ready for inclusion in the next Release:

* REQ_105_architecture_adapter — HookedModel base class + canonical hook/weight vocabulary
* REQ_112_hooked_transformer — HookedTransformer subclass + canary analyzer migration (byte-identical to develop, 6054/6054 artifacts)
* REQ_113_hooked_mlp — HookedOneHotMLP + HookedEmbeddingMLP subclasses
* REQ_114_analyzer_migration — bulk analyzer migration onto HookedModel; bundles retired; quarantine load-bearing (byte-identical to develop, 3027/3027 artifacts on canon)
