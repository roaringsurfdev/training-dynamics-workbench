# v0.6.0 — Cross-Epoch Analyzers

## Key Decision
PCA trajectory analysis was previously computed at render time in the dashboard (every epoch slider change recomputed PCA from raw snapshots). REQ_038 introduced a `CrossEpochAnalyzer` protocol and two-phase pipeline so this computation happens once during analysis and is stored as an artifact.

## What Changed
- **New protocol**: `CrossEpochAnalyzer` — runs after per-epoch analysis, consumes artifacts across all checkpoints
- **Pipeline Phase 2**: Executes cross-epoch analyzers after per-epoch phase, with dependency validation and skip-if-exists logic
- **ParameterTrajectoryPCA**: First cross-epoch analyzer — precomputes PCA projections + velocity for 4 component groups
- **Renderer refactor**: All 7 trajectory renderers accept precomputed data instead of raw snapshots
- **Dashboard v2**: Loads from `cross_epoch.npz` instead of computing PCA at render time

## Key Files
| File | Role |
|------|------|
| `analysis/protocols.py` | CrossEpochAnalyzer protocol definition |
| `analysis/pipeline.py` | Two-phase execution logic |
| `analysis/analyzers/parameter_trajectory_pca.py` | First cross-epoch analyzer |
| `analysis/artifact_loader.py` | load_cross_epoch / has_cross_epoch |
| `visualization/renderers/parameter_trajectory.py` | Refactored renderers |
| `tests/test_cross_epoch_analyzers.py` | 30 new tests |

## Opens for Future
- Phase transition detection as a cross-epoch analyzer
- Representational similarity analysis across training
- Per-input trace analysis (user's core research direction)
