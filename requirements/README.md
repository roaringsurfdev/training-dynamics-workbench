# Requirements

This directory contains requirements for the Training Dynamics Workbench.

## Directory Structure

```
requirements/
├── README.md           # This file
├── active/             # Requirements currently being worked on
├── future/             # Deferred/future requirements (not yet scheduled)
└── archive/            # Completed requirements organized by milestone
    └── v0.1.0-mvp/     # MVP milestone (released 2026-02-01)
```

## Current Status

**Latest Version:** 0.1.0 (MVP)

### Active Requirements

| Requirement | Description |
|-------------|-------------|
| [REQ_011](active/REQ_011_ruff_pyright_enforcement.md) | Make ruff/pyright CI checks blocking |
| [REQ_012](active/REQ_012_remove_neel_plotly.md) | Remove neel-plotly dependency |
| [REQ_013](active/REQ_013_python_311_compatibility.md) | Support Python 3.11+ |
| [REQ_016](active/REQ_016_cuda_compute.md) | CUDA support for analysis |
| [REQ_018](active/REQ_018_cuda_training.md) | CUDA support for training |

### Future Requirements

Requirements that have been documented but are not yet scheduled for implementation.

| Requirement | Description | Priority | Effort |
|-------------|-------------|----------|--------|
| [REQ_014](future/REQ_014_checkpoint_click_navigation.md) | Click-to-navigate checkpoint markers | Low | Medium |
| [REQ_015](future/REQ_015_checkpoint_editor.md) | Checkpoint editor using Train/Test loss curve | Medium | High |
| [REQ_017](future/REQ_017_multi_model_support.md) | Support for multiple toy models | High | High |
| [REQ_019](future/REQ_019_multiscale_activation_visualization.md) | Multi-scale activation visualization (downsampled) | Medium | Medium |

### Completed Milestones

| Version | Name | Date | Requirements |
|---------|------|------|--------------|
| [0.1.0](archive/v0.1.0-mvp/MILESTONE_SUMMARY.md) | MVP | 2026-02-01 | REQ_001 through REQ_010 |

## Working with Requirements

### Adding New Requirements

1. Create a new requirement file using the naming convention:
   ```
   REQ_XXX_short_description.md
   ```

2. Place in the appropriate directory:
   - `active/` - Requirements scheduled for near-term implementation
   - `future/` - Documented requirements not yet scheduled (deferred, exploratory)

3. Use the template structure:
   - Problem Statement
   - Conditions of Satisfaction
   - Constraints
   - Context & Assumptions

4. Reference by number: "Work on REQ_011"

### Completing a Milestone

When a set of requirements is complete:

1. Create archive directory: `archive/vX.Y.Z-name/`
2. Move completed requirements to archive
3. Create `MILESTONE_SUMMARY.md` in archive
4. Update `CHANGELOG.md` in project root
5. Bump version in `dashboard/version.py`

### Referencing Archived Requirements

For historical context on any requirement:
```
requirements/archive/v0.1.0-mvp/REQ_001_configurable_checkpoint_epochs.md
```

The milestone summary provides quick reference:
```
requirements/archive/v0.1.0-mvp/MILESTONE_SUMMARY.md
```

## Quick Reference

For current project capabilities, see:
- [CHANGELOG.md](../CHANGELOG.md) - Version history and features
- [README.md](../README.md) - Getting started guide
- [PROJECT.md](../PROJECT.md) - Project scope and vision
