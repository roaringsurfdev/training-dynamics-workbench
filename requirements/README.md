# Requirements

This directory contains requirements for the Training Dynamics Workbench.

## Directory Structure

```
requirements/
├── README.md           # This file
├── active/             # Requirements currently being worked on
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

### Completed Milestones

| Version | Name | Date | Requirements |
|---------|------|------|--------------|
| [0.1.0](archive/v0.1.0-mvp/MILESTONE_SUMMARY.md) | MVP | 2026-02-01 | REQ_001 through REQ_010 |

## Working with Requirements

### Adding New Requirements

1. Create a new requirement file in `active/` using the naming convention:
   ```
   REQ_XXX_short_description.md
   ```

2. Use the template structure:
   - Problem Statement
   - Conditions of Satisfaction
   - Constraints
   - Context & Assumptions

3. Reference by number: "Work on REQ_011"

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
