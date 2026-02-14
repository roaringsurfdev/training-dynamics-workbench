# REQ_039: Source Layout Restructuring

**Status:** Ready for implementation
**Priority:** High — do before codebase grows further
**Branch:** `feature/src-layout-restructuring` (to be created at execution time)

## Problem Statement

The project has three core API packages (`analysis/`, `families/`, `visualization/`) sitting at the project root alongside their consumers (`dashboard/`, `dashboard_v2/`, `tests/`, `notebooks/`). There is no structural distinction between the public API and its consumers. The existing `tdw/` package serves as a notebook facade but doesn't contain the packages it wraps.

As the project grows toward contributor support, this flat layout creates:
- Ambiguity about what constitutes the installable API
- No namespace isolation (any root-level directory looks like a top-level package)
- Growing git blame cost if restructured later

## Solution

Move `analysis/`, `families/`, and `visualization/` under `src/tdw/`, establishing a standard Python src-layout. The `tdw` package becomes the single installable namespace containing the full API.

## Target Structure

```
project-root/
  src/tdw/                 # Installable API package (import tdw.*)
    __init__.py            # Public facade (load_family, list_families)
    config.py              # Path resolution
    loaded_family.py       # Notebook-friendly wrapper
    analysis/              # Pipeline, analyzers, artifacts, protocols
    families/              # Family registry, variants, model families
    visualization/         # Renderers, export
  dashboard/               # Consumer — Gradio (frozen, stays at root)
  dashboard_v2/            # Consumer — Dash (active, stays at root)
  tests/                   # Test suite (stays at root)
  notebooks/               # Research notebooks (stays at root)
  model_families/          # JSON config + data (stays at root)
  results/                 # Generated artifacts (stays at root)
```

## Conditions of Satisfaction

1. All 566+ tests pass after restructuring
2. `ruff check .` reports 0 errors
3. `pyright` reports 0 errors
4. `pip install -e .` succeeds and `import tdw.analysis`, `import tdw.families`, `import tdw.visualization` all work
5. Git blame is preserved via `git mv`
6. Dashboards (Gradio and Dash) both launch and function correctly
7. Notebooks using `from tdw import load_family` continue to work unchanged

## Constraints

- Single atomic commit for the restructure (no valid intermediate state)
- Feature branch workflow per CLAUDE.md
- Consumer packages (`dashboard/`, `dashboard_v2/`) stay at project root
- `model_families/` and `results/` are data directories, not Python packages — stay at root

---

## Detailed Analysis

### Import Inventory

All imports in the project are **absolute** (zero relative imports). This makes the restructuring a mechanical find-replace operation.

#### Internal Cross-Imports (within the 3 packages being moved)

| Source File | Import |
|-------------|--------|
| `families/implementations/modulo_addition_1layer.py` | `from analysis.library import get_fourier_basis` |
| `families/registry.py` | `from families.json_family import JsonModelFamily` + 2 more |
| `families/__init__.py` | 6 lines: `from families.*` |
| `visualization/export.py` | `from analysis.artifact_loader import ArtifactLoader` |
| `visualization/renderers/parameter_trajectory.py` | `from analysis.library.weights import COMPONENT_GROUPS` |
| `visualization/renderers/effective_dimensionality.py` | `from analysis.library.weights import ...` |
| `visualization/__init__.py` | 12 lines: `from visualization.renderers.*` |
| `analysis/__init__.py` | 4 lines: `from analysis.*` |
| `analysis/analyzers/__init__.py` | 11 lines: `from analysis.analyzers.*` |
| `analysis/analyzers/registry.py` | `from analysis.protocols import ...` |
| `analysis/analyzers/*.py` (8 files) | `from analysis.library.*` imports |
| `analysis/library/__init__.py` | 5 lines: `from analysis.library.*` |
| `analysis/library/trajectory.py` | `from analysis.library.weights import ...` |
| `analysis/pipeline.py` | `from analysis.protocols import ...` |

**Total:** ~60 internal import lines across ~28 files.

#### tdw Facade Imports

| File | Import |
|------|--------|
| `tdw/__init__.py` line 53 | `from families.registry import FamilyRegistry` (runtime, in `load_family()`) |
| `tdw/__init__.py` line 70 | `from families.registry import FamilyRegistry` (runtime, in `list_families()`) |
| `tdw/loaded_family.py` lines 7-10 | `from families.protocols`, `from families.registry`, `from families.types`, `from families.variant` |

**Note:** `from tdw.config` and `from tdw.loaded_family` (lines 28-29 of `__init__.py`) already use the `tdw.` prefix and need no change.

#### Consumer Imports — Dashboard (Gradio)

| File | Imports |
|------|---------|
| `dashboard/app.py` | `from analysis import ...` (2 lines), `from analysis.analyzers import ...` (1 line), `from analysis.library.weights import ...` (1 line), `from families import ...` (1 line), `from visualization import ...` (1 block ~15 names) |
| `dashboard/state.py` | `from analysis import ArtifactLoader` (runtime, line 95) |
| `dashboard/components/family_selector.py` | `from families import ...` (TYPE_CHECKING + 2 runtime) |

#### Consumer Imports — Dashboard v2 (Dash)

| File | Imports |
|------|---------|
| `dashboard_v2/state.py` | `from analysis import ArtifactLoader`, `from families import FamilyRegistry, Variant` |
| `dashboard_v2/callbacks.py` | `from analysis.library.weights import ATTENTION_MATRICES`, `from visualization.renderers.*` (multiple blocks) |
| `dashboard_v2/layout.py` | `from analysis.library.weights import WEIGHT_MATRIX_NAMES`, `from visualization.renderers.landscape_flatness import FLATNESS_METRICS` |

#### Consumer Imports — Tests (~16 files)

| File | Packages Imported |
|------|-------------------|
| `test_analysis_library.py` | `analysis` |
| `test_analysis_pipeline.py` | `analysis`, `families` |
| `test_artifact_loader.py` | `analysis` |
| `test_attention_freq_analyzer.py` | `analysis` |
| `test_attention_patterns_analyzer.py` | `analysis` |
| `test_coarseness_analyzer.py` | `analysis`, `families` |
| `test_cross_epoch_analyzers.py` | `analysis`, `families`, `visualization` |
| `test_dominant_frequencies_analyzer.py` | `analysis`, `families` |
| `test_effective_dimensionality.py` | `analysis`, `families`, `visualization` |
| `test_families.py` | `families` |
| `test_landscape_flatness.py` | `analysis`, `families`, `visualization` |
| `test_modulo_addition_family.py` | `analysis`, `families` |
| `test_neuron_freq_specialization.py` | `analysis` |
| `test_notebook_api.py` | `families` |
| `test_parameter_trajectory.py` | `analysis`, `families`, `visualization` |
| `test_visualization_export.py` | `visualization` |

**Total across all consumers:** ~90 import lines across ~25 files.

#### Root-Level Scripts (Legacy)

| File | Import |
|------|--------|
| `ModuloAdditionRefactored.py` | `from visualization import line` |
| `ModuloAdditionSpecification.py` | (no imports from core packages) |
| `FourierEvaluation.py` | (no imports from core packages) |

These are early MVP scaffolding files, candidates for future removal.

#### Notebooks

`notebooks/scratch_pad.py` uses `from tdw import load_family` — **no change needed**.

### Non-Code References

| File | Reference | Action |
|------|-----------|--------|
| `pyproject.toml` line 6 | `packages = ["tdw", "families", "analysis", "visualization", "dashboard", "dashboard_v2"]` | **CRITICAL** — change to `["src/tdw", "dashboard", "dashboard_v2"]` |
| `model_families/.../family.json` line 5 | `"class_type": "families.implementations..."` | Update to `tdw.families.implementations...` (metadata only — NOT dynamically loaded) |
| `CHANGELOG.md` | Architecture diagrams in ~8 sections | Update directory paths |
| `CLAUDE.md` | "Project Structure" section | Update directory layout |
| `src/tdw/__init__.py` | Docstring example `from visualization import` | Update to `from tdw.visualization import` |
| `.github/workflows/ci.yml` | Commands like `ruff check .`, `pytest` | No change needed (path-agnostic) |
| `.gitignore` | No references to moved packages | No change needed |

### Key Technical Detail: `tdw/config.py` Project Root Resolution

`_resolve_project_root()` in `tdw/config.py` (line 76):
```python
current = Path(__file__).resolve().parent.parent
```

After move to `src/tdw/config.py`, `.parent.parent` yields `src/` not project root. The walking loop on lines 77-78 handles this correctly (it checks all ancestors for `pyproject.toml`), but update to `.parent.parent.parent` for clarity and to avoid an unnecessary iteration.

### Key Technical Detail: Hatch src-layout

`packages = ["src/tdw"]` tells hatchling to find the `tdw` package inside `src/`. It automatically strips the `src/` prefix — the installed package is importable as `tdw`, not `src.tdw`.

Reference: [Hatch build configuration](https://hatch.pypa.io/1.13/config/build/)

---

## Implementation Plan

### Step 1: Create Feature Branch
```
git checkout develop && git checkout -b feature/src-layout-restructuring
```

### Step 2: Move Files with `git mv`
```
mkdir -p src/tdw
git mv tdw/__init__.py src/tdw/__init__.py
git mv tdw/config.py src/tdw/config.py
git mv tdw/loaded_family.py src/tdw/loaded_family.py
git mv analysis src/tdw/analysis
git mv families src/tdw/families
git mv visualization src/tdw/visualization
```

### Step 3: Rewrite All Imports

Apply 6 mechanical find-replace patterns to all `.py` files in `src/tdw/`, `dashboard/`, `dashboard_v2/`, `tests/`, and root-level scripts:

| Find | Replace |
|------|---------|
| `from analysis` | `from tdw.analysis` |
| `from families` | `from tdw.families` |
| `from visualization` | `from tdw.visualization` |
| `import analysis` | `import tdw.analysis` |
| `import families` | `import tdw.families` |
| `import visualization` | `import tdw.visualization` |

**Safety:** No existing `from tdw.analysis` imports exist. The only `tdw.*` imports (`from tdw.config`, `from tdw.loaded_family`) don't match these patterns.

**Scope exclusions:** `notebooks/` (already uses `from tdw import`).

### Step 4: Fix `config.py` Root Walk
Update `src/tdw/config.py` line 76: `.parent.parent` → `.parent.parent.parent`

### Step 5: Update `pyproject.toml`
```toml
packages = ["src/tdw", "dashboard", "dashboard_v2"]
```

### Step 6: Update `family.json` Metadata
Update `class_type` string to use `tdw.families.implementations...` prefix.

### Step 7: Reinstall
```
pip install -e .
```

### Step 8: Verify
1. Python smoke test: `from tdw.analysis import ArtifactLoader` etc.
2. `ruff check .` — 0 errors
3. `pyright` — 0 errors
4. `pytest` — all tests pass

### Step 9: Update Documentation
- `CLAUDE.md` project structure section
- `CHANGELOG.md` (restructuring note for next release)
- Docstrings in `src/tdw/__init__.py` and `src/tdw/visualization/__init__.py`

### Step 10: Commit
Single atomic commit with descriptive message. Request merge approval to `develop`.

## Risk Assessment

| Risk | Level | Mitigation |
|------|-------|------------|
| Missed import | Low | ruff catches unused/broken imports; full test suite validates |
| pyproject.toml misconfiguration | Medium | Smoke test immediately after `pip install -e .` |
| Config path resolution | Low | Walking loop in `_resolve_project_root()` handles any depth |
| No valid intermediate state | N/A | Single atomic commit; trivial revert |
