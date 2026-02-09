# REQ_028: UI Improvements — Variant Dropdown

**Status:** Draft
**Priority:** Medium (quality-of-life improvement)
**Dependencies:** REQ_021d (Dashboard Integration)
**Last Updated:** 2026-02-08

## Problem Statement

Two usability issues with the variant dropdown on the Analysis tab:

### 1. Variants are unsorted

Variants in the dropdown appear in registry discovery order (filesystem order), not alphabetical order. With growing numbers of variants (currently 8 across 4 primes x 2 seeds), finding a specific variant requires scanning the entire list. Alphabetical sorting would let the researcher jump to a known variant predictably.

### 2. No default "unselected" state

When the page loads or a family is selected, the dropdown shows the first variant as the displayed value, but data doesn't load until the change event fires. This means:
- The UI appears to show a selected variant, but no data is loaded
- To view the first variant in the list, the researcher must select a different variant first, then select the first one back — triggering the change event

This is confusing and creates the false impression that the first variant is loaded when it isn't.

## Design

### Alphabetical Sorting

Modify `get_variant_choices()` in `dashboard/components/family_selector.py` to sort the returned choices list alphabetically by variant name (the second element of each tuple). The display name includes state indicators and parameters, but sorting should use the underlying variant name for stable ordering.

### Default "No Selection" State

Ensure the variant dropdown has no initial selection (`value=None`) and displays placeholder text indicating that selection is required. Gradio dropdowns support a `value=None` state that shows the label as placeholder.

The current code already sets `value=None` in `on_family_change()` and `refresh_variants()`. The issue is the initial page load: `initial_variant_choices` is pre-computed and passed to the dropdown constructor, but no explicit `value=None` is set. Verify that the dropdown constructor includes `value=None` to prevent Gradio from auto-selecting the first item.

## Scope

This requirement covers:
1. Alphabetical sorting of variant choices in `get_variant_choices()`
2. Explicit `value=None` on variant dropdown construction to prevent auto-selection
3. Verify that "no variant selected" state renders correctly (empty plots, appropriate status message)

This requirement does **not** cover:
- Changes to the family dropdown
- Changes to variant display format or state indicators
- Variant filtering or search

## Conditions of Satisfaction

- [ ] Variants in the dropdown are sorted alphabetically by variant name
- [ ] On page load, no variant is pre-selected (dropdown shows placeholder/label text)
- [ ] On family change, no variant is pre-selected
- [ ] Selecting the first variant in the sorted list loads its data correctly (change event fires)
- [ ] Existing variant selection behavior is otherwise unchanged

## Constraints

**Must have:**
- Sort by variant name (e.g., `modulo_addition_1layer_p101_seed485`), not by display name
- Explicit `value=None` to prevent auto-selection

**Must avoid:**
- Breaking the existing variant change handler or data loading flow
- Changing the variant display name format

**Flexible:**
- Whether sorting is ascending or descending (ascending is expected default)
