"""Family and variant selection components for the dashboard.

REQ_021d: Dashboard Integration with Model Families
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from families import FamilyRegistry, ModelFamily, Variant


def get_family_choices(registry: FamilyRegistry) -> list[tuple[str, str]]:
    """Get choices for family dropdown.

    Args:
        registry: The FamilyRegistry instance

    Returns:
        List of (display_name, family_name) tuples for gr.Dropdown
    """
    families = registry.list_families()
    return [(f.display_name, f.name) for f in families]


def get_variant_choices(
    registry: FamilyRegistry, family_name: str | None
) -> list[tuple[str, str]]:
    """Get choices for variant dropdown.

    Args:
        registry: The FamilyRegistry instance
        family_name: Name of the selected family

    Returns:
        List of (display_name, variant_name) tuples for gr.Dropdown
    """
    if not family_name or family_name not in registry:
        return []

    family = registry.get_family(family_name)
    variants = registry.get_variants(family)

    choices = []
    for variant in variants:
        state_indicator = get_state_indicator(variant)
        params_str = format_variant_params(variant)
        display_name = f"{state_indicator} {params_str} [{variant.state.value}]"
        choices.append((display_name, variant.name))

    return choices


def get_state_indicator(variant: Variant) -> str:
    """Get visual indicator for variant state.

    Args:
        variant: The Variant instance

    Returns:
        Unicode character indicating state
    """
    from families import VariantState

    indicators = {
        VariantState.UNTRAINED: "○",  # Empty circle
        VariantState.TRAINED: "●",  # Filled circle
        VariantState.ANALYZED: "◉",  # Circle with dot (analyzed)
    }
    return indicators.get(variant.state, "?")


def format_variant_params(variant: Variant) -> str:
    """Format variant parameters for display.

    Args:
        variant: The Variant instance

    Returns:
        Formatted parameter string like "p=113, seed=42"
    """
    parts = []
    for key, value in variant.params.items():
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def get_variant_table_data(
    registry: FamilyRegistry, family_name: str | None
) -> list[list[str]]:
    """Get data for variant table display.

    Args:
        registry: The FamilyRegistry instance
        family_name: Name of the selected family

    Returns:
        List of rows: [[parameters, state, actions], ...]
    """
    if not family_name or family_name not in registry:
        return []

    family = registry.get_family(family_name)
    variants = registry.get_variants(family)

    rows = []
    for variant in variants:
        state_indicator = get_state_indicator(variant)
        params_str = format_variant_params(variant)
        state_str = f"{state_indicator} {variant.state.value.capitalize()}"
        actions = get_available_actions(variant)

        rows.append([params_str, state_str, actions, variant.name])

    return rows


def get_available_actions(variant: Variant) -> str:
    """Get comma-separated list of available actions for a variant.

    Args:
        variant: The Variant instance

    Returns:
        String describing available actions
    """
    from families import VariantState

    action_map = {
        VariantState.UNTRAINED: "Train",
        VariantState.TRAINED: "Analyze",
        VariantState.ANALYZED: "View, Re-analyze",
    }
    return action_map.get(variant.state, "")
