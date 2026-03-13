"""Intervention Hook Verification page (REQ_070).

Shows per-frequency amplitude of hook_attn_out — baseline vs. hook-modified —
for a selected intervention variant and checkpoint epoch.  Lets the user
confirm that the FrequencyGainHook is applying the intended gain to the
intended frequencies before running further experiments.

This page maintains its own local state (separate from the global
variant-selector-store) because it needs to select intervention variants
specifically and runs model forward passes rather than reading artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, set_props
from dash.exceptions import PreventUpdate

from dashboard.state import get_registry
from miscope.families.implementations.frequency_gain_hook import compute_hook_verification
from miscope.visualization.renderers.intervention_check import render_hook_verification_chart

_INTERVENTION_FAMILY = "modadd_intervention"


# ---------------------------------------------------------------------------
# Page-local server state
# ---------------------------------------------------------------------------


@dataclass
class _InterventionCheckState:
    """Holds the last computed verification result to avoid redundant recomputes."""

    variant_name: str | None = None
    epoch: int | None = None
    last_result: dict[str, Any] | None = None
    available_checkpoints: list[int] = field(default_factory=list)


_state = _InterventionCheckState()


def _get_intervention_variant_options() -> list[dict]:
    """Return dropdown options for all variants in the intervention family."""
    try:
        registry = get_registry()
        family = registry.get_family(_INTERVENTION_FAMILY)
        variants = registry.get_variants(family)
    except KeyError:
        return []

    options = []
    for v in sorted(variants, key=lambda x: x.name):
        # Try to surface the label from intervention config for readability
        try:
            cfg = v.model_config
            iv = cfg.get("intervention", {})
            label = iv.get("label", v.name)
        except Exception:
            label = v.name
        options.append({"label": label, "value": v.name})
    return options


def _load_variant_checkpoints(variant_name: str) -> list[int]:
    """Return sorted checkpoint epochs for the given variant."""
    try:
        registry = get_registry()
        family = registry.get_family(_INTERVENTION_FAMILY)
        variants = registry.get_variants(family)
        for v in variants:
            if v.name == variant_name:
                return v.get_available_checkpoints()
    except Exception:
        pass
    return []


def _compute_and_cache(variant_name: str, epoch: int) -> dict[str, Any] | None:
    """Run compute_hook_verification if variant/epoch changed; return cached result otherwise."""
    if _state.variant_name == variant_name and _state.epoch == epoch and _state.last_result is not None:
        return _state.last_result

    try:
        registry = get_registry()
        family = registry.get_family(_INTERVENTION_FAMILY)
        variants = registry.get_variants(family)
        variant = next((v for v in variants if v.name == variant_name), None)
        if variant is None:
            return None
        result = compute_hook_verification(variant, epoch, device="cpu")
    except Exception as exc:
        print(f"intervention_check: compute failed for {variant_name} epoch {epoch}: {exc}")
        return None

    _state.variant_name = variant_name
    _state.epoch = epoch
    _state.last_result = result
    return result


def _empty_figure(message: str = "Select a variant") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        template="plotly_white",
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Layout & nav
# ---------------------------------------------------------------------------


def create_intervention_check_page_nav() -> html.Div:
    return html.Div(
        children=[
            dcc.Store(
                id="intervention-check-store",
                storage_type="memory",
                data={"variant_name": None, "epoch_index": 0, "max_index": 0},
            ),
            dbc.Label("Intervention Variant", className="fw-bold"),
            dcc.Dropdown(
                id="intervention-check-variant-dropdown",
                placeholder="Select intervention variant...",
                options=_get_intervention_variant_options(),
            ),
            html.Br(),
            dbc.Label("Epoch", className="fw-bold"),
            html.Div(
                [
                    dbc.Button(
                        "<",
                        id="intervention-check-epoch-prev",
                        size="sm",
                        color="secondary",
                        outline=True,
                        n_clicks=0,
                        style={"flexShrink": "0"},
                    ),
                    dcc.Slider(
                        id="intervention-check-epoch-slider",
                        min=0,
                        max=0,
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    dbc.Button(
                        ">",
                        id="intervention-check-epoch-next",
                        size="sm",
                        color="secondary",
                        outline=True,
                        n_clicks=0,
                        style={"flexShrink": "0"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "4px"},
            ),
            html.Div(
                id="intervention-check-epoch-display",
                children="Epoch —",
                className="text-muted small mb-3",
            ),
            html.Hr(),
            html.Div(
                id="intervention-check-status",
                children="No variant selected",
                className="text-muted small",
            ),
        ]
    )


def create_intervention_check_page_layout() -> html.Div:
    return html.Div(
        children=[
            html.H4("Intervention Hook Verification", className="mb-3"),
            html.P(
                "Verify that the FrequencyGainHook applies the intended gain "
                "to the targeted frequencies in hook_attn_out at each checkpoint epoch.",
                className="text-muted",
            ),
            dbc.Row(
                dbc.Col(
                    dcc.Graph(
                        id="intervention-check-figure",
                        config={"displayModeBar": True},
                        style={"height": "500px"},
                        figure=_empty_figure(),
                    )
                )
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register_intervention_check_callbacks(app: Dash) -> None:
    """Register all callbacks for the Intervention Check page."""

    @app.callback(
        Output("intervention-check-epoch-slider", "max"),
        Output("intervention-check-epoch-slider", "value"),
        Output("intervention-check-status", "children"),
        Input("intervention-check-variant-dropdown", "value"),
        prevent_initial_call=True,
    )
    def on_variant_selected(variant_name: str | None):
        if variant_name is None:
            raise PreventUpdate

        checkpoints = _load_variant_checkpoints(variant_name)
        _state.available_checkpoints = checkpoints

        if not checkpoints:
            return 0, 0, f"{variant_name} — no checkpoints found"

        max_index = len(checkpoints) - 1
        set_props(
            "intervention-check-store",
            {"data": {"variant_name": variant_name, "epoch_index": 0, "max_index": max_index}},
        )
        return max_index, 0, f"{variant_name} — {len(checkpoints)} checkpoints"

    @app.callback(
        Output("intervention-check-epoch-display", "children"),
        Output("intervention-check-figure", "figure"),
        Input("intervention-check-epoch-slider", "value"),
        State("intervention-check-store", "data"),
        prevent_initial_call=True,
    )
    def on_epoch_changed(epoch_index: int, store_data: dict | None):
        stored = store_data or {}
        variant_name = stored.get("variant_name")

        if not variant_name or not _state.available_checkpoints:
            raise PreventUpdate

        epoch_index = max(0, min(epoch_index, len(_state.available_checkpoints) - 1))
        epoch = _state.available_checkpoints[epoch_index]

        set_props(
            "intervention-check-store",
            {
                "data": {
                    **stored,
                    "epoch_index": epoch_index,
                }
            },
        )

        result = _compute_and_cache(variant_name, epoch)
        if result is None:
            return f"Epoch {epoch}", _empty_figure(f"Failed to compute for epoch {epoch}")

        fig = render_hook_verification_chart(result)
        return f"Epoch {epoch}", fig

    @app.callback(
        Input("intervention-check-epoch-prev", "n_clicks"),
        Input("intervention-check-epoch-next", "n_clicks"),
        State("intervention-check-store", "data"),
        prevent_initial_call=True,
    )
    def on_epoch_nav(
        _prev: int, _next: int, store_data: dict | None
    ) -> None:
        from dash import ctx

        stored = store_data or {}
        current_index = stored.get("epoch_index", 0)
        max_index = stored.get("max_index", 0)

        if ctx.triggered_id == "intervention-check-epoch-prev":
            new_index = max(0, current_index - 1)
        else:
            new_index = min(max_index, current_index + 1)

        if new_index != current_index:
            set_props("intervention-check-epoch-slider", {"value": new_index})
        else:
            raise PreventUpdate
