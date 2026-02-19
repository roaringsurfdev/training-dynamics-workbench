"""REQ_042: Neuron Dynamics page.

Dedicated page showing per-neuron frequency trajectory over training,
frequency switch distribution, and commitment timeline. Reveals
subnetwork competition dynamics during grokking.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, dcc, html

from dashboard_v2.components.family_selector import get_family_choices, get_variant_choices
from dashboard_v2.layout import (
    _FLEX_WRAPPER_STYLE,
    _PAGE_CONTENT_STYLE,
    create_collapsed_page_sidebar,
    create_page_sidebar,
)
from dashboard_v2.state import get_registry, server_state
from miscope.visualization.renderers.neuron_freq_clusters import (
    render_commitment_timeline,
    render_neuron_freq_trajectory,
    render_switch_count_distribution,
)

# ---------------------------------------------------------------------------
# Plot IDs (prefixed "nd-" to avoid collisions)
# ---------------------------------------------------------------------------

_PLOT_IDS = [
    "nd-trajectory-plot",
    "nd-switch-plot",
    "nd-commitment-plot",
]

# ---------------------------------------------------------------------------
# Module-level cache for cross-epoch data
# ---------------------------------------------------------------------------

_cached_cross_epoch: dict | None = None
_cached_variant: str | None = None


def _empty_figure(message: str = "No data") -> go.Figure:
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
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


def _graph(graph_id: str, height: str = "400px") -> dcc.Graph:
    return dcc.Graph(
        id=graph_id,
        config={"displayModeBar": True},
        style={"height": height},
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def create_neuron_dynamics_layout(initial: dict | None = None) -> html.Div:
    """Create the Neuron Dynamics page layout with left sidebar."""
    initial = initial or {}

    sort_toggle = dbc.RadioItems(
        id="nd-sort-toggle",
        options=[
            {"label": "Natural Order", "value": "natural"},
            {"label": "Sorted by Final Frequency", "value": "sorted"},
        ],
        value="natural",
        className="mb-3",
    )

    sidebar = create_page_sidebar(
        prefix="nd-",
        initial_family=initial.get("family_name"),
        initial_variant=initial.get("variant_name"),
        extra_controls=[
            dbc.Label("Sort Order", className="fw-bold"),
            sort_toggle,
        ],
    )

    grid = html.Div(
        [
            # Trajectory heatmap (full width)
            dbc.Row(dbc.Col(_graph("nd-trajectory-plot", "600px"))),
            # Switch distribution | Commitment timeline
            dbc.Row(
                [
                    dbc.Col(_graph("nd-switch-plot", "350px"), width=6),
                    dbc.Col(_graph("nd-commitment-plot", "350px"), width=6),
                ]
            ),
        ],
    )

    return html.Div(
        [
            sidebar,
            create_collapsed_page_sidebar(),
            html.Div(grid, style=_PAGE_CONTENT_STYLE),
        ],
        style=_FLEX_WRAPPER_STYLE,
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register_neuron_dynamics_callbacks(app: Dash) -> None:
    """Register all callbacks for the Neuron Dynamics page."""

    @app.callback(
        Output("nd-family-dropdown", "options"),
        Input("nd-family-dropdown", "id"),
    )
    def populate_nd_families(_: str) -> list[dict]:
        registry = get_registry()
        choices = get_family_choices(registry)
        return [{"label": display, "value": name} for display, name in choices]

    @app.callback(
        Output("nd-variant-dropdown", "options"),
        Output("nd-variant-dropdown", "value"),
        Input("nd-family-dropdown", "value"),
        State("selection-store", "data"),
    )
    def on_nd_family_change(family_name: str | None, store_data: dict | None):
        if not family_name:
            return [], None
        registry = get_registry()
        choices = get_variant_choices(registry, family_name)
        options = [{"label": display, "value": name} for display, name in choices]
        stored = store_data or {}
        if stored.get("family_name") == family_name:
            variant_names = {opt["value"] for opt in options}
            if stored.get("variant_name") in variant_names:
                return options, stored["variant_name"]
        return options, None

    @app.callback(
        *[Output(pid, "figure") for pid in _PLOT_IDS],
        Output("nd-status", "children"),
        Input("nd-variant-dropdown", "value"),
        State("nd-family-dropdown", "value"),
        State("nd-sort-toggle", "value"),
    )
    def on_nd_variant_change(
        variant_name: str | None,
        family_name: str | None,
        sort_value: str,
    ):
        global _cached_cross_epoch, _cached_variant

        empty = _empty_figure("Select a variant")
        if not variant_name or not family_name:
            _cached_cross_epoch = None
            _cached_variant = None
            return empty, empty, empty, "No variant selected"

        loaded = server_state.load_variant(family_name, variant_name)
        if not loaded:
            err = _empty_figure("Failed to load variant")
            return err, err, err, "Load failed"

        loader = server_state.get_loader()
        if loader is None or not loader.has_cross_epoch("neuron_dynamics"):
            msg = "No neuron_dynamics data. Run the analysis pipeline first."
            no_data = _empty_figure(msg)
            return no_data, no_data, no_data, msg

        cross_epoch = loader.load_cross_epoch("neuron_dynamics")
        _cached_cross_epoch = cross_epoch
        _cached_variant = variant_name

        prime = server_state.model_config.get("prime", 101)  # type: ignore
        seed = server_state.model_config.get("seed")  # type: ignore
        sorted_flag = sort_value == "sorted"

        trajectory = render_neuron_freq_trajectory(cross_epoch, prime, sorted_by_final=sorted_flag)
        switch_dist = render_switch_count_distribution(cross_epoch, prime, seed=seed)
        commitment = render_commitment_timeline(cross_epoch, prime, seed=seed)

        n_neurons = cross_epoch["switch_counts"].shape[0]
        n_switchers = int((cross_epoch["switch_counts"] > 0).sum())
        status = f"{variant_name} — {n_switchers}/{n_neurons} neurons switch frequencies"

        return trajectory, switch_dist, commitment, status

    @app.callback(
        Output("nd-trajectory-plot", "figure", allow_duplicate=True),
        Input("nd-sort-toggle", "value"),
        prevent_initial_call=True,
    )
    def on_nd_sort_change(sort_value: str):
        if _cached_cross_epoch is None:
            return _empty_figure("Select a variant")

        prime = server_state.model_config.get("prime", 101)  # type: ignore
        sorted_flag = sort_value == "sorted"
        return render_neuron_freq_trajectory(
            _cached_cross_epoch, prime, sorted_by_final=sorted_flag
        )

    # --- Sync variant selection to cross-page store ---
    @app.callback(
        Output("selection-store", "data", allow_duplicate=True),
        Input("nd-variant-dropdown", "value"),
        State("nd-family-dropdown", "value"),
        prevent_initial_call=True,
    )
    def sync_nd_variant_to_store(variant: str | None, family: str | None):
        p = Patch()
        p["family_name"] = family
        p["variant_name"] = variant
        return p

    # --- Sync epoch slider to cross-page store ---
    @app.callback(
        Output("selection-store", "data", allow_duplicate=True),
        Input("nd-epoch-slider", "value"),
        prevent_initial_call=True,
    )
    def sync_nd_epoch_to_store(epoch_idx: int):
        epoch = server_state.get_epoch_at_index(epoch_idx)
        p = Patch()
        p["epoch"] = epoch
        return p
