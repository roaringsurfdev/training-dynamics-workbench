"""REQ_044/REQ_045: Representational Geometry page.

Shows how class manifold geometry evolves during training. Top section:
time-series of geometric measures (SNR, circularity, Fisher, etc.).
Bottom section: centroid PCA snapshot, distance heatmap, and Fisher
discriminant heatmap at a selected epoch.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from dashboard_v2.components.family_selector import get_family_choices, get_variant_choices
from dashboard_v2.state import get_registry, server_state
from miscope.visualization.renderers.repr_geometry import (
    render_centroid_distances,
    render_centroid_pca,
    render_fisher_heatmap,
    render_geometry_timeseries,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SITE_OPTIONS = [
    {"label": "All Sites", "value": "all"},
    {"label": "Post-Embed", "value": "resid_pre"},
    {"label": "Attn Out", "value": "attn_out"},
    {"label": "MLP Out", "value": "mlp_out"},
    {"label": "Resid Post", "value": "resid_post"},
]

_PLOT_IDS = [
    "rg-timeseries-plot",
    "rg-centroid-pca-plot",
    "rg-centroid-dist-plot",
    "rg-fisher-heatmap-plot",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def create_repr_geometry_layout() -> html.Div:
    """Create the Representational Geometry page layout."""
    controls = dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Family", className="fw-bold small"),
                    dcc.Dropdown(id="rg-family-dropdown", placeholder="Select family..."),
                ],
                width=2,
            ),
            dbc.Col(
                [
                    dbc.Label("Variant", className="fw-bold small"),
                    dcc.Dropdown(id="rg-variant-dropdown", placeholder="Select variant..."),
                ],
                width=3,
            ),
            dbc.Col(
                [
                    dbc.Label("Activation Site", className="fw-bold small"),
                    dcc.Dropdown(
                        id="rg-site-dropdown",
                        options=_SITE_OPTIONS,
                        value="all",
                        clearable=False,
                    ),
                ],
                width=2,
            ),
            dbc.Col(
                [
                    dbc.Label("Snapshot Epoch", className="fw-bold small"),
                    dcc.Slider(
                        id="rg-epoch-slider",
                        min=0,
                        max=1,
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ],
                width=3,
            ),
            dbc.Col(
                [
                    html.Div(
                        id="rg-status",
                        children="No variant selected",
                        className="text-muted small mt-4",
                    ),
                ],
                width=2,
            ),
        ],
        className="mb-3 align-items-end",
    )

    grid = html.Div(
        [
            # Time-series (full width, tall — auto-sized by renderer)
            dbc.Row(dbc.Col(_graph("rg-timeseries-plot", "1400px"))),
            # Fisher discriminant heatmap | Distance heatmap
            dbc.Row(
                [
                    dbc.Col(_graph("rg-fisher-heatmap-plot", "500px"), width=6),
                    dbc.Col(_graph("rg-centroid-dist-plot", "500px"), width=6),
                ]
            ),
            # Centroid PCA
            dbc.Row(
                [
                    dbc.Col(_graph("rg-centroid-pca-plot", "800px"), width=6),
                ]
            ),
        ],
    )

    return html.Div(
        [
            html.Div(
                [controls, grid],
                style={"padding": "20px", "overflowY": "auto", "height": "calc(100vh - 56px)"},
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def register_repr_geometry_callbacks(app: Dash) -> None:
    """Register all callbacks for the Representational Geometry page."""

    @app.callback(
        Output("rg-family-dropdown", "options"),
        Input("rg-family-dropdown", "id"),
    )
    def populate_rg_families(_: str) -> list[dict]:
        registry = get_registry()
        choices = get_family_choices(registry)
        return [{"label": display, "value": name} for display, name in choices]

    @app.callback(
        Output("rg-variant-dropdown", "options"),
        Output("rg-variant-dropdown", "value"),
        Input("rg-family-dropdown", "value"),
    )
    def on_rg_family_change(family_name: str | None):
        if not family_name:
            return [], None
        registry = get_registry()
        choices = get_variant_choices(registry, family_name)
        return [{"label": display, "value": name} for display, name in choices], None

    @app.callback(
        *[Output(pid, "figure") for pid in _PLOT_IDS],
        Output("rg-epoch-slider", "max"),
        Output("rg-epoch-slider", "value"),
        Output("rg-status", "children"),
        Input("rg-variant-dropdown", "value"),
        State("rg-family-dropdown", "value"),
        State("rg-site-dropdown", "value"),
    )
    def on_rg_variant_change(
        variant_name: str | None,
        family_name: str | None,
        site_value: str,
    ):
        empty = _empty_figure("Select a variant")
        if not variant_name or not family_name:
            return empty, empty, empty, empty, 1, 0, "No variant selected"

        loaded = server_state.load_variant(family_name, variant_name)
        if not loaded:
            err = _empty_figure("Failed to load variant")
            return err, err, err, err, 1, 0, "Load failed"

        loader = server_state.get_loader()
        if loader is None or "repr_geometry" not in server_state.available_analyzers:
            msg = "No repr_geometry data. Run the analysis pipeline first."
            no_data = _empty_figure(msg)
            return no_data, no_data, no_data, no_data, 1, 0, msg

        # Centroid snapshot at first epoch
        epochs = loader.get_epochs("repr_geometry")
        slider_max = max(len(epochs) - 1, 1)
        epoch = epochs[0] if epochs else 0

        # Time-series from summary (with epoch indicator)
        site = None if site_value == "all" else site_value
        timeseries = _render_timeseries(loader, site, current_epoch=epoch)
        prime = server_state.model_config.get("prime", 113) if server_state.model_config else 113
        snapshot_site = "resid_post" if site_value == "all" else site_value
        pca_fig, dist_fig, fisher_fig = _render_snapshot(loader, epoch, snapshot_site, prime)

        status = f"{variant_name} — {len(epochs)} epochs"
        return timeseries, pca_fig, dist_fig, fisher_fig, slider_max, 0, status

    @app.callback(
        Output("rg-timeseries-plot", "figure", allow_duplicate=True),
        Input("rg-site-dropdown", "value"),
        State("rg-epoch-slider", "value"),
        prevent_initial_call=True,
    )
    def on_rg_site_change(site_value: str, epoch_idx: int):
        loader = server_state.get_loader()
        if loader is None or "repr_geometry" not in server_state.available_analyzers:
            return _empty_figure("No data")
        site = None if site_value == "all" else site_value
        epochs = loader.get_epochs("repr_geometry")
        epoch = epochs[epoch_idx] if epochs and epoch_idx < len(epochs) else None
        return _render_timeseries(loader, site, current_epoch=epoch)

    @app.callback(
        *[Output(pid, "figure", allow_duplicate=True) for pid in _PLOT_IDS],
        Input("rg-epoch-slider", "value"),
        State("rg-site-dropdown", "value"),
        prevent_initial_call=True,
    )
    def on_rg_epoch_change(epoch_idx: int, site_value: str):
        loader = server_state.get_loader()
        if loader is None or "repr_geometry" not in server_state.available_analyzers:
            empty = _empty_figure("No data")
            return empty, empty, empty, empty

        epochs = loader.get_epochs("repr_geometry")
        if not epochs or epoch_idx >= len(epochs):
            empty = _empty_figure("Invalid epoch")
            return empty, empty, empty, empty

        epoch = epochs[epoch_idx]
        prime = server_state.model_config.get("prime", 113) if server_state.model_config else 113
        site = None if site_value == "all" else site_value
        snapshot_site = "resid_post" if site_value == "all" else site_value

        timeseries = _render_timeseries(loader, site, current_epoch=epoch)
        pca_fig, dist_fig, fisher_fig = _render_snapshot(loader, epoch, snapshot_site, prime)
        return timeseries, pca_fig, dist_fig, fisher_fig


def _render_timeseries(loader, site: str | None, current_epoch: int | None = None) -> go.Figure:
    """Render time-series from summary data."""
    if not loader.has_summary("repr_geometry"):
        return _empty_figure("No summary data. Run analysis pipeline first.")
    try:
        summary = loader.load_summary("repr_geometry")
        return render_geometry_timeseries(summary, site=site, current_epoch=current_epoch)
    except FileNotFoundError:
        return _empty_figure("No summary data")


def _render_snapshot(
    loader, epoch: int, site: str, prime: int
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Render centroid PCA, distance heatmap, and Fisher heatmap."""
    try:
        epoch_data = loader.load_epoch("repr_geometry", epoch)
        pca_fig = render_centroid_pca(epoch_data, epoch, site=site, p=prime)
        dist_fig = render_centroid_distances(epoch_data, epoch, site=site, p=prime)
        fisher_fig = render_fisher_heatmap(epoch_data, epoch, site=site, p=prime)
        return pca_fig, dist_fig, fisher_fig
    except FileNotFoundError:
        empty = _empty_figure(f"No data for epoch {epoch}")
        return empty, empty, empty
