"""REQ_044/REQ_045: Representational Geometry page.

Shows how class manifold geometry evolves during training. Top section:
time-series of geometric measures (SNR, circularity, Fisher, etc.).
Bottom section: centroid PCA snapshot, distance heatmap, and Fisher
discriminant heatmap at a selected epoch.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, dcc, html, no_update

from dashboard_v2.components.family_selector import get_family_choices, get_variant_choices
from dashboard_v2.layout import (
    _FLEX_WRAPPER_STYLE,
    _PAGE_CONTENT_STYLE,
    create_collapsed_page_sidebar,
    create_page_sidebar,
)
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


def create_repr_geometry_layout(initial: dict | None = None) -> html.Div:
    """Create the Representational Geometry page layout with left sidebar."""
    initial = initial or {}

    site_dropdown = dcc.Dropdown(
        id="rg-site-dropdown",
        options=_SITE_OPTIONS,
        value="all",
        clearable=False,
    )

    sidebar = create_page_sidebar(
        prefix="rg-",
        initial_family=initial.get("family_name"),
        initial_variant=initial.get("variant_name"),
        extra_controls=[
            dbc.Label("Activation Site", className="fw-bold"),
            site_dropdown,
        ],
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
            sidebar,
            create_collapsed_page_sidebar(),
            html.Div(grid, style=_PAGE_CONTENT_STYLE),
        ],
        style=_FLEX_WRAPPER_STYLE,
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
        State("selection-store", "data"),
    )
    def on_rg_variant_change(
        variant_name: str | None,
        family_name: str | None,
        site_value: str,
        store_data: dict | None,
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

        epochs = loader.get_epochs("repr_geometry")
        slider_max = max(len(epochs) - 1, 1)

        stored_epoch = (store_data or {}).get("epoch")
        if stored_epoch is not None and epochs:
            initial_epoch_idx = min(range(len(epochs)), key=lambda i: abs(epochs[i] - stored_epoch))
        else:
            initial_epoch_idx = 0
        epoch = epochs[initial_epoch_idx] if epochs else 0

        site = None if site_value == "all" else site_value
        timeseries = _render_timeseries(loader, site, current_epoch=epoch)
        prime = server_state.model_config.get("prime", 113) if server_state.model_config else 113
        snapshot_site = "resid_post" if site_value == "all" else site_value
        pca_fig, dist_fig, fisher_fig = _render_snapshot(loader, epoch, snapshot_site, prime)

        status = f"{variant_name} — {len(epochs)} epochs"
        return timeseries, pca_fig, dist_fig, fisher_fig, slider_max, initial_epoch_idx, status

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

    # --- Click on timeseries → navigate epoch slider ---
    @app.callback(
        Output("rg-epoch-slider", "value", allow_duplicate=True),
        Input("rg-timeseries-plot", "clickData"),
        prevent_initial_call=True,
    )
    def on_rg_timeseries_click(click_data):
        if not click_data or not click_data.get("points"):
            return no_update
        clicked_epoch = click_data["points"][0].get("x")
        if clicked_epoch is None:
            return no_update
        loader = server_state.get_loader()
        if loader is None:
            return no_update
        epochs = loader.get_epochs("repr_geometry")
        if not epochs:
            return no_update
        return min(range(len(epochs)), key=lambda i: abs(epochs[i] - int(clicked_epoch)))

    # --- Sync variant selection to cross-page store ---
    @app.callback(
        Output("selection-store", "data", allow_duplicate=True),
        Input("rg-variant-dropdown", "value"),
        State("rg-family-dropdown", "value"),
        prevent_initial_call=True,
    )
    def sync_rg_variant_to_store(variant: str | None, family: str | None):
        p = Patch()
        p["family_name"] = family
        p["variant_name"] = variant
        return p

    # --- Sync epoch slider to cross-page store ---
    @app.callback(
        Output("selection-store", "data", allow_duplicate=True),
        Input("rg-epoch-slider", "value"),
        prevent_initial_call=True,
    )
    def sync_rg_epoch_to_store(epoch_idx: int):
        loader = server_state.get_loader()
        if loader is None:
            return no_update
        epochs = loader.get_epochs("repr_geometry")
        epoch = epochs[epoch_idx] if epochs and epoch_idx < len(epochs) else None
        if epoch is None:
            return no_update
        p = Patch()
        p["epoch"] = epoch
        return p


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
