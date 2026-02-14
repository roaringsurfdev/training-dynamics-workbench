"""Callback registrations for the Dash dashboard.

Organized by interaction pattern:
- Variant selection: loads data, renders all plots
- Epoch change (summary): Patch() marker updates only
- Epoch change (per-epoch): full re-render of detail plots
- Click-to-navigate: summary plot click → epoch slider update
- Control-specific: neuron slider, flatness metric
- Sidebar toggle
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, no_update

from dashboard.components.family_selector import get_family_choices, get_variant_choices
from dashboard.components.loss_curves import render_loss_curves_with_indicator
from dashboard_v2.state import get_registry, server_state
from visualization.renderers.dominant_frequencies import render_dominant_frequencies
from visualization.renderers.landscape_flatness import render_flatness_trajectory
from visualization.renderers.neuron_activations import render_neuron_heatmap
from visualization.renderers.neuron_freq_clusters import (
    render_freq_clusters,
    render_specialization_trajectory,
)


def _empty_figure(message: str = "No data") -> go.Figure:
    """Create a placeholder figure with a centered message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
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


# ---------------------------------------------------------------------------
# Summary figure cache
# ---------------------------------------------------------------------------
# Keyed by (variant_name, plot_name). Stores base figures so Patch() can
# update just the epoch marker without full re-render.
# _figure_epoch tracks what epoch each cached figure currently shows,
# used to skip redundant Patch() calls that race with full renders.

_figure_cache: dict[str, go.Figure] = {}
_figure_epoch: dict[str, int] = {}


def _cache_key(plot_name: str) -> str:
    """Build cache key from current variant + plot name."""
    variant = server_state.variant
    vname = variant.name if variant else "none"
    return f"{vname}:{plot_name}"


def _render_and_cache_loss(epoch: int) -> go.Figure:
    """Render loss curves and cache the figure."""
    fig = render_loss_curves_with_indicator(
        server_state.train_losses,
        server_state.test_losses,
        current_epoch=epoch,
        checkpoint_epochs=server_state.available_epochs,
    )
    key = _cache_key("loss")
    _figure_cache[key] = fig
    _figure_epoch[key] = epoch
    return fig


def _render_and_cache_spec_trajectory(epoch: int) -> go.Figure:
    """Render specialization trajectory and cache."""
    loader = server_state.get_loader()
    if loader is None or "neuron_freq_norm" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    if not loader.has_summary("neuron_freq_norm"):
        return _empty_figure("No summary data")
    try:
        summary_data = loader.load_summary("neuron_freq_norm")
        fig = render_specialization_trajectory(summary_data, current_epoch=epoch)
        key = _cache_key("spec_traj")
        _figure_cache[key] = fig
        _figure_epoch[key] = epoch
        return fig
    except FileNotFoundError:
        return _empty_figure("No summary data")


def _render_and_cache_flatness(epoch: int, metric: str) -> go.Figure:
    """Render flatness trajectory and cache."""
    loader = server_state.get_loader()
    if loader is None or "landscape_flatness" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    if not loader.has_summary("landscape_flatness"):
        return _empty_figure("No summary data")
    try:
        summary_data = loader.load_summary("landscape_flatness")
        fig = render_flatness_trajectory(summary_data, current_epoch=epoch, metric=metric)
        key = _cache_key("flatness")
        _figure_cache[key] = fig
        _figure_epoch[key] = epoch
        return fig
    except FileNotFoundError:
        return _empty_figure("No summary data")


def _patch_epoch_marker(epoch: int) -> Patch:
    """Create a Patch that updates the vline epoch marker.

    All summary renderers add the vline as the last shape via add_vline(),
    which creates both a shape (in layout.shapes) and an annotation
    (in layout.annotations). We update both.
    """
    patched = Patch()
    patched["layout"]["shapes"][-1]["x0"] = epoch
    patched["layout"]["shapes"][-1]["x1"] = epoch
    patched["layout"]["annotations"][-1]["x"] = epoch
    patched["layout"]["annotations"][-1]["text"] = f"Epoch {epoch}"
    return patched


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------


def register_callbacks(app: Dash) -> None:
    """Register all callbacks on the Dash app."""

    # --- Populate family dropdown on load ---
    @app.callback(
        Output("family-dropdown", "options"),
        Input("family-dropdown", "id"),  # fires once on load
    )
    def populate_families(_: str) -> list[dict]:
        registry = get_registry()
        choices = get_family_choices(registry)
        return [{"label": display, "value": name} for display, name in choices]

    # --- Family change → update variant dropdown ---
    @app.callback(
        Output("variant-dropdown", "options"),
        Output("variant-dropdown", "value"),
        Input("family-dropdown", "value"),
    )
    def on_family_change(family_name: str | None):
        if not family_name:
            return [], None
        registry = get_registry()
        choices = get_variant_choices(registry, family_name)
        return [{"label": display, "value": name} for display, name in choices], None

    # --- Variant change → load data, render all plots ---
    @app.callback(
        Output("loss-plot", "figure"),
        Output("spec-trajectory-plot", "figure"),
        Output("flatness-trajectory-plot", "figure"),
        Output("freq-plot", "figure"),
        Output("activation-plot", "figure"),
        Output("clusters-plot", "figure"),
        Output("epoch-slider", "max"),
        Output("epoch-slider", "value"),
        Output("neuron-slider", "max"),
        Output("epoch-display", "children"),
        Output("neuron-display", "children"),
        Output("status-display", "children"),
        Input("variant-dropdown", "value"),
        State("family-dropdown", "value"),
        State("flatness-metric-dropdown", "value"),
    )
    def on_variant_change(
        variant_name: str | None,
        family_name: str | None,
        flatness_metric: str,
    ):
        empty = _empty_figure("Select a variant")
        if not variant_name or not family_name:
            server_state.clear()
            return (
                empty, empty, empty, empty, empty, empty,
                1, 0, 511,
                "Epoch 0 (Index 0)", "Neuron 0", "No variant selected",
            )

        if not server_state.load_variant(family_name, variant_name):
            server_state.clear()
            return (
                _empty_figure("Variant not found"),
                *[empty] * 5,
                1, 0, 511,
                "Epoch 0 (Index 0)", "Neuron 0", "Variant not found",
            )

        epoch = server_state.get_epoch_at_index(0)
        max_idx = max(0, len(server_state.available_epochs) - 1)

        # Render and cache summary plots
        loss_fig = _render_and_cache_loss(epoch)
        spec_traj_fig = _render_and_cache_spec_trajectory(epoch)
        flatness_fig = _render_and_cache_flatness(epoch, flatness_metric)

        # Render per-epoch plots
        freq_fig = _render_per_epoch_freq(epoch)
        activation_fig = _render_per_epoch_activation(epoch, 0)
        clusters_fig = _render_per_epoch_clusters(epoch)

        # Status (variant is guaranteed non-None after successful load_variant)
        variant = server_state.variant
        assert variant is not None
        status_parts = [f"Variant: {variant.name}"]
        status_parts.append(f"State: {variant.state.value}")
        if server_state.available_epochs:
            status_parts.append(f"{len(server_state.available_epochs)} checkpoints")
        status = " | ".join(status_parts)

        return (
            loss_fig, spec_traj_fig, flatness_fig,
            freq_fig, activation_fig, clusters_fig,
            max_idx, 0, max(0, server_state.n_neurons - 1),
            f"Epoch {epoch} (Index 0)", "Neuron 0", status,
        )

    # --- Epoch change → Patch summary markers + re-render per-epoch ---
    @app.callback(
        Output("loss-plot", "figure", allow_duplicate=True),
        Output("spec-trajectory-plot", "figure", allow_duplicate=True),
        Output("flatness-trajectory-plot", "figure", allow_duplicate=True),
        Output("freq-plot", "figure", allow_duplicate=True),
        Output("activation-plot", "figure", allow_duplicate=True),
        Output("clusters-plot", "figure", allow_duplicate=True),
        Output("epoch-display", "children", allow_duplicate=True),
        Input("epoch-slider", "value"),
        State("neuron-slider", "value"),
        prevent_initial_call=True,
    )
    def on_epoch_change(epoch_idx: int, neuron_idx: int):
        epoch = server_state.get_epoch_at_index(epoch_idx)

        # Patch summary plots (no full re-render).
        # If the cached figure already shows this epoch (e.g. variant just loaded),
        # return no_update to avoid a Patch racing against the full figure render.
        loss_key = _cache_key("loss")
        spec_key = _cache_key("spec_traj")
        flat_key = _cache_key("flatness")

        if loss_key in _figure_cache:
            if _figure_epoch.get(loss_key) == epoch:
                loss_patch = no_update
            else:
                loss_patch = _patch_epoch_marker(epoch)
                _figure_epoch[loss_key] = epoch
        else:
            loss_patch = _render_and_cache_loss(epoch)

        if spec_key in _figure_cache:
            if _figure_epoch.get(spec_key) == epoch:
                spec_patch = no_update
            else:
                spec_patch = _patch_epoch_marker(epoch)
                _figure_epoch[spec_key] = epoch
        else:
            spec_patch = _render_and_cache_spec_trajectory(epoch)

        if flat_key in _figure_cache:
            if _figure_epoch.get(flat_key) == epoch:
                flat_patch = no_update
            else:
                flat_patch = _patch_epoch_marker(epoch)
                _figure_epoch[flat_key] = epoch
        else:
            flat_patch = _render_and_cache_flatness(epoch, "mean_delta_loss")

        # Full re-render per-epoch plots
        freq_fig = _render_per_epoch_freq(epoch)
        activation_fig = _render_per_epoch_activation(epoch, neuron_idx)
        clusters_fig = _render_per_epoch_clusters(epoch)

        epoch_display = f"Epoch {epoch} (Index {epoch_idx})"

        return (
            loss_patch, spec_patch, flat_patch,
            freq_fig, activation_fig, clusters_fig,
            epoch_display,
        )

    # --- Click-to-navigate on summary plots ---
    @app.callback(
        Output("epoch-slider", "value", allow_duplicate=True),
        Input("loss-plot", "clickData"),
        Input("spec-trajectory-plot", "clickData"),
        Input("flatness-trajectory-plot", "clickData"),
        prevent_initial_call=True,
    )
    def on_summary_click(loss_click, spec_click, flat_click):
        from dash import ctx

        # Use triggered_id to get click data from the plot that was actually clicked
        click_map = {
            "loss-plot": loss_click,
            "spec-trajectory-plot": spec_click,
            "flatness-trajectory-plot": flat_click,
        }
        triggered = ctx.triggered_id
        click_data = click_map.get(triggered) if isinstance(triggered, str) else None
        if not click_data or not click_data.get("points"):
            return no_update
        clicked_epoch = click_data["points"][0].get("x")
        if clicked_epoch is None:
            return no_update
        return server_state.nearest_epoch_index(int(clicked_epoch))

    # --- Click on freq clusters heatmap → navigate to neuron ---
    @app.callback(
        Output("neuron-slider", "value", allow_duplicate=True),
        Input("clusters-plot", "clickData"),
        prevent_initial_call=True,
    )
    def on_clusters_click(click_data):
        if not click_data or not click_data.get("points"):
            return no_update
        # Heatmap x-axis is neuron index
        neuron_idx = click_data["points"][0].get("x")
        if neuron_idx is None:
            return no_update
        neuron_idx = int(neuron_idx)
        if 0 <= neuron_idx < server_state.n_neurons:
            return neuron_idx
        return no_update

    # --- Neuron slider → re-render activation only ---
    @app.callback(
        Output("activation-plot", "figure", allow_duplicate=True),
        Output("neuron-display", "children", allow_duplicate=True),
        Input("neuron-slider", "value"),
        State("epoch-slider", "value"),
        prevent_initial_call=True,
    )
    def on_neuron_change(neuron_idx: int, epoch_idx: int):
        epoch = server_state.get_epoch_at_index(epoch_idx)
        fig = _render_per_epoch_activation(epoch, neuron_idx)
        return fig, f"Neuron {neuron_idx}"

    # --- Flatness metric change → full re-render flatness trajectory ---
    @app.callback(
        Output("flatness-trajectory-plot", "figure", allow_duplicate=True),
        Input("flatness-metric-dropdown", "value"),
        State("epoch-slider", "value"),
        prevent_initial_call=True,
    )
    def on_flatness_metric_change(metric: str, epoch_idx: int):
        epoch = server_state.get_epoch_at_index(epoch_idx)
        return _render_and_cache_flatness(epoch, metric)

    # --- Sidebar toggle ---
    @app.callback(
        Output("sidebar", "style"),
        Output("sidebar-collapsed", "style"),
        Output("collapsed-status", "children"),
        Input("sidebar-toggle", "n_clicks"),
        Input("sidebar-expand", "n_clicks"),
        State("sidebar", "style"),
        prevent_initial_call=True,
    )
    def toggle_sidebar(collapse_clicks, expand_clicks, current_style):
        from dash import ctx

        sidebar_visible = {
            "width": "280px",
            "minWidth": "280px",
            "padding": "20px",
            "backgroundColor": "#f8f9fa",
            "overflowY": "auto",
            "borderRight": "1px solid #dee2e6",
            "height": "100vh",
            "position": "sticky",
            "top": "0",
        }
        sidebar_hidden = {**sidebar_visible, "display": "none"}

        collapsed_visible = {
            "width": "40px",
            "minWidth": "40px",
            "padding": "10px 5px",
            "backgroundColor": "#f8f9fa",
            "borderRight": "1px solid #dee2e6",
            "height": "100vh",
            "position": "sticky",
            "top": "0",
            "display": "block",
        }
        collapsed_hidden = {**collapsed_visible, "display": "none"}

        epoch = server_state.get_epoch_at_index(0)
        status_text = f"E:{epoch}"

        if ctx.triggered_id == "sidebar-toggle":
            return sidebar_hidden, collapsed_visible, status_text
        else:
            return sidebar_visible, collapsed_hidden, ""


# ---------------------------------------------------------------------------
# Per-epoch rendering helpers
# ---------------------------------------------------------------------------


def _render_per_epoch_freq(epoch: int) -> go.Figure:
    """Render dominant frequencies for a single epoch."""
    loader = server_state.get_loader()
    if loader is None or "dominant_frequencies" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    try:
        epoch_data = loader.load_epoch("dominant_frequencies", epoch)
        return render_dominant_frequencies(epoch_data, epoch=epoch, threshold=1.0)
    except FileNotFoundError:
        return _empty_figure("No data for this epoch")


def _render_per_epoch_activation(epoch: int, neuron_idx: int) -> go.Figure:
    """Render neuron activation heatmap for a single epoch."""
    loader = server_state.get_loader()
    if loader is None or "neuron_activations" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    try:
        epoch_data = loader.load_epoch("neuron_activations", epoch)
        return render_neuron_heatmap(epoch_data, epoch=epoch, neuron_idx=neuron_idx)
    except FileNotFoundError:
        return _empty_figure("No data for this epoch")


def _render_per_epoch_clusters(epoch: int) -> go.Figure:
    """Render frequency clusters for a single epoch."""
    loader = server_state.get_loader()
    if loader is None or "neuron_freq_norm" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    try:
        epoch_data = loader.load_epoch("neuron_freq_norm", epoch)
        return render_freq_clusters(epoch_data, epoch=epoch)
    except FileNotFoundError:
        return _empty_figure("No data for this epoch")
