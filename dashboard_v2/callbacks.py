"""Callback registrations for the Dash dashboard.

All 18 Analysis tab visualizations with:
- Variant selection: loads data, renders all plots
- Epoch change: Patch() for summary/trajectory markers, full re-render for per-epoch
- Click-to-navigate: summary/trajectory plot click → epoch slider update
- Click-to-navigate: freq clusters click → neuron slider update
- Control-specific: neuron, position pair, trajectory group, SV matrix/head, flatness metric
- Sidebar toggle
"""

from __future__ import annotations

import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, no_update

from analysis.library.weights import ATTENTION_MATRICES
from dashboard.components.family_selector import get_family_choices, get_variant_choices
from dashboard.components.loss_curves import render_loss_curves_with_indicator
from dashboard_v2.state import get_registry, server_state
from visualization.renderers.attention_freq import (
    render_attention_freq_heatmap,
    render_attention_specialization_trajectory,
)
from visualization.renderers.attention_patterns import render_attention_heads
from visualization.renderers.dominant_frequencies import render_dominant_frequencies
from visualization.renderers.effective_dimensionality import (
    render_dimensionality_trajectory,
    render_singular_value_spectrum,
)
from visualization.renderers.landscape_flatness import (
    render_flatness_trajectory,
    render_perturbation_distribution,
)
from visualization.renderers.neuron_activations import render_neuron_heatmap
from visualization.renderers.neuron_freq_clusters import (
    render_freq_clusters,
    render_specialization_by_frequency,
    render_specialization_trajectory,
)
from visualization.renderers.parameter_trajectory import (
    get_group_label,
    render_component_velocity,
    render_parameter_trajectory,
    render_trajectory_3d,
    render_trajectory_pc1_pc3,
    render_trajectory_pc2_pc3,
)


def _empty_figure(message: str = "No data") -> go.Figure:
    """Create a placeholder figure with a centered message."""
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


# ---------------------------------------------------------------------------
# Summary figure cache
# ---------------------------------------------------------------------------
# Keyed by (variant_name, plot_name). Stores base figures so Patch() can
# update just the epoch marker without full re-render.
# _figure_epoch tracks what epoch each cached figure currently shows,
# used to skip redundant Patch() calls that race with full renders.

_figure_cache: dict[str, go.Figure] = {}
_figure_epoch: dict[str, int] = {}

# Plot names for all summary/trajectory plots that use Patch()
_PATCHABLE_PLOTS = [
    "loss",
    "spec_traj",
    "spec_freq",
    "attn_spec",
    "dim_traj",
    "flatness",
    "velocity",
]
# Trajectory 2D/3D plots use scatter trace markers (not add_vline shapes),
# so they need full re-render on epoch change instead of Patch().
_TRAJECTORY_RERENDER_PLOTS = [
    "trajectory",
    "trajectory_3d",
    "trajectory_pc1_pc3",
    "trajectory_pc2_pc3",
]


def _cache_key(plot_name: str) -> str:
    """Build cache key from current variant + plot name."""
    variant = server_state.variant
    vname = variant.name if variant else "none"
    return f"{vname}:{plot_name}"


def _cache_figure(plot_name: str, fig: go.Figure, epoch: int) -> go.Figure:
    """Store a figure in the cache with its epoch and return it."""
    key = _cache_key(plot_name)
    _figure_cache[key] = fig
    _figure_epoch[key] = epoch
    return fig


def _patch_or_skip(plot_name: str, epoch: int) -> Patch | go.Figure:
    """Return a Patch for epoch marker update, or no_update if already current.

    If the figure is not cached, returns a full re-render.
    """
    key = _cache_key(plot_name)
    if key not in _figure_cache:
        # No cached figure — need full render (caller handles this)
        return no_update  # type: ignore[return-value]
    if _figure_epoch.get(key) == epoch:
        return no_update  # type: ignore[return-value]
    _figure_epoch[key] = epoch
    return _patch_epoch_marker(epoch)


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


def _extract_group_pca(
    cross_epoch_data: dict,
    group: str,
) -> dict:
    """Extract a pca_result dict for a specific component group."""
    return {
        "projections": cross_epoch_data[f"{group}__projections"],
        "explained_variance_ratio": cross_epoch_data[f"{group}__explained_variance_ratio"],
        "explained_variance": cross_epoch_data[f"{group}__explained_variance"],
    }


# ---------------------------------------------------------------------------
# Summary/cross-epoch render-and-cache functions
# ---------------------------------------------------------------------------


def _render_loss(epoch: int) -> go.Figure:
    return _cache_figure(
        "loss",
        render_loss_curves_with_indicator(
            server_state.train_losses,
            server_state.test_losses,
            current_epoch=epoch,
            checkpoint_epochs=server_state.available_epochs,
        ),
        epoch,
    )


def _render_spec_trajectory(epoch: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "neuron_freq_norm" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    if not loader.has_summary("neuron_freq_norm"):
        return _empty_figure("No summary data")
    try:
        summary = loader.load_summary("neuron_freq_norm")
        return _cache_figure(
            "spec_traj", render_specialization_trajectory(summary, current_epoch=epoch), epoch
        )
    except FileNotFoundError:
        return _empty_figure("No summary data")


def _render_spec_freq(epoch: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "neuron_freq_norm" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    if not loader.has_summary("neuron_freq_norm"):
        return _empty_figure("No summary data")
    try:
        summary = loader.load_summary("neuron_freq_norm")
        return _cache_figure(
            "spec_freq", render_specialization_by_frequency(summary, current_epoch=epoch), epoch
        )
    except FileNotFoundError:
        return _empty_figure("No summary data")


def _render_attn_spec(epoch: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "attention_freq" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    if not loader.has_summary("attention_freq"):
        return _empty_figure("No summary data")
    try:
        summary = loader.load_summary("attention_freq")
        return _cache_figure(
            "attn_spec",
            render_attention_specialization_trajectory(summary, current_epoch=epoch),
            epoch,
        )
    except FileNotFoundError:
        return _empty_figure("No summary data")


def _render_dim_trajectory(epoch: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "effective_dimensionality" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    if not loader.has_summary("effective_dimensionality"):
        return _empty_figure("No summary data")
    try:
        summary = loader.load_summary("effective_dimensionality")
        return _cache_figure(
            "dim_traj", render_dimensionality_trajectory(summary, current_epoch=epoch), epoch
        )
    except FileNotFoundError:
        return _empty_figure("No summary data")


def _render_flatness(epoch: int, metric: str) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "landscape_flatness" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    if not loader.has_summary("landscape_flatness"):
        return _empty_figure("No summary data")
    try:
        summary = loader.load_summary("landscape_flatness")
        return _cache_figure(
            "flatness",
            render_flatness_trajectory(summary, current_epoch=epoch, metric=metric),
            epoch,
        )
    except FileNotFoundError:
        return _empty_figure("No summary data")


def _render_trajectory_plots(
    epoch: int,
    group: str,
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure]:
    """Render all 5 trajectory/velocity plots and cache them."""
    data = server_state.get_trajectory_data()
    if data is None:
        empty = _empty_figure("Run analysis first")
        return empty, empty, empty, empty, empty
    try:
        traj_epochs = data["epochs"].tolist()
        pca = _extract_group_pca(data, group)
        label = get_group_label(group)
        t_fig = _cache_figure(
            "trajectory",
            render_parameter_trajectory(pca, traj_epochs, epoch, group_label=label),
            epoch,
        )
        t3d_fig = _cache_figure(
            "trajectory_3d",
            render_trajectory_3d(pca, traj_epochs, epoch, group_label=label),
            epoch,
        )
        pc13_fig = _cache_figure(
            "trajectory_pc1_pc3",
            render_trajectory_pc1_pc3(pca, traj_epochs, epoch, group_label=label),
            epoch,
        )
        pc23_fig = _cache_figure(
            "trajectory_pc2_pc3",
            render_trajectory_pc2_pc3(pca, traj_epochs, epoch, group_label=label),
            epoch,
        )
        vel_fig = _cache_figure(
            "velocity",
            render_component_velocity(data, traj_epochs, epoch),
            epoch,
        )
        return t_fig, t3d_fig, pc13_fig, pc23_fig, vel_fig
    except Exception:
        empty = _empty_figure("Error rendering trajectory")
        return empty, empty, empty, empty, empty


# ---------------------------------------------------------------------------
# Per-epoch render functions
# ---------------------------------------------------------------------------


def _render_freq(epoch: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "dominant_frequencies" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    try:
        return render_dominant_frequencies(
            loader.load_epoch("dominant_frequencies", epoch), epoch=epoch, threshold=1.0
        )
    except FileNotFoundError:
        return _empty_figure("No data for this epoch")


def _render_activation(epoch: int, neuron_idx: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "neuron_activations" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    try:
        return render_neuron_heatmap(
            loader.load_epoch("neuron_activations", epoch), epoch=epoch, neuron_idx=neuron_idx
        )
    except FileNotFoundError:
        return _empty_figure("No data for this epoch")


def _render_clusters(epoch: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "neuron_freq_norm" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    try:
        return render_freq_clusters(loader.load_epoch("neuron_freq_norm", epoch), epoch=epoch)
    except FileNotFoundError:
        return _empty_figure("No data for this epoch")


def _render_attention(epoch: int, to_pos: int, from_pos: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "attention_patterns" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    try:
        return render_attention_heads(
            loader.load_epoch("attention_patterns", epoch),
            epoch=epoch,
            to_position=to_pos,
            from_position=from_pos,
        )
    except FileNotFoundError:
        return _empty_figure("No data for this epoch")


def _render_attn_freq(epoch: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "attention_freq" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    try:
        return render_attention_freq_heatmap(
            loader.load_epoch("attention_freq", epoch), epoch=epoch
        )
    except FileNotFoundError:
        return _empty_figure("No data for this epoch")


def _render_sv_spectrum(epoch: int, matrix: str, head_idx: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "effective_dimensionality" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    try:
        head = head_idx if matrix in ATTENTION_MATRICES else None
        return render_singular_value_spectrum(
            loader.load_epoch("effective_dimensionality", epoch),
            epoch=epoch,
            matrix_name=matrix,
            head_idx=head,
        )
    except FileNotFoundError:
        return _empty_figure("No data for this epoch")


def _render_perturbation(epoch: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "landscape_flatness" not in server_state.available_analyzers:
        return _empty_figure("Run analysis first")
    try:
        return render_perturbation_distribution(
            loader.load_epoch("landscape_flatness", epoch), epoch=epoch
        )
    except FileNotFoundError:
        return _empty_figure("No data for this epoch")


# ---------------------------------------------------------------------------
# All 18 plot IDs (output order for variant change and epoch change)
# ---------------------------------------------------------------------------

_ALL_PLOT_IDS = [
    "loss-plot",
    "freq-plot",
    "activation-plot",
    "clusters-plot",
    "spec-trajectory-plot",
    "spec-freq-plot",
    "attention-plot",
    "attn-freq-plot",
    "attn-spec-plot",
    "trajectory-plot",
    "trajectory-3d-plot",
    "trajectory-pc1-pc3-plot",
    "trajectory-pc2-pc3-plot",
    "velocity-plot",
    "dim-trajectory-plot",
    "sv-spectrum-plot",
    "flatness-trajectory-plot",
    "perturbation-plot",
]

# Summary/trajectory plot IDs that support click-to-navigate (epoch)
_CLICK_NAV_PLOT_IDS = [
    "loss-plot",
    "spec-trajectory-plot",
    "spec-freq-plot",
    "attn-spec-plot",
    "dim-trajectory-plot",
    "flatness-trajectory-plot",
]


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------


def register_callbacks(app: Dash) -> None:  # noqa: C901
    """Register all callbacks on the Dash app."""

    # --- Populate family dropdown on load ---
    @app.callback(
        Output("family-dropdown", "options"),
        Input("family-dropdown", "id"),
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

    # --- Variant change → load data, render all 18 plots ---
    @app.callback(
        *[Output(pid, "figure") for pid in _ALL_PLOT_IDS],
        Output("epoch-slider", "max"),
        Output("epoch-slider", "value"),
        Output("neuron-slider", "max"),
        Output("sv-head-slider", "max"),
        Output("epoch-display", "children"),
        Output("neuron-display", "children"),
        Output("status-display", "children"),
        Input("variant-dropdown", "value"),
        State("family-dropdown", "value"),
        State("flatness-metric-dropdown", "value"),
        State("trajectory-group-radio", "value"),
        State("position-pair-dropdown", "value"),
        State("sv-matrix-dropdown", "value"),
        State("sv-head-slider", "value"),
    )
    def on_variant_change(
        variant_name,
        family_name,
        flatness_metric,
        traj_group,
        position_pair,
        sv_matrix,
        sv_head,
    ):
        n_plots = len(_ALL_PLOT_IDS)
        empty = _empty_figure("Select a variant")

        if not variant_name or not family_name:
            server_state.clear()
            return (
                *[empty] * n_plots,
                1,
                0,
                511,
                3,
                "Epoch 0 (Index 0)",
                "Neuron 0",
                "No variant selected",
            )

        if not server_state.load_variant(family_name, variant_name):
            server_state.clear()
            return (
                *[_empty_figure("Variant not found")] * n_plots,
                1,
                0,
                511,
                3,
                "Epoch 0 (Index 0)",
                "Neuron 0",
                "Variant not found",
            )

        epoch = server_state.get_epoch_at_index(0)
        max_idx = max(0, len(server_state.available_epochs) - 1)
        to_pos, from_pos = _parse_position_pair(position_pair)

        # Summary/cross-epoch plots (cached for Patch)
        loss = _render_loss(epoch)
        spec_traj = _render_spec_trajectory(epoch)
        spec_freq = _render_spec_freq(epoch)
        attn_spec = _render_attn_spec(epoch)
        dim_traj = _render_dim_trajectory(epoch)
        flatness = _render_flatness(epoch, flatness_metric)
        t, t3d, pc13, pc23, vel = _render_trajectory_plots(epoch, traj_group)

        # Per-epoch plots
        freq = _render_freq(epoch)
        activation = _render_activation(epoch, 0)
        clusters = _render_clusters(epoch)
        attention = _render_attention(epoch, to_pos, from_pos)
        attn_freq = _render_attn_freq(epoch)
        sv_spectrum = _render_sv_spectrum(epoch, sv_matrix, sv_head)
        perturbation = _render_perturbation(epoch)

        # Status
        variant = server_state.variant
        assert variant is not None
        status_parts = [f"Variant: {variant.name}"]
        status_parts.append(f"State: {variant.state.value}")
        if server_state.available_epochs:
            status_parts.append(f"{len(server_state.available_epochs)} checkpoints")
        status = " | ".join(status_parts)

        return (
            # 18 plots in _ALL_PLOT_IDS order
            loss,
            freq,
            activation,
            clusters,
            spec_traj,
            spec_freq,
            attention,
            attn_freq,
            attn_spec,
            t,
            t3d,
            pc13,
            pc23,
            vel,
            dim_traj,
            sv_spectrum,
            flatness,
            perturbation,
            # Controls
            max_idx,
            0,
            max(0, server_state.n_neurons - 1),
            max(0, server_state.n_heads - 1),
            f"Epoch {epoch} (Index 0)",
            "Neuron 0",
            status,
        )

    # --- Epoch change → Patch summary markers + re-render per-epoch ---
    @app.callback(
        *[Output(pid, "figure", allow_duplicate=True) for pid in _ALL_PLOT_IDS],
        Output("epoch-display", "children", allow_duplicate=True),
        Input("epoch-slider", "value"),
        State("neuron-slider", "value"),
        State("position-pair-dropdown", "value"),
        State("sv-matrix-dropdown", "value"),
        State("sv-head-slider", "value"),
        State("trajectory-group-radio", "value"),
        prevent_initial_call=True,
    )
    def on_epoch_change(epoch_idx, neuron_idx, position_pair, sv_matrix, sv_head, traj_group):
        epoch = server_state.get_epoch_at_index(epoch_idx)
        to_pos, from_pos = _parse_position_pair(position_pair)

        # Patch summary/trajectory plots (skip if already at this epoch)
        loss_p = _patch_or_skip("loss", epoch)
        spec_traj_p = _patch_or_skip("spec_traj", epoch)
        spec_freq_p = _patch_or_skip("spec_freq", epoch)
        attn_spec_p = _patch_or_skip("attn_spec", epoch)
        dim_traj_p = _patch_or_skip("dim_traj", epoch)
        flatness_p = _patch_or_skip("flatness", epoch)

        # Velocity plot uses add_vline() — patchable
        vel_p = _patch_or_skip("velocity", epoch)

        # Trajectory 2D/3D plots use scatter trace markers (not shapes),
        # so they need full re-render to update the current-epoch marker.
        data = server_state.get_trajectory_data()
        if data is not None:
            traj_epochs = data["epochs"].tolist()
            pca = _extract_group_pca(data, traj_group)
            label = get_group_label(traj_group)
            traj_p = _cache_figure(
                "trajectory",
                render_parameter_trajectory(pca, traj_epochs, epoch, group_label=label),
                epoch,
            )
            t3d_p = _cache_figure(
                "trajectory_3d",
                render_trajectory_3d(pca, traj_epochs, epoch, group_label=label),
                epoch,
            )
            pc13_p = _cache_figure(
                "trajectory_pc1_pc3",
                render_trajectory_pc1_pc3(pca, traj_epochs, epoch, group_label=label),
                epoch,
            )
            pc23_p = _cache_figure(
                "trajectory_pc2_pc3",
                render_trajectory_pc2_pc3(pca, traj_epochs, epoch, group_label=label),
                epoch,
            )
        else:
            traj_p = no_update
            t3d_p = no_update
            pc13_p = no_update
            pc23_p = no_update

        # Full re-render per-epoch plots
        freq = _render_freq(epoch)
        activation = _render_activation(epoch, neuron_idx)
        clusters = _render_clusters(epoch)
        attention = _render_attention(epoch, to_pos, from_pos)
        attn_freq = _render_attn_freq(epoch)
        sv_spectrum = _render_sv_spectrum(epoch, sv_matrix, sv_head)
        perturbation = _render_perturbation(epoch)

        return (
            loss_p,
            freq,
            activation,
            clusters,
            spec_traj_p,
            spec_freq_p,
            attention,
            attn_freq,
            attn_spec_p,
            traj_p,
            t3d_p,
            pc13_p,
            pc23_p,
            vel_p,
            dim_traj_p,
            sv_spectrum,
            flatness_p,
            perturbation,
            f"Epoch {epoch} (Index {epoch_idx})",
        )

    # --- Click-to-navigate on summary plots → epoch ---
    @app.callback(
        Output("epoch-slider", "value", allow_duplicate=True),
        *[Input(pid, "clickData") for pid in _CLICK_NAV_PLOT_IDS],
        prevent_initial_call=True,
    )
    def on_summary_click(*click_args):
        from dash import ctx

        click_map = dict(zip(_CLICK_NAV_PLOT_IDS, click_args, strict=False))
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
    def on_neuron_change(neuron_idx, epoch_idx):
        epoch = server_state.get_epoch_at_index(epoch_idx)
        return _render_activation(epoch, neuron_idx), f"Neuron {neuron_idx}"

    # --- Position pair change → re-render attention only ---
    @app.callback(
        Output("attention-plot", "figure", allow_duplicate=True),
        Input("position-pair-dropdown", "value"),
        State("epoch-slider", "value"),
        prevent_initial_call=True,
    )
    def on_position_change(position_pair, epoch_idx):
        epoch = server_state.get_epoch_at_index(epoch_idx)
        to_pos, from_pos = _parse_position_pair(position_pair)
        return _render_attention(epoch, to_pos, from_pos)

    # --- Trajectory group change → re-render 5 trajectory plots ---
    @app.callback(
        Output("trajectory-plot", "figure", allow_duplicate=True),
        Output("trajectory-3d-plot", "figure", allow_duplicate=True),
        Output("trajectory-pc1-pc3-plot", "figure", allow_duplicate=True),
        Output("trajectory-pc2-pc3-plot", "figure", allow_duplicate=True),
        Output("velocity-plot", "figure", allow_duplicate=True),
        Input("trajectory-group-radio", "value"),
        State("epoch-slider", "value"),
        prevent_initial_call=True,
    )
    def on_trajectory_group_change(group, epoch_idx):
        epoch = server_state.get_epoch_at_index(epoch_idx)
        return _render_trajectory_plots(epoch, group)

    # --- SV matrix/head change → re-render spectrum + toggle head visibility ---
    @app.callback(
        Output("sv-spectrum-plot", "figure", allow_duplicate=True),
        Output("sv-head-container", "style"),
        Input("sv-matrix-dropdown", "value"),
        Input("sv-head-slider", "value"),
        State("epoch-slider", "value"),
        prevent_initial_call=True,
    )
    def on_sv_change(matrix, head_idx, epoch_idx):
        epoch = server_state.get_epoch_at_index(epoch_idx)
        fig = _render_sv_spectrum(epoch, matrix, head_idx)
        head_style = {"display": "block"} if matrix in ATTENTION_MATRICES else {"display": "none"}
        return fig, head_style

    # --- Flatness metric change → full re-render flatness trajectory ---
    @app.callback(
        Output("flatness-trajectory-plot", "figure", allow_duplicate=True),
        Input("flatness-metric-dropdown", "value"),
        State("epoch-slider", "value"),
        prevent_initial_call=True,
    )
    def on_flatness_metric_change(metric, epoch_idx):
        epoch = server_state.get_epoch_at_index(epoch_idx)
        return _render_flatness(epoch, metric)

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
# Helpers
# ---------------------------------------------------------------------------


def _parse_position_pair(value: str | None) -> tuple[int, int]:
    """Parse 'to,from' position pair string into (to_position, from_position)."""
    if value:
        parts = value.split(",")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    return 2, 0
