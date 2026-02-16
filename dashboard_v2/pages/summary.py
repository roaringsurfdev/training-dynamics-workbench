"""REQ_041: Summary Lens page.

Dense, read-mostly layout showing a single variant's full training story
with 12 cross-epoch/summary visualizations. A temporal cursor (epoch slider)
synchronizes a vertical indicator across all time-axis plots.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Dash, Input, Output, Patch, State, dcc, html, no_update

from dashboard_v2.components.family_selector import get_family_choices, get_variant_choices
from dashboard_v2.components.loss_curves import render_loss_curves_with_indicator
from dashboard_v2.state import get_registry, server_state
from miscope.visualization.renderers.attention_freq import (
    render_attention_dominant_frequencies,
    render_attention_specialization_trajectory,
)
from miscope.visualization.renderers.dominant_frequencies import (
    render_dominant_frequencies_over_time,
)
from miscope.visualization.renderers.effective_dimensionality import (
    render_dimensionality_trajectory,
)
from miscope.visualization.renderers.neuron_freq_clusters import (
    render_specialization_by_frequency,
    render_specialization_trajectory,
)
from miscope.visualization.renderers.parameter_trajectory import (
    render_component_velocity,
    render_parameter_trajectory,
    render_trajectory_3d,
    render_trajectory_pc1_pc3,
    render_trajectory_pc2_pc3,
)

# ---------------------------------------------------------------------------
# Plot IDs (all prefixed "summary-" to avoid collisions)
# ---------------------------------------------------------------------------

_PLOT_IDS = [
    "summary-loss-plot",
    "summary-freq-over-time-plot",
    "summary-spec-trajectory-plot",
    "summary-spec-freq-plot",
    "summary-attn-spec-plot",
    "summary-attn-dom-freq-plot",
    "summary-trajectory-3d-plot",
    "summary-trajectory-plot",
    "summary-trajectory-pc1-pc3-plot",
    "summary-trajectory-pc2-pc3-plot",
    "summary-velocity-plot",
    "summary-dim-trajectory-plot",
]

# Plots whose epoch indicator is a vline shape (Patch-able)
_PATCHABLE_PLOTS = [
    "summary-loss-plot",
    "summary-spec-trajectory-plot",
    "summary-spec-freq-plot",
    "summary-attn-spec-plot",
    "summary-attn-dom-freq-plot",
    "summary-velocity-plot",
    "summary-dim-trajectory-plot",
]

# Trajectory plots that need full re-render (scatter marker, not vline)
_TRAJECTORY_PLOTS = [
    "summary-trajectory-plot",
    "summary-trajectory-3d-plot",
    "summary-trajectory-pc1-pc3-plot",
    "summary-trajectory-pc2-pc3-plot",
]

# ---------------------------------------------------------------------------
# Figure cache (module-level, same pattern as callbacks.py)
# ---------------------------------------------------------------------------

_figure_cache: dict[str, go.Figure] = {}
_figure_epoch: dict[str, int] = {}


def _cache_key(plot_id: str) -> str:
    variant = server_state.variant
    vname = variant.name if variant else "none"
    return f"{vname}:{plot_id}"


def _cache_figure(plot_id: str, fig: go.Figure, epoch: int) -> go.Figure:
    key = _cache_key(plot_id)
    _figure_cache[key] = fig
    _figure_epoch[key] = epoch
    return fig


def _patch_epoch_marker(epoch: int) -> Patch:
    """Patch the vline epoch indicator (shape + annotation)."""
    patched = Patch()
    patched["layout"]["shapes"][-1]["x0"] = epoch
    patched["layout"]["shapes"][-1]["x1"] = epoch
    patched["layout"]["annotations"][-1]["x"] = epoch
    patched["layout"]["annotations"][-1]["text"] = f"Epoch {epoch}"
    return patched


def _patch_or_skip(plot_id: str, epoch: int) -> Patch:
    """Return a Patch for the epoch marker, or no_update if unchanged."""
    key = _cache_key(plot_id)
    if key not in _figure_cache:
        return no_update  # type: ignore[return-value]
    if _figure_epoch.get(key) == epoch:
        return no_update  # type: ignore[return-value]
    _figure_epoch[key] = epoch
    return _patch_epoch_marker(epoch)


def _extract_group_pca(cross_epoch_data: dict, group: str) -> dict:
    """Extract PCA result dict for a component group."""
    return {
        "projections": cross_epoch_data[f"{group}__projections"],
        "explained_variance_ratio": cross_epoch_data[f"{group}__explained_variance_ratio"],
        "explained_variance": cross_epoch_data[f"{group}__explained_variance"],
    }


# ---------------------------------------------------------------------------
# Empty figure placeholder
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


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _s_render_loss(epoch: int) -> go.Figure:
    return _cache_figure(
        "summary-loss-plot",
        render_loss_curves_with_indicator(
            server_state.train_losses,
            server_state.test_losses,
            current_epoch=epoch,
            checkpoint_epochs=server_state.available_epochs,
        ),
        epoch,
    )


def _s_render_freq_over_time() -> go.Figure:
    """Rendered once on variant selection — no epoch param."""
    loader = server_state.get_loader()
    if loader is None or "dominant_frequencies" not in server_state.available_analyzers:
        return _empty_figure("No dominant_frequencies data")
    try:
        artifact = loader.load_epochs("dominant_frequencies")
        return render_dominant_frequencies_over_time(artifact)
    except FileNotFoundError:
        return _empty_figure("No dominant_frequencies data")


def _s_render_spec_trajectory(epoch: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "neuron_freq_norm" not in server_state.available_analyzers:
        return _empty_figure("No neuron specialization data")
    if not loader.has_summary("neuron_freq_norm"):
        return _empty_figure("No summary data")
    try:
        summary = loader.load_summary("neuron_freq_norm")
        return _cache_figure(
            "summary-spec-trajectory-plot",
            render_specialization_trajectory(summary, current_epoch=epoch),
            epoch,
        )
    except FileNotFoundError:
        return _empty_figure("No summary data")


def _s_render_spec_freq(epoch: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "neuron_freq_norm" not in server_state.available_analyzers:
        return _empty_figure("No neuron specialization data")
    if not loader.has_summary("neuron_freq_norm"):
        return _empty_figure("No summary data")
    try:
        summary = loader.load_summary("neuron_freq_norm")
        return _cache_figure(
            "summary-spec-freq-plot",
            render_specialization_by_frequency(summary, current_epoch=epoch),
            epoch,
        )
    except FileNotFoundError:
        return _empty_figure("No summary data")


def _s_render_attn_spec(epoch: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "attention_freq" not in server_state.available_analyzers:
        return _empty_figure("No attention specialization data")
    if not loader.has_summary("attention_freq"):
        return _empty_figure("No summary data")
    try:
        summary = loader.load_summary("attention_freq")
        return _cache_figure(
            "summary-attn-spec-plot",
            render_attention_specialization_trajectory(summary, current_epoch=epoch),
            epoch,
        )
    except FileNotFoundError:
        return _empty_figure("No summary data")


def _s_render_attn_dom_freq(epoch: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "attention_freq" not in server_state.available_analyzers:
        return _empty_figure("No attention specialization data")
    if not loader.has_summary("attention_freq"):
        return _empty_figure("No summary data")
    try:
        summary = loader.load_summary("attention_freq")
        return _cache_figure(
            "summary-attn-dom-freq-plot",
            render_attention_dominant_frequencies(summary, current_epoch=epoch),
            epoch,
        )
    except FileNotFoundError:
        return _empty_figure("No summary data")


def _s_render_trajectory_plots(
    epoch: int,
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, go.Figure]:
    """Render 4 PCA trajectory plots + component velocity."""
    data = server_state.get_trajectory_data()
    if data is None:
        empty = _empty_figure("No trajectory data")
        return empty, empty, empty, empty, empty
    try:
        traj_epochs = data["epochs"].tolist()
        pca = _extract_group_pca(data, "all")
        label = "All Parameters"
        t3d = _cache_figure(
            "summary-trajectory-3d-plot",
            render_trajectory_3d(pca, traj_epochs, epoch, group_label=label),
            epoch,
        )
        t_fig = _cache_figure(
            "summary-trajectory-plot",
            render_parameter_trajectory(pca, traj_epochs, epoch, group_label=label),
            epoch,
        )
        pc13 = _cache_figure(
            "summary-trajectory-pc1-pc3-plot",
            render_trajectory_pc1_pc3(pca, traj_epochs, epoch, group_label=label),
            epoch,
        )
        pc23 = _cache_figure(
            "summary-trajectory-pc2-pc3-plot",
            render_trajectory_pc2_pc3(pca, traj_epochs, epoch, group_label=label),
            epoch,
        )
        vel = _cache_figure(
            "summary-velocity-plot",
            render_component_velocity(data, traj_epochs, epoch),
            epoch,
        )
        return t3d, t_fig, pc13, pc23, vel
    except Exception:
        empty = _empty_figure("Error rendering trajectory")
        return empty, empty, empty, empty, empty


def _s_render_dim_trajectory(epoch: int) -> go.Figure:
    loader = server_state.get_loader()
    if loader is None or "effective_dimensionality" not in server_state.available_analyzers:
        return _empty_figure("No dimensionality data")
    if not loader.has_summary("effective_dimensionality"):
        return _empty_figure("No summary data")
    try:
        summary = loader.load_summary("effective_dimensionality")
        return _cache_figure(
            "summary-dim-trajectory-plot",
            render_dimensionality_trajectory(summary, current_epoch=epoch),
            epoch,
        )
    except FileNotFoundError:
        return _empty_figure("No summary data")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _graph(graph_id: str, height: str = "400px") -> dcc.Graph:
    return dcc.Graph(
        id=graph_id,
        config={"displayModeBar": True},
        style={"height": height},
    )


def create_summary_layout() -> html.Div:
    """Create the Summary Lens page layout."""
    controls = dbc.Row(
        [
            dbc.Col(
                [
                    dbc.Label("Family", className="fw-bold small"),
                    dcc.Dropdown(id="summary-family-dropdown", placeholder="Select family..."),
                ],
                width=2,
            ),
            dbc.Col(
                [
                    dbc.Label("Variant", className="fw-bold small"),
                    dcc.Dropdown(id="summary-variant-dropdown", placeholder="Select variant..."),
                ],
                width=3,
            ),
            dbc.Col(
                [
                    dbc.Label("Epoch", className="fw-bold small"),
                    dcc.Slider(
                        id="summary-epoch-slider",
                        min=0,
                        max=1,
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ],
                width=5,
            ),
            dbc.Col(
                [
                    dbc.Label("\u00a0", className="fw-bold small"),  # spacer
                    html.Div(
                        id="summary-epoch-display",
                        children="Epoch 0",
                        className="text-muted small",
                    ),
                ],
                width=2,
            ),
        ],
        className="mb-3 align-items-end",
    )

    grid = html.Div(
        [
            # Loss curve (full width)
            dbc.Row(dbc.Col(_graph("summary-loss-plot", "300px"))),
            # Embedding Fourier over time (full width)
            dbc.Row(dbc.Col(_graph("summary-freq-over-time-plot", "350px"))),
            # Neuron specialization | Attention head specialization
            dbc.Row(
                [
                    dbc.Col(_graph("summary-spec-trajectory-plot", "350px"), width=7),
                    dbc.Col(_graph("summary-attn-spec-plot", "350px"), width=5),
                ]
            ),
            # Specialized neurons by frequency (full width)
            dbc.Row(dbc.Col(_graph("summary-spec-freq-plot", "400px"))),
            # Attention dominant frequencies (full width)
            dbc.Row(dbc.Col(_graph("summary-attn-dom-freq-plot", "300px"))),
            # Trajectory 3D (full width)
            dbc.Row(dbc.Col(_graph("summary-trajectory-3d-plot", "550px"))),
            # PC1/PC2 | PC1/PC3 | PC2/PC3
            dbc.Row(
                [
                    dbc.Col(_graph("summary-trajectory-plot", "400px"), width=4),
                    dbc.Col(_graph("summary-trajectory-pc1-pc3-plot", "400px"), width=4),
                    dbc.Col(_graph("summary-trajectory-pc2-pc3-plot", "400px"), width=4),
                ]
            ),
            # Component velocity | Effective dimensionality
            dbc.Row(
                [
                    dbc.Col(_graph("summary-velocity-plot", "350px"), width=6),
                    dbc.Col(_graph("summary-dim-trajectory-plot", "350px"), width=6),
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


def register_summary_callbacks(app: Dash) -> None:
    """Register all callbacks for the Summary Lens page."""

    # --- Populate family dropdown on page load ---
    @app.callback(
        Output("summary-family-dropdown", "options"),
        Input("summary-family-dropdown", "id"),
    )
    def populate_summary_families(_: str) -> list[dict]:
        registry = get_registry()
        choices = get_family_choices(registry)
        return [{"label": display, "value": name} for display, name in choices]

    # --- Family change → update variant dropdown ---
    @app.callback(
        Output("summary-variant-dropdown", "options"),
        Output("summary-variant-dropdown", "value"),
        Input("summary-family-dropdown", "value"),
    )
    def on_summary_family_change(family_name: str | None):
        if not family_name:
            return [], None
        registry = get_registry()
        choices = get_variant_choices(registry, family_name)
        return [{"label": display, "value": name} for display, name in choices], None

    # --- Variant change → load data, render all 12 figures ---
    @app.callback(
        *[Output(pid, "figure") for pid in _PLOT_IDS],
        Output("summary-epoch-slider", "max"),
        Output("summary-epoch-slider", "value"),
        Output("summary-epoch-display", "children"),
        Input("summary-variant-dropdown", "value"),
        State("summary-family-dropdown", "value"),
    )
    def on_summary_variant_change(variant_name: str | None, family_name: str | None):
        empty = _empty_figure("Select a variant")
        if not variant_name or not family_name:
            return (*[empty] * len(_PLOT_IDS), 1, 0, "No variant selected")

        loaded = server_state.load_variant(family_name, variant_name)
        if not loaded:
            err = _empty_figure("Failed to load variant")
            return (*[err] * len(_PLOT_IDS), 1, 0, "Load failed")

        # Pre-warm trajectory cache
        server_state.get_trajectory_data()

        epochs = server_state.available_epochs
        epoch = epochs[0] if epochs else 0
        slider_max = max(len(epochs) - 1, 1)

        # Render all 12 figures
        loss = _s_render_loss(epoch)
        freq_over_time = _s_render_freq_over_time()
        spec_traj = _s_render_spec_trajectory(epoch)
        spec_freq = _s_render_spec_freq(epoch)
        attn_spec = _s_render_attn_spec(epoch)
        attn_dom = _s_render_attn_dom_freq(epoch)
        t3d, t_fig, pc13, pc23, vel = _s_render_trajectory_plots(epoch)
        dim_traj = _s_render_dim_trajectory(epoch)

        display = f"Epoch {epoch} (Index 0 / {len(epochs)})"

        # Return in _PLOT_IDS order
        return (
            loss,
            freq_over_time,
            spec_traj,
            spec_freq,
            attn_spec,
            attn_dom,
            t3d,
            t_fig,
            pc13,
            pc23,
            vel,
            dim_traj,
            slider_max,
            0,
            display,
        )

    # --- Epoch slider change → temporal cursor update ---
    @app.callback(
        *[Output(pid, "figure", allow_duplicate=True) for pid in _PLOT_IDS],
        Output("summary-epoch-display", "children", allow_duplicate=True),
        Input("summary-epoch-slider", "value"),
        prevent_initial_call=True,
    )
    def on_summary_epoch_change(epoch_idx: int):
        epoch = server_state.get_epoch_at_index(epoch_idx)

        # Patchable plots: move vline via Patch()
        loss = _patch_or_skip("summary-loss-plot", epoch)
        spec_traj = _patch_or_skip("summary-spec-trajectory-plot", epoch)
        spec_freq = _patch_or_skip("summary-spec-freq-plot", epoch)
        attn_spec = _patch_or_skip("summary-attn-spec-plot", epoch)
        attn_dom = _patch_or_skip("summary-attn-dom-freq-plot", epoch)
        vel = _patch_or_skip("summary-velocity-plot", epoch)
        dim_traj = _patch_or_skip("summary-dim-trajectory-plot", epoch)

        # Freq over time: no epoch param, skip
        freq_over_time = no_update

        # Trajectory plots: full re-render (scatter marker highlight)
        t3d, t_fig, pc13, pc23, _vel_dup = _s_render_trajectory_plots(epoch)
        # _vel_dup is redundant (already patched above), but we need the
        # trajectory render call. Use the patched version for velocity.

        epochs = server_state.available_epochs
        display = f"Epoch {epoch} (Index {epoch_idx} / {len(epochs)})"

        return (
            loss,
            freq_over_time,
            spec_traj,
            spec_freq,
            attn_spec,
            attn_dom,
            t3d,
            t_fig,
            pc13,
            pc23,
            vel,
            dim_traj,
            display,
        )
