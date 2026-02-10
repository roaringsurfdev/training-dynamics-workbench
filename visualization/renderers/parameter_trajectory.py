"""REQ_029: Parameter Space Trajectory Visualizations.

Renders PCA trajectory projections, explained variance, and
parameter velocity from weight matrix snapshots across training.

All cross-epoch renderers accept a list of per-epoch snapshot dicts
loaded via ArtifactLoader.load_epochs("parameter_snapshot"). PCA and
velocity are computed at render time from the raw weight matrices.
"""

import numpy as np
import plotly.graph_objects as go

from analysis.library.trajectory import (
    compute_parameter_velocity,
    compute_pca_trajectory,
)
from analysis.library.weights import COMPONENT_GROUPS


def render_parameter_trajectory(
    snapshots: list[dict[str, np.ndarray]],
    epochs: list[int],
    current_epoch: int,
    components: list[str] | None = None,
    title: str | None = None,
    height: int = 450,
) -> go.Figure:
    """2D PCA trajectory of parameters across training.

    Points colored by epoch (blue=early, red=late), connected by path.
    Current epoch highlighted with a larger marker.

    Args:
        snapshots: List of per-epoch snapshot dicts, ordered by epoch.
        epochs: Epoch numbers corresponding to each snapshot.
        current_epoch: Current epoch for highlight marker.
        components: Weight matrix names to include. None = all.
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with 2D trajectory scatter plot.
    """
    pca_result = compute_pca_trajectory(snapshots, components, n_components=2)
    projections = pca_result["projections"]
    var_ratio = pca_result["explained_variance_ratio"]

    fig = go.Figure()

    # Trajectory path (thin line connecting points)
    fig.add_trace(
        go.Scatter(
            x=projections[:, 0],
            y=projections[:, 1],
            mode="lines",
            line=dict(color="rgba(150, 150, 150, 0.4)", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Trajectory points colored by epoch
    fig.add_trace(
        go.Scatter(
            x=projections[:, 0],
            y=projections[:, 1],
            mode="markers",
            marker=dict(
                size=5,
                color=epochs,
                colorscale="Bluered",
                showscale=True,
                colorbar=dict(title="Epoch", thickness=15),
            ),
            name="Checkpoints",
            hovertemplate=(
                "Epoch %{customdata}<br>"
                "PC1: %{x:.4f}<br>"
                "PC2: %{y:.4f}<extra></extra>"
            ),
            customdata=epochs,
        )
    )

    # Highlight current epoch
    if current_epoch in epochs:
        idx = epochs.index(current_epoch)
        fig.add_trace(
            go.Scatter(
                x=[projections[idx, 0]],
                y=[projections[idx, 1]],
                mode="markers",
                marker=dict(
                    size=14,
                    color="red",
                    symbol="star",
                    line=dict(width=1, color="black"),
                ),
                name=f"Epoch {current_epoch}",
                hovertemplate=f"Epoch {current_epoch}<extra></extra>",
            )
        )

    component_label = _get_component_label(components)
    if title is None:
        title = f"Parameter Trajectory ({component_label})"

    pct1 = var_ratio[0] * 100
    pct2 = var_ratio[1] * 100

    fig.update_layout(
        title=title,
        xaxis_title=f"PC1 ({pct1:.1f}% var)",
        yaxis_title=f"PC2 ({pct2:.1f}% var)",
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def render_explained_variance(
    snapshots: list[dict[str, np.ndarray]],
    components: list[str] | None = None,
    n_components: int = 10,
    title: str | None = None,
    height: int = 300,
) -> go.Figure:
    """Scree plot of explained variance per principal component.

    Args:
        snapshots: List of per-epoch snapshot dicts.
        components: Weight matrix names to include. None = all.
        n_components: Max number of PCs to show.
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with bar chart and cumulative line.
    """
    n_components = min(n_components, len(snapshots))
    pca_result = compute_pca_trajectory(snapshots, components, n_components)
    var_ratio = pca_result["explained_variance_ratio"]
    cumulative = np.cumsum(var_ratio)

    pc_labels = [f"PC{i+1}" for i in range(len(var_ratio))]

    fig = go.Figure()

    # Individual variance bars
    fig.add_trace(
        go.Bar(
            x=pc_labels,
            y=var_ratio * 100,
            name="Individual",
            marker_color="rgba(31, 119, 180, 0.7)",
            hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
        )
    )

    # Cumulative line
    fig.add_trace(
        go.Scatter(
            x=pc_labels,
            y=cumulative * 100,
            mode="lines+markers",
            name="Cumulative",
            line=dict(color="rgba(214, 39, 40, 0.8)", width=2),
            marker=dict(size=5),
            hovertemplate="%{x}: %{y:.1f}% cumulative<extra></extra>",
        )
    )

    component_label = _get_component_label(components)
    if title is None:
        title = f"Explained Variance ({component_label})"

    fig.update_layout(
        title=title,
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained (%)",
        yaxis=dict(range=[0, 105]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def render_parameter_velocity(
    snapshots: list[dict[str, np.ndarray]],
    epochs: list[int],
    current_epoch: int,
    components: list[str] | None = None,
    title: str | None = None,
    height: int = 300,
) -> go.Figure:
    """Parameter velocity (L2 norm of change) over training epochs.

    Args:
        snapshots: List of per-epoch snapshot dicts, ordered by epoch.
        epochs: Epoch numbers corresponding to each snapshot.
        current_epoch: Current epoch for vertical indicator.
        components: Weight matrix names to include. None = all.
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with velocity line plot.
    """
    velocity = compute_parameter_velocity(snapshots, components, epochs=epochs)
    velocity_epochs = epochs[1:]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=velocity_epochs,
            y=velocity,
            mode="lines",
            name="Velocity",
            line=dict(color="rgba(31, 119, 180, 1.0)", width=2),
            hovertemplate="Epoch %{x}<br>Velocity: %{y:.4f}<extra></extra>",
        )
    )

    # Epoch indicator
    fig.add_vline(
        x=current_epoch,
        line_dash="solid",
        line_color="red",
        line_width=2,
        annotation_text=f"Epoch {current_epoch}",
        annotation_position="top right",
        annotation_font_color="red",
    )

    component_label = _get_component_label(components)
    if title is None:
        title = f"Parameter Velocity ({component_label})"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="||delta theta|| / epoch",
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def render_component_velocity(
    snapshots: list[dict[str, np.ndarray]],
    epochs: list[int],
    current_epoch: int,
    title: str | None = None,
    height: int = 350,
) -> go.Figure:
    """Per-component group velocity over training epochs.

    One line per component group (Embedding, Attention, MLP).
    Reveals timing differences in when parts of the model move.

    Args:
        snapshots: List of per-epoch snapshot dicts, ordered by epoch.
        epochs: Epoch numbers corresponding to each snapshot.
        current_epoch: Current epoch for vertical indicator.
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with multi-line velocity plot.
    """
    velocity_epochs = epochs[1:]
    colors = {
        "embedding": "rgba(31, 119, 180, 1.0)",
        "attention": "rgba(44, 160, 44, 1.0)",
        "mlp": "rgba(214, 39, 40, 1.0)",
    }

    fig = go.Figure()

    for group_name, group_components in COMPONENT_GROUPS.items():
        velocity = compute_parameter_velocity(snapshots, group_components, epochs=epochs)
        fig.add_trace(
            go.Scatter(
                x=velocity_epochs,
                y=velocity,
                mode="lines",
                name=group_name.capitalize(),
                line=dict(color=colors.get(group_name, "gray"), width=2),
                hovertemplate=(
                    f"{group_name.capitalize()}<br>"
                    "Epoch %{x}<br>"
                    "Velocity: %{y:.4f}<extra></extra>"
                ),
            )
        )

    # Epoch indicator
    fig.add_vline(
        x=current_epoch,
        line_dash="solid",
        line_color="red",
        line_width=2,
        annotation_text=f"Epoch {current_epoch}",
        annotation_position="top right",
        annotation_font_color="red",
    )

    if title is None:
        title = "Component Velocity Comparison"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="||delta theta|| / epoch",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def _get_component_label(components: list[str] | None) -> str:
    """Generate a human-readable label for the selected components."""
    if components is None:
        return "All Parameters"

    # Check if components match a named group
    component_set = set(components)
    for group_name, group_components in COMPONENT_GROUPS.items():
        if component_set == set(group_components):
            return group_name.capitalize()

    return f"{len(components)} matrices"
