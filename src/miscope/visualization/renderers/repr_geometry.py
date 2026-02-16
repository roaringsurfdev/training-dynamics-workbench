"""REQ_044: Representational Geometry Visualizations.

Renders time-series of geometric measures (SNR, circularity, Fisher, etc.)
from summary data, and centroid PCA snapshots + distance heatmaps from
per-epoch data.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from miscope.analysis.library.geometry import _pca_project_2d

# Colors for activation sites
_SITE_COLORS = {
    "resid_pre": "rgba(31, 119, 180, 1.0)",
    "attn_out": "rgba(44, 160, 44, 1.0)",
    "mlp_out": "rgba(214, 39, 40, 1.0)",
    "resid_post": "rgba(148, 103, 189, 1.0)",
}

_SITE_LABELS = {
    "resid_pre": "Post-Embed",
    "attn_out": "Attn Out",
    "mlp_out": "MLP Out",
    "resid_post": "Resid Post",
}

_ALL_SITES = ["resid_pre", "attn_out", "mlp_out", "resid_post"]


def render_geometry_timeseries(
    summary_data: dict[str, np.ndarray],
    site: str | None = None,
    current_epoch: int | None = None,
    height: int = 1200,
) -> go.Figure:
    """Multi-panel time-series of geometric measures.

    Panels: SNR, center spread + mean radius, circularity + Fourier
    alignment, mean dimensionality, Fisher discriminant (mean + min).

    Args:
        summary_data: From ArtifactLoader.load_summary("repr_geometry").
            Contains "epochs" and site-prefixed scalar keys.
        site: Activation site to display. None = show all sites.
        current_epoch: Current epoch for vertical indicator.
        height: Total figure height in pixels.

    Returns:
        Plotly Figure with 5 vertically stacked subplots.
    """
    epochs = summary_data["epochs"]
    sites = [site] if site else _ALL_SITES

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(
            "Signal-to-Noise Ratio (SNR)",
            "Center Spread & Mean Radius",
            "Circularity & Fourier Alignment",
            "Mean Effective Dimensionality",
            "Fisher Discriminant",
        ),
    )

    for s in sites:
        color = _SITE_COLORS.get(s, "gray")
        label = _SITE_LABELS.get(s, s)
        show_legend = True

        # Panel 1: SNR (log scale)
        key = f"{s}_snr"
        if key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[key],
                    mode="lines",
                    name=label,
                    legendgroup=s,
                    showlegend=show_legend,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{label}<br>Epoch %{{x}}<br>SNR: %{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            show_legend = False

        # Panel 2: Center spread + mean radius
        spread_key = f"{s}_center_spread"
        radius_key = f"{s}_mean_radius"
        if spread_key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[spread_key],
                    mode="lines",
                    name=f"{label} spread",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{label} spread<br>Epoch %{{x}}<br>%{{y:.4f}}<extra></extra>",
                ),
                row=2,
                col=1,
            )
        if radius_key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[radius_key],
                    mode="lines",
                    name=f"{label} radius",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2, dash="dash"),
                    hovertemplate=f"{label} radius<br>Epoch %{{x}}<br>%{{y:.4f}}<extra></extra>",
                ),
                row=2,
                col=1,
            )

        # Panel 3: Circularity + Fourier alignment
        circ_key = f"{s}_circularity"
        fourier_key = f"{s}_fourier_alignment"
        if circ_key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[circ_key],
                    mode="lines",
                    name=f"{label} circ",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{label} circularity<br>Epoch %{{x}}<br>%{{y:.3f}}<extra></extra>",
                ),
                row=3,
                col=1,
            )
        if fourier_key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[fourier_key],
                    mode="lines",
                    name=f"{label} Fourier",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2, dash="dot"),
                    hovertemplate=f"{label} Fourier align<br>Epoch %{{x}}<br>%{{y:.3f}}<extra></extra>",
                ),
                row=3,
                col=1,
            )

        # Panel 4: Mean dimensionality
        dim_key = f"{s}_mean_dim"
        if dim_key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[dim_key],
                    mode="lines",
                    name=f"{label} dim",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{label} mean dim<br>Epoch %{{x}}<br>%{{y:.1f}}<extra></extra>",
                ),
                row=4,
                col=1,
            )

        # Panel 5: Fisher discriminant (mean solid, min dashed)
        fisher_mean_key = f"{s}_fisher_mean"
        fisher_min_key = f"{s}_fisher_min"
        if fisher_mean_key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[fisher_mean_key],
                    mode="lines",
                    name=f"{label} Fisher mean",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{label} Fisher mean<br>Epoch %{{x}}<br>%{{y:.2f}}<extra></extra>",
                ),
                row=5,
                col=1,
            )
        if fisher_min_key in summary_data:
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=summary_data[fisher_min_key],
                    mode="lines",
                    name=f"{label} Fisher min",
                    legendgroup=s,
                    showlegend=False,
                    line=dict(color=color, width=2, dash="dash"),
                    hovertemplate=f"{label} Fisher min<br>Epoch %{{x}}<br>%{{y:.2f}}<extra></extra>",
                ),
                row=5,
                col=1,
            )

    # Add epoch indicator
    if current_epoch is not None:
        for row in range(1, 6):
            fig.add_vline(
                x=current_epoch,
                line_dash="solid",
                line_color="red",
                line_width=1,
                row=row,
                col=1,
            )

    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], row=3, col=1)
    fig.update_xaxes(title_text="Epoch", row=5, col=1)

    fig.update_layout(
        template="plotly_white",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=40, b=40),
    )

    return fig


def render_centroid_pca(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    site: str = "resid_post",
    p: int | None = None,
    height: int = 500,
) -> go.Figure:
    """PCA scatter of class centroids at a single epoch.

    Projects centroids into top-2 PCs and colors by residue class
    using a cyclic colormap.

    Args:
        epoch_data: From ArtifactLoader.load_epoch("repr_geometry", epoch).
        epoch: Epoch number (for title).
        site: Activation site to display.
        p: Number of classes (inferred from centroid shape if None).
        height: Figure height in pixels.

    Returns:
        Plotly Figure with centroid PCA scatter.
    """
    centroid_key = f"{site}_centroids"
    centroids = epoch_data[centroid_key]
    if p is None:
        p = centroids.shape[0]

    projected, var_explained = _pca_project_2d(centroids)
    residues = np.arange(p)

    # Cyclic colorscale: map residue to hue
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=projected[:, 0],
            y=projected[:, 1],
            mode="markers+text",
            marker=dict(
                size=10,
                color=residues,
                colorscale="HSV",
                showscale=True,
                colorbar=dict(title="Residue"),
            ),
            text=[str(r) for r in residues],
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate="Residue %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
        )
    )

    site_label = _SITE_LABELS.get(site, site)
    fig.update_layout(
        title=f"Class Centroids PCA — {site_label} — Epoch {epoch} ({var_explained:.0%} var explained)",
        xaxis_title="PC1",
        yaxis_title="PC2",
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    return fig


def render_centroid_distances(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    site: str = "resid_post",
    p: int | None = None,
    height: int = 500,
) -> go.Figure:
    """Pairwise centroid distance heatmap at a single epoch.

    For learned modular structure, this should show a circulant pattern
    where distance depends on |r - s| mod p.

    Args:
        epoch_data: From ArtifactLoader.load_epoch("repr_geometry", epoch).
        epoch: Epoch number (for title).
        site: Activation site to display.
        p: Number of classes (inferred from centroid shape if None).
        height: Figure height in pixels.

    Returns:
        Plotly Figure with p×p distance heatmap.
    """
    centroid_key = f"{site}_centroids"
    centroids = epoch_data[centroid_key]
    if p is None:
        p = centroids.shape[0]

    # Compute pairwise distances
    diffs = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diffs**2, axis=2))

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=distances,
            x=list(range(p)),
            y=list(range(p)),
            colorscale="Viridis",
            colorbar=dict(title="Distance"),
            hovertemplate="Class %{x} ↔ Class %{y}<br>Distance: %{z:.3f}<extra></extra>",
        )
    )

    site_label = _SITE_LABELS.get(site, site)
    fig.update_layout(
        title=f"Centroid Pairwise Distances — {site_label} — Epoch {epoch}",
        xaxis_title="Residue Class",
        yaxis_title="Residue Class",
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
        yaxis=dict(autorange="reversed"),
    )

    return fig
