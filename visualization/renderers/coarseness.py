"""REQ_024: Coarseness Visualizations.

Renders coarseness analysis data — both cross-epoch trajectories
from summary statistics and per-epoch distributions from per-neuron
coarseness values.
"""

import numpy as np
import plotly.graph_objects as go


def render_coarseness_trajectory(
    summary_data: dict[str, np.ndarray],
    current_epoch: int,
    blob_threshold: float = 0.7,
    title: str | None = None,
    height: int = 300,
) -> go.Figure:
    """Render mean coarseness trajectory with percentile band and epoch indicator.

    Shows how coarseness evolves across training checkpoints, with a shaded
    band for the 25th-75th percentile range and a vertical indicator at the
    current epoch.

    Args:
        summary_data: Dict from ArtifactLoader.load_summary("coarseness"),
            containing 'epochs', 'mean_coarseness', 'p25_coarseness',
            'p75_coarseness' arrays.
        current_epoch: Current epoch for vertical indicator line.
        blob_threshold: Coarseness threshold for blob neurons (horizontal ref).
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with trajectory line, percentile band, and indicators.
    """
    epochs = summary_data["epochs"]
    mean = summary_data["mean_coarseness"]
    p25 = summary_data["p25_coarseness"]
    p75 = summary_data["p75_coarseness"]

    fig = go.Figure()

    # P25-P75 percentile band (lower bound, invisible)
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=p25,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # P25-P75 percentile band (upper bound, filled to lower)
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=p75,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.15)",
            name="p25–p75",
            hoverinfo="skip",
        )
    )

    # Mean coarseness line
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=mean,
            mode="lines",
            name="Mean",
            line=dict(color="rgba(31, 119, 180, 1.0)", width=2),
            hovertemplate="Epoch %{x}<br>Mean Coarseness: %{y:.4f}<extra></extra>",
        )
    )

    # Blob threshold reference line
    fig.add_hline(
        y=blob_threshold,
        line_dash="dash",
        line_color="rgba(200, 100, 100, 0.5)",
        annotation_text=f"Blob threshold ({blob_threshold})",
        annotation_position="top right",
        annotation_font_color="rgba(200, 100, 100, 0.7)",
        annotation_font_size=10,
    )

    # Vertical epoch indicator
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
        title = "Coarseness Over Training"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Coarseness",
        yaxis=dict(range=[0, 1]),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def render_coarseness_distribution(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    blob_threshold: float = 0.7,
    plaid_threshold: float = 0.5,
    n_bins: int = 20,
    title: str | None = None,
    height: int = 300,
) -> go.Figure:
    """Render histogram of per-neuron coarseness values at a single epoch.

    Neurons are colored by region: plaid (< plaid_threshold),
    transitional (plaid_threshold to blob_threshold), and blob (>= blob_threshold).

    Args:
        epoch_data: Dict from ArtifactLoader.load_epoch("coarseness", epoch),
            containing 'coarseness' array of shape (d_mlp,).
        epoch: Epoch number (used for title).
        blob_threshold: Threshold above which neurons are "blob" neurons.
        plaid_threshold: Threshold below which neurons are "plaid" neurons.
        n_bins: Number of histogram bins over [0, 1].
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with colored histogram and threshold reference lines.
    """
    coarseness = epoch_data["coarseness"]
    n_total = len(coarseness)
    n_blob = int(np.sum(coarseness >= blob_threshold))

    # Split neurons into three groups for color coding
    plaid_vals = coarseness[coarseness < plaid_threshold]
    trans_vals = coarseness[(coarseness >= plaid_threshold) & (coarseness < blob_threshold)]
    blob_vals = coarseness[coarseness >= blob_threshold]

    bin_size = 1.0 / n_bins
    xbins = dict(start=0, end=1, size=bin_size)

    fig = go.Figure()

    # Plaid neurons
    if len(plaid_vals) > 0:
        fig.add_trace(
            go.Histogram(
                x=plaid_vals,
                xbins=xbins,
                marker_color="rgba(31, 119, 180, 0.7)",
                name="Plaid",
                hovertemplate="Coarseness: %{x:.2f}<br>Count: %{y}<extra></extra>",
            )
        )

    # Transitional neurons
    if len(trans_vals) > 0:
        fig.add_trace(
            go.Histogram(
                x=trans_vals,
                xbins=xbins,
                marker_color="rgba(148, 103, 189, 0.7)",
                name="Transitional",
                hovertemplate="Coarseness: %{x:.2f}<br>Count: %{y}<extra></extra>",
            )
        )

    # Blob neurons
    if len(blob_vals) > 0:
        fig.add_trace(
            go.Histogram(
                x=blob_vals,
                xbins=xbins,
                marker_color="rgba(214, 39, 40, 0.7)",
                name="Blob",
                hovertemplate="Coarseness: %{x:.2f}<br>Count: %{y}<extra></extra>",
            )
        )

    # Stack the histograms
    fig.update_layout(barmode="stack")

    # Threshold reference lines
    fig.add_vline(
        x=plaid_threshold,
        line_dash="dash",
        line_color="rgba(100, 100, 100, 0.5)",
        annotation_text="Plaid",
        annotation_position="top left",
        annotation_font_size=10,
        annotation_font_color="rgba(100, 100, 100, 0.7)",
    )

    fig.add_vline(
        x=blob_threshold,
        line_dash="dash",
        line_color="rgba(200, 100, 100, 0.5)",
        annotation_text="Blob",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color="rgba(200, 100, 100, 0.7)",
    )

    if title is None:
        title = f"Coarseness Distribution — Epoch {epoch} ({n_blob}/{n_total} blob neurons)"

    fig.update_layout(
        title=title,
        xaxis_title="Coarseness",
        yaxis_title="Neuron Count",
        xaxis=dict(range=[0, 1]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def render_blob_count_trajectory(
    summary_data: dict[str, np.ndarray],
    current_epoch: int | None = None,
    title: str | None = None,
    height: int = 300,
) -> go.Figure:
    """Render blob neuron count over training epochs.

    Args:
        summary_data: Dict from ArtifactLoader.load_summary("coarseness"),
            containing 'epochs' and 'blob_count' arrays.
        current_epoch: Optional current epoch for vertical indicator.
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with blob count line plot.
    """
    epochs = summary_data["epochs"]
    blob_count = summary_data["blob_count"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=blob_count,
            mode="lines+markers",
            name="Blob Neurons",
            line=dict(color="rgba(214, 39, 40, 0.8)", width=2),
            marker=dict(size=4),
            hovertemplate="Epoch %{x}<br>Blob Count: %{y:.0f}<extra></extra>",
        )
    )

    if current_epoch is not None:
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
        title = "Blob Neuron Count Over Training"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Count (coarseness >= 0.7)",
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def render_coarseness_by_neuron(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    blob_threshold: float = 0.7,
    plaid_threshold: float = 0.5,
    title: str | None = None,
    height: int = 300,
    width: int = 900,
) -> go.Figure:
    """Render coarseness value per neuron index, colored by classification.

    Args:
        epoch_data: Dict from ArtifactLoader.load_epoch("coarseness", epoch),
            containing 'coarseness' array of shape (d_mlp,).
        epoch: Epoch number (used for title).
        blob_threshold: Threshold above which neurons are "blob" neurons.
        plaid_threshold: Threshold below which neurons are "plaid" neurons.
        title: Custom title.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure with bar chart of per-neuron coarseness.
    """
    coarseness = epoch_data["coarseness"]
    n_neurons = len(coarseness)
    neuron_indices = list(range(n_neurons))

    # Color each bar by region
    colors = []
    for val in coarseness:
        if val >= blob_threshold:
            colors.append("rgba(214, 39, 40, 0.7)")
        elif val >= plaid_threshold:
            colors.append("rgba(148, 103, 189, 0.7)")
        else:
            colors.append("rgba(31, 119, 180, 0.7)")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=neuron_indices,
            y=coarseness,
            marker_color=colors,
            hovertemplate="Neuron %{x}<br>Coarseness: %{y:.4f}<extra></extra>",
        )
    )

    # Threshold reference lines
    fig.add_hline(
        y=blob_threshold,
        line_dash="dash",
        line_color="rgba(200, 100, 100, 0.5)",
        annotation_text="Blob",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color="rgba(200, 100, 100, 0.7)",
    )

    fig.add_hline(
        y=plaid_threshold,
        line_dash="dash",
        line_color="rgba(100, 100, 100, 0.4)",
        annotation_text="Plaid",
        annotation_position="bottom right",
        annotation_font_size=10,
        annotation_font_color="rgba(100, 100, 100, 0.6)",
    )

    if title is None:
        title = f"Coarseness by Neuron — Epoch {epoch}"

    fig.update_layout(
        title=title,
        xaxis_title="Neuron Index",
        yaxis_title="Coarseness",
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        height=height,
        width=width,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig
