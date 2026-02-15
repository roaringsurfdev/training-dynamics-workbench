"""REQ_006: Neuron Frequency Cluster Visualization.

Renders neuron-frequency specialization matrix as a heatmap showing
what fraction of each neuron's activation is explained by each frequency.
"""

import numpy as np
import plotly.graph_objects as go


def render_freq_clusters(
    epoch_data: dict[str, np.ndarray],
    epoch: int,
    sparse_labels: bool = True,
    label_interval: int = 5,
    title: str | None = None,
    colorscale: str = "Viridis",
    show_colorbar: bool = True,
    height: int = 400,
    width: int = 900,
) -> go.Figure:
    """Render neuron frequency cluster heatmap.

    Shows what fraction of each neuron's activation variance is explained
    by each frequency component.

    Args:
        epoch_data: Dict containing 'norm_matrix' array of shape (n_freq, d_mlp).
        epoch: Epoch number (used for title).
        sparse_labels: If True, only show every nth frequency label.
        label_interval: Show label every N frequencies when sparse_labels=True.
        title: Custom title.
        colorscale: Plotly colorscale name.
        show_colorbar: Whether to display the colorbar.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure with heatmap.
    """
    data = epoch_data["norm_matrix"]  # Shape: (n_freq, d_mlp)
    n_freq, d_mlp = data.shape

    # Generate frequency labels (1-indexed as in original)
    all_freq_labels = [str(i + 1) for i in range(n_freq)]

    # For sparse labels, only show every nth
    if sparse_labels:
        y_tickvals = list(range(0, n_freq, label_interval))
        y_ticktext = [all_freq_labels[i] for i in y_tickvals]
    else:
        y_tickvals = list(range(n_freq))
        y_ticktext = all_freq_labels

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=data,
            x=list(range(d_mlp)),
            y=list(range(n_freq)),
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            showscale=show_colorbar,
            hovertemplate="Neuron %{x}<br>"
            + "Freq %{customdata}<br>"
            + "Frac Explained: %{z:.4f}<extra></extra>",
            customdata=[[i + 1 for _ in range(d_mlp)] for i in range(n_freq)],
            colorbar=dict(
                title=dict(text="Frac<br>Explained", side="right"),
                thickness=15,
                len=0.9,
            )
            if show_colorbar
            else None,
        )
    )

    if title is None:
        title = f"Neuron Frequency Specialization - Epoch {epoch}"

    fig.update_layout(
        title=title,
        xaxis_title="Neuron",
        yaxis_title="Frequency",
        yaxis=dict(
            tickvals=y_tickvals,
            ticktext=y_ticktext,
        ),
        height=height,
        width=width,
        template="plotly_white",
        # Minimal margins to maximize data visibility
        margin=dict(l=60, r=80, t=50, b=50),
    )

    return fig


def render_freq_clusters_comparison(
    artifact: dict[str, np.ndarray],
    epoch_indices: list[int],
    sparse_labels: bool = True,
    label_interval: int = 5,
    title: str | None = None,
    colorscale: str = "Viridis",
    subplot_height: int = 300,
) -> go.Figure:
    """Render frequency cluster heatmaps for multiple epochs side by side.

    Args:
        artifact: Dict containing 'epochs' and 'norm_matrix' arrays
            (stacked format from ArtifactLoader.load_epochs()).
        epoch_indices: List of epoch indices to compare.
        sparse_labels: If True, only show every nth frequency label.
        label_interval: Show label every N frequencies.
        title: Main title.
        colorscale: Plotly colorscale name.
        subplot_height: Height of each subplot in pixels.

    Returns:
        Plotly Figure with stacked heatmaps.
    """
    from plotly.subplots import make_subplots

    epochs = artifact["epochs"]
    norm_matrix = artifact["norm_matrix"]
    n_epochs_to_show = len(epoch_indices)

    # Create subplot titles
    subplot_titles = [f"Epoch {int(epochs[idx])}" for idx in epoch_indices]

    fig = make_subplots(
        rows=n_epochs_to_show,
        cols=1,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        shared_xaxes=True,
    )

    n_freq = norm_matrix.shape[1]
    d_mlp = norm_matrix.shape[2]

    # Frequency labels
    all_freq_labels = [str(i + 1) for i in range(n_freq)]
    if sparse_labels:
        y_tickvals = list(range(0, n_freq, label_interval))
        y_ticktext = [all_freq_labels[i] for i in y_tickvals]
    else:
        y_tickvals = list(range(n_freq))
        y_ticktext = all_freq_labels

    for i, epoch_idx in enumerate(epoch_indices):
        data = norm_matrix[epoch_idx]
        epoch = int(epochs[epoch_idx])

        fig.add_trace(
            go.Heatmap(
                z=data,
                x=list(range(d_mlp)),
                y=list(range(n_freq)),
                colorscale=colorscale,
                zmin=0,
                zmax=1,
                showscale=(i == 0),
                hovertemplate=f"Epoch {epoch}<br>"
                + "Neuron %{x}<br>"
                + "Freq %{customdata}<br>"
                + "Frac: %{z:.4f}<extra></extra>",
                customdata=[[j + 1 for _ in range(d_mlp)] for j in range(n_freq)],
            ),
            row=i + 1,
            col=1,
        )

        fig.update_yaxes(
            tickvals=y_tickvals, ticktext=y_ticktext, title_text="Freq", row=i + 1, col=1
        )

    # Only show x-axis label on bottom
    fig.update_xaxes(title_text="Neuron", row=n_epochs_to_show, col=1)

    if title is None:
        title = "Neuron Frequency Specialization Over Training"

    fig.update_layout(
        title=title,
        height=n_epochs_to_show * subplot_height + 100,
        template="plotly_white",
    )

    return fig


def get_specialized_neurons(
    epoch_data: dict[str, np.ndarray],
    frequency: int,
    threshold: float = 0.85,
) -> list[int]:
    """Get neurons specialized for a specific frequency.

    Args:
        epoch_data: Dict containing 'norm_matrix' array of shape (n_freq, d_mlp).
        frequency: Target frequency (1-indexed).
        threshold: Minimum fraction explained to be considered specialized.

    Returns:
        List of neuron indices specialized for the frequency.
    """
    norm_matrix = epoch_data["norm_matrix"]
    freq_idx = frequency - 1  # Convert to 0-indexed

    if freq_idx < 0 or freq_idx >= norm_matrix.shape[0]:
        raise ValueError(f"frequency {frequency} out of valid range")

    specialized = np.where(norm_matrix[freq_idx] > threshold)[0]
    return list(specialized)


def get_neuron_specialization(
    epoch_data: dict[str, np.ndarray],
    neuron_idx: int,
) -> tuple[int, float]:
    """Get the dominant frequency for a specific neuron.

    Args:
        epoch_data: Dict containing 'norm_matrix' array of shape (n_freq, d_mlp).
        neuron_idx: Which neuron to analyze.

    Returns:
        Tuple of (frequency, fraction_explained) where frequency is 1-indexed.
    """
    norm_matrix = epoch_data["norm_matrix"]

    if neuron_idx < 0 or neuron_idx >= norm_matrix.shape[1]:
        raise IndexError(f"neuron_idx {neuron_idx} out of range")

    neuron_data = norm_matrix[:, neuron_idx]
    dominant_freq_idx = int(np.argmax(neuron_data))
    fraction = float(neuron_data[dominant_freq_idx])

    return (dominant_freq_idx + 1, fraction)  # 1-indexed frequency


def render_specialization_trajectory(
    summary_data: dict[str, np.ndarray],
    current_epoch: int,
    title: str | None = None,
    height: int = 300,
) -> go.Figure:
    """Render specialized neuron counts over training.

    Shows total, low-, mid-, and high-frequency specialized neuron counts
    as separate lines, with a vertical indicator at the current epoch.

    Args:
        summary_data: Dict from ArtifactLoader.load_summary("neuron_freq_norm"),
            containing 'epochs', 'specialized_count_total',
            'specialized_count_low', 'specialized_count_mid',
            'specialized_count_high' arrays.
        current_epoch: Current epoch for vertical indicator line.
        title: Custom title.
        height: Figure height in pixels.

    Returns:
        Plotly Figure with line traces and epoch indicator.
    """
    epochs = summary_data["epochs"]
    total = summary_data["specialized_count_total"]
    low = summary_data["specialized_count_low"]
    mid = summary_data["specialized_count_mid"]
    high = summary_data["specialized_count_high"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=total,
            mode="lines",
            name="Total",
            line=dict(color="rgba(31, 119, 180, 1.0)", width=2.5),
            hovertemplate="Epoch %{x}<br>Total Specialized: %{y:.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=low,
            mode="lines",
            name="Low freq",
            line=dict(color="rgba(44, 160, 44, 1.0)", width=1.5),
            hovertemplate="Epoch %{x}<br>Low Freq: %{y:.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=mid,
            mode="lines",
            name="Mid freq",
            line=dict(color="rgba(255, 127, 14, 1.0)", width=1.5),
            hovertemplate="Epoch %{x}<br>Mid Freq: %{y:.0f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=high,
            mode="lines",
            name="High freq",
            line=dict(color="rgba(214, 39, 40, 1.0)", width=1.5),
            hovertemplate="Epoch %{x}<br>High Freq: %{y:.0f}<extra></extra>",
        )
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
        title = "Neuron Specialization Over Training"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Specialized Neuron Count",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(l=60, r=20, t=50, b=50),
    )

    return fig


def render_specialization_by_frequency(
    summary_data: dict[str, np.ndarray],
    current_epoch: int | None = None,
    title: str | None = None,
    colorscale: str = "YlOrRd",
    height: int = 400,
    width: int = 900,
) -> go.Figure:
    """Render per-frequency specialized neuron counts over training as a heatmap.

    Shows how many neurons are specialized in each frequency at each epoch,
    revealing which frequencies gain specialists first during grokking.

    Args:
        summary_data: Dict from ArtifactLoader.load_summary("neuron_freq_norm"),
            containing 'epochs' and 'specialized_count_per_freq' arrays.
        current_epoch: Optional current epoch for vertical indicator.
        title: Custom title.
        colorscale: Plotly colorscale name.
        height: Figure height in pixels.
        width: Figure width in pixels.

    Returns:
        Plotly Figure with heatmap (frequencies × epochs).
    """
    epochs = summary_data["epochs"]
    # Shape: (n_epochs, n_freq) — transpose to (n_freq, n_epochs) for Y=freq, X=epoch
    per_freq = summary_data["specialized_count_per_freq"]
    data = per_freq.T  # (n_freq, n_epochs)
    n_freq = data.shape[0]

    freq_labels = [str(i + 1) for i in range(n_freq)]

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=data,
            x=epochs,
            y=list(range(n_freq)),
            colorscale=colorscale,
            hovertemplate=(
                "Epoch %{x}<br>Freq %{customdata}<br>Specialized: %{z:.0f}<extra></extra>"
            ),
            customdata=[[i + 1 for _ in range(len(epochs))] for i in range(n_freq)],
            colorbar=dict(
                title=dict(text="Count", side="right"),
                thickness=15,
                len=0.9,
            ),
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
        title = "Specialized Neurons by Frequency Over Training"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Frequency",
        yaxis=dict(
            tickvals=list(range(n_freq)),
            ticktext=freq_labels,
        ),
        height=height,
        width=width,
        template="plotly_white",
        margin=dict(l=60, r=80, t=50, b=50),
    )

    return fig
