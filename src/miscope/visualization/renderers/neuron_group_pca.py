"""Neuron Group PCA renderers.

Two views into within-frequency-group coordination in weight space:
- pca_cohesion: PC1 variance explained per group over epochs
- spread: mean L2 distance from group centroid per group over epochs

High PC1 var → neurons in the group are aligned (coordinated unit).
Low PC1 var → neurons with the same dominant frequency are spread across
multiple weight-space directions (diffuse, independent).
"""

import colorsys

import numpy as np
import plotly.graph_objects as go


def _freq_color(freq_idx: int, n_freq: int) -> str:
    """Consistent HSL color for frequency index (0-indexed)."""
    hue = freq_idx / max(n_freq, 1)
    r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.5)
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def render_neuron_group_pca_cohesion(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """Line plot of PC1 variance explained per frequency group over epochs.

    Each line is one frequency group. PC1 var explained near 1.0 means
    all neurons in the group are pointing in the same direction in weight
    space — they move as a coordinated unit. Low values indicate diffuse,
    independent neuron trajectories within the group.

    Args:
        data: cross_epoch artifact from neuron_group_pca
        epoch: optional epoch cursor (vertical line)
    """
    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    group_sizes = data["group_sizes"]
    pc1_var = data["pc1_var"]  # (n_epochs, n_groups)
    n_freq = int(group_freqs.max()) + 1 if len(group_freqs) > 0 else 1

    fig = go.Figure()

    for g_idx, (freq, size) in enumerate(zip(group_freqs, group_sizes)):
        color = _freq_color(int(freq), n_freq)
        y = pc1_var[:, g_idx].tolist()
        fig.add_trace(
            go.Scatter(
                x=epochs.tolist(),
                y=y,
                mode="lines",
                name=f"freq {freq} (n={size})",
                line=dict(color=color, width=1.5),
                hovertemplate=f"freq={freq} n={size}<br>epoch=%{{x}}<br>PC1 var=%{{y:.3f}}<extra></extra>",
            )
        )

    if epoch is not None:
        fig.add_vline(x=epoch, line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"))

    fig.update_layout(
        title="Within-group PC1 variance explained (W_in)",
        xaxis_title="Epoch",
        yaxis_title="PC1 variance explained",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=20, t=50, b=60),
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            font=dict(size=10),
        ),
    )
    return fig


def render_neuron_group_spread(
    data: dict,
    epoch: int | None = None,
    **kwargs,
) -> go.Figure:
    """Line plot of mean within-group L2 spread per frequency group over epochs.

    Each line is one frequency group. Low spread means group neurons have
    converged toward a common weight vector. High and rising spread means
    the group is expanding in weight space.

    Args:
        data: cross_epoch artifact from neuron_group_pca
        epoch: optional epoch cursor (vertical line)
    """
    epochs = data["epochs"]
    group_freqs = data["group_freqs"]
    group_sizes = data["group_sizes"]
    mean_spread = data["mean_spread"]  # (n_epochs, n_groups)
    n_freq = int(group_freqs.max()) + 1 if len(group_freqs) > 0 else 1

    fig = go.Figure()

    for g_idx, (freq, size) in enumerate(zip(group_freqs, group_sizes)):
        color = _freq_color(int(freq), n_freq)
        y = mean_spread[:, g_idx].tolist()
        fig.add_trace(
            go.Scatter(
                x=epochs.tolist(),
                y=y,
                mode="lines",
                name=f"freq {freq} (n={size})",
                line=dict(color=color, width=1.5),
                hovertemplate=f"freq={freq} n={size}<br>epoch=%{{x}}<br>spread=%{{y:.4f}}<extra></extra>",
            )
        )

    if epoch is not None:
        fig.add_vline(x=epoch, line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dash"))

    fig.update_layout(
        title="Within-group mean L2 spread (W_in)",
        xaxis_title="Epoch",
        yaxis_title="Mean L2 distance from group centroid",
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=20, t=50, b=60),
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            font=dict(size=10),
        ),
    )
    return fig
