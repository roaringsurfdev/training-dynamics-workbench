"""Input trace renderers — REQ_075.

Three views for the per-input prediction trace:
  - accuracy_grid: p×p heatmap of correct/incorrect at a single epoch
  - residue_class_accuracy_timeline: per-class accuracy across training
  - pair_graduation_heatmap: p×p map of graduation epochs
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go


def render_accuracy_grid(
    data: dict[str, Any],
    epoch: int | None,
    **kwargs: Any,
) -> go.Figure:
    """p×p heatmap of correct/incorrect predictions at a single checkpoint.

    Training pairs correct = blue, incorrect = light gray, test pairs = white.
    Anti-diagonal structure (a+b = constant) is visible when the model is
    learning by residue class.

    Args:
        data: Dict with 'epoch_data' (from input_trace per-epoch artifact) and 'prime'
        epoch: Checkpoint epoch for the title
    """
    prime = int(data["prime"])
    epoch_data = data["epoch_data"]
    pair_indices = epoch_data["pair_indices"].astype(np.int32)
    correct = epoch_data["correct"]

    # 0 = test pair (white), 1 = training incorrect (light gray), 2 = training correct (blue)
    grid = np.zeros((prime, prime), dtype=np.float32)
    a_coords = pair_indices[:, 0]
    b_coords = pair_indices[:, 1]
    grid[b_coords, a_coords] = 1.0
    grid[b_coords[correct], a_coords[correct]] = 2.0

    colorscale = [
        [0.0, "white"],
        [0.49, "white"],
        [0.5, "#cccccc"],
        [0.74, "#cccccc"],
        [0.75, "#1f77b4"],
        [1.0, "#1f77b4"],
    ]

    epoch_label = epoch if epoch is not None else "?"
    n_correct = int(correct.sum())
    n_pairs = len(correct)
    accuracy = n_correct / n_pairs if n_pairs > 0 else 0.0

    fig = go.Figure(
        go.Heatmap(
            z=grid,
            colorscale=colorscale,
            zmin=0,
            zmax=2,
            showscale=False,
            hovertemplate="a=%{x}, b=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Training Pair Accuracy — Epoch {epoch_label} ({accuracy:.1%} correct)",
        xaxis_title="a",
        yaxis_title="b",
        template="plotly_white",
    )
    return fig


def render_residue_class_accuracy_timeline(
    data: dict[str, Any],
    epoch: int | None,
    **kwargs: Any,
) -> go.Figure:
    """Line plot of per-residue-class accuracy across all checkpoints.

    One trace per residue class c (0 to p-1), colored circularly.
    A staircase pattern where classes graduate in discrete steps is the
    predicted attractor signature of frequency-based learning.

    GCD blind spots: residue classes that are multiples of gcd(committed
    frequency, p) may graduate together — watch for grouped staircases
    corresponding to the model's committed frequency set.

    Args:
        data: Dict with 'summary' (from input_trace summary.npz) and 'prime'
        epoch: Cursor epoch (shown as vertical line)
    """
    prime = int(data["prime"])
    summary = data["summary"]
    epochs_arr = summary["epochs"]
    residue_accuracy = summary["residue_class_accuracy"]  # (n_epochs, p)
    overall_accuracy = summary["overall_accuracy"]         # (n_epochs,)

    # Circular colormap via HSV
    colors = [
        f"hsl({int(360 * c / prime)}, 70%, 50%)" for c in range(prime)
    ]

    fig = go.Figure()

    for c in range(prime):
        fig.add_trace(
            go.Scatter(
                x=epochs_arr,
                y=residue_accuracy[:, c],
                mode="lines",
                line=dict(color=colors[c], width=1),
                name=f"c={c}",
                showlegend=False,
                hovertemplate=f"c={c}: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=epochs_arr,
            y=overall_accuracy,
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            name="Overall",
            hovertemplate="Overall: %{y:.2f}<extra></extra>",
        )
    )

    if epoch is not None:
        fig.add_vline(
            x=epoch,
            line=dict(color="red", width=1, dash="dot"),
        )

    fig.update_layout(
        title="Residue Class Accuracy by Epoch",
        xaxis_title="Epoch",
        yaxis_title="Fraction Correct",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
    )
    return fig


def render_pair_graduation_heatmap(
    data: dict[str, Any],
    epoch: int | None,
    **kwargs: Any,
) -> go.Figure:
    """p×p heatmap of graduation epochs for each training pair.

    Cell color encodes the epoch at which pair (a, b) first achieved
    stable correctness. Never-graduated pairs (graduation_epoch = -1) are gray.
    Test pairs (absent from pair_indices) are white.

    Anti-diagonal structure: if the model learns by residue class, pairs
    with the same sum (a+b = constant) should share similar graduation epochs,
    producing uniformly-colored anti-diagonals.

    Args:
        data: Dict with 'graduation' (from input_trace_graduation cross_epoch) and 'prime'
        epoch: Unused (summary view; no cursor)
    """
    prime = int(data["prime"])
    graduation = data["graduation"]
    graduation_epochs = graduation["graduation_epochs"].astype(np.float32)
    pair_indices = graduation["pair_indices"].astype(np.int32)

    NEVER_GRAD = -999.0
    TEST_PAIR = np.nan

    grid = np.full((prime, prime), TEST_PAIR, dtype=np.float32)
    a_coords = pair_indices[:, 0]
    b_coords = pair_indices[:, 1]

    for i, (a, b) in enumerate(zip(a_coords, b_coords)):
        g = graduation_epochs[i]
        grid[b, a] = NEVER_GRAD if g == -1 else g

    graduated_vals = graduation_epochs[graduation_epochs >= 0]
    vmin = float(graduated_vals.min()) if len(graduated_vals) > 0 else 0
    vmax = float(graduated_vals.max()) if len(graduated_vals) > 0 else 1

    colorscale = [
        [0.0, "#aaaaaa"],       # never-graduated (gray)
        [0.001, "#aaaaaa"],
        [0.001, "navy"],        # earliest graduation (dark)
        [1.0, "lightyellow"],   # latest graduation (light)
    ]

    # Remap: NEVER_GRAD → 0, graduated → rescaled to [0.001, 1.0]
    epoch_range = vmax - vmin if vmax > vmin else 1.0
    grid_remapped = np.where(
        np.isnan(grid),
        TEST_PAIR,
        np.where(
            grid == NEVER_GRAD,
            0.0,
            0.001 + 0.999 * (grid - vmin) / epoch_range,
        ),
    )

    fig = go.Figure(
        go.Heatmap(
            z=grid_remapped,
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar=dict(
                title="Graduation Epoch",
                tickvals=[0.0, 0.001, 0.5, 1.0],
                ticktext=["Never", str(int(vmin)), str(int((vmin + vmax) / 2)), str(int(vmax))],
            ),
            hovertemplate="a=%{x}, b=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Pair Graduation Epochs",
        xaxis_title="a",
        yaxis_title="b",
        template="plotly_white",
    )
    return fig
