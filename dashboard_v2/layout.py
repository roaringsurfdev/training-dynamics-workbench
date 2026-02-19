"""Layout components for the Dash dashboard.

Defines the sidebar (controls) and content area (plots) layout.
All 18 Analysis tab visualizations organized by rendering group.

REQ_040: create_layout() now wraps pages in a navbar + URL-driven container.
The visualization layout (sidebar + plots) moved to create_visualization_layout().
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from dashboard_v2.navigation import create_navbar
from miscope.analysis.library.weights import WEIGHT_MATRIX_NAMES
from miscope.visualization.renderers.landscape_flatness import FLATNESS_METRICS

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

_SIDEBAR_STYLE = {
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

_COLLAPSED_SIDEBAR_STYLE = {
    "width": "40px",
    "minWidth": "40px",
    "padding": "10px 5px",
    "backgroundColor": "#f8f9fa",
    "borderRight": "1px solid #dee2e6",
    "height": "100vh",
    "position": "sticky",
    "top": "0",
    "display": "none",
}

_PAGE_CONTENT_STYLE = {
    "flex": "1",
    "padding": "20px",
    "overflowY": "auto",
    "height": "100vh",
}

_FLEX_WRAPPER_STYLE = {
    "display": "flex",
    "height": "calc(100vh - 56px)",
    "overflow": "hidden",
}


def create_sidebar(
    initial_family: str | None = None,
    initial_variant: str | None = None,
) -> html.Div:
    """Create the collapsible sidebar with all controls."""
    return html.Div(
        id="sidebar",
        children=[
            html.Div(
                className="sidebar-header",
                children=[
                    html.H5("Controls", className="mb-0"),
                    dbc.Button(
                        "\u2190",
                        id="sidebar-toggle",
                        size="sm",
                        color="secondary",
                        outline=True,
                        className="ms-auto",
                    ),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
            html.Hr(),
            # Family selector
            dbc.Label("Family", className="fw-bold"),
            dcc.Dropdown(id="family-dropdown", placeholder="Select family...", value=initial_family),
            html.Br(),
            # Variant selector
            dbc.Label("Variant", className="fw-bold"),
            dcc.Dropdown(id="variant-dropdown", placeholder="Select variant...", value=initial_variant),
            html.Br(),
            # Epoch slider
            dbc.Label("Epoch", className="fw-bold"),
            dcc.Slider(
                id="epoch-slider",
                min=0,
                max=1,
                step=1,
                value=0,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(
                id="epoch-display",
                children="Epoch 0 (Index 0)",
                className="text-muted small mb-3",
            ),
            # Neuron slider
            dbc.Label("Neuron Index", className="fw-bold"),
            dcc.Slider(
                id="neuron-slider",
                min=0,
                max=511,
                step=1,
                value=0,
                marks=None,
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.Div(
                id="neuron-display",
                children="Neuron 0",
                className="text-muted small mb-3",
            ),
            html.Hr(),
            # Attention position pair (REQ_025)
            dbc.Label("Attention Relationship", className="fw-bold"),
            dcc.Dropdown(
                id="position-pair-dropdown",
                options=[
                    {"label": "= attending to a", "value": "2,0"},
                    {"label": "= attending to b", "value": "2,1"},
                    {"label": "b attending to a", "value": "1,0"},
                    {"label": "b attending to b", "value": "1,1"},
                    {"label": "a attending to a", "value": "0,0"},
                    {"label": "a attending to b", "value": "0,1"},
                ],
                value="2,0",
                clearable=False,
            ),
            html.Br(),
            # Trajectory component group (REQ_029, REQ_032)
            dbc.Label("Trajectory Group", className="fw-bold"),
            dbc.RadioItems(
                id="trajectory-group-radio",
                options=[
                    {"label": "All", "value": "all"},
                    {"label": "Embedding", "value": "embedding"},
                    {"label": "Attention", "value": "attention"},
                    {"label": "MLP", "value": "mlp"},
                ],
                value="all",
                inline=True,
                className="mb-3",
            ),
            html.Hr(),
            # SV matrix selector (REQ_030)
            dbc.Label("SV Matrix", className="fw-bold"),
            dcc.Dropdown(
                id="sv-matrix-dropdown",
                options=[{"label": name, "value": name} for name in WEIGHT_MATRIX_NAMES],
                value="W_in",
                clearable=False,
            ),
            html.Div(
                id="sv-head-container",
                children=[
                    dbc.Label("Attention Head", className="fw-bold small mt-2"),
                    dcc.Slider(
                        id="sv-head-slider",
                        min=0,
                        max=3,
                        step=1,
                        value=0,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ],
                style={"display": "none"},
            ),
            html.Br(),
            # Flatness metric selector (REQ_031)
            dbc.Label("Flatness Metric", className="fw-bold"),
            dcc.Dropdown(
                id="flatness-metric-dropdown",
                options=[
                    {"label": display, "value": key} for key, display in FLATNESS_METRICS.items()
                ],
                value="mean_delta_loss",
                clearable=False,
            ),
            html.Hr(),
            # Status
            html.Div(
                id="status-display",
                children="No variant selected",
                className="text-muted small",
            ),
        ],
        style=_SIDEBAR_STYLE,
    )


def create_collapsed_sidebar() -> html.Div:
    """Create the collapsed sidebar status bar."""
    return html.Div(
        id="sidebar-collapsed",
        children=[
            dbc.Button(
                "\u2192",
                id="sidebar-expand",
                size="sm",
                color="secondary",
                outline=True,
            ),
            html.Div(
                id="collapsed-status",
                children="",
                className="small text-muted mt-2",
                style={"writingMode": "vertical-lr", "textOrientation": "mixed"},
            ),
        ],
        style=_COLLAPSED_SIDEBAR_STYLE,
    )


def _graph(graph_id: str, height: str = "400px") -> dcc.Graph:
    """Create a dcc.Graph with consistent config."""
    return dcc.Graph(
        id=graph_id,
        config={"displayModeBar": True},
        style={"height": height},
    )


def create_content_area() -> html.Div:
    """Create the scrollable content area with all 18 plot containers.

    Organized by rendering group, matching the Gradio dashboard order.
    """
    return html.Div(
        id="content-area",
        children=[
            html.H4("Training Dynamics Analysis", className="mb-3"),
            # --- Loss ---
            _graph("loss-plot", "350px"),
            # --- Frequency Analysis ---
            _graph("freq-plot", "400px"),
            _graph("activation-plot", "600px"),
            _graph("clusters-plot", "450px"),
            # --- Neuron Specialization (summary, click-to-navigate) ---
            _graph("spec-trajectory-plot", "350px"),
            _graph("spec-freq-plot", "450px"),
            # --- Attention (per-epoch) ---
            _graph("attention-plot", "400px"),
            _graph("attn-freq-plot", "400px"),
            # --- Attention Specialization (summary, click-to-navigate) ---
            _graph("attn-spec-plot", "350px"),
            # --- Trajectory (cross-epoch) ---
            _graph("trajectory-plot", "500px"),
            _graph("trajectory-3d-plot", "600px"),
            _graph("trajectory-pc1-pc3-plot", "500px"),
            _graph("trajectory-pc2-pc3-plot", "500px"),
            _graph("velocity-plot", "400px"),
            # --- Dimensionality (summary + per-epoch) ---
            _graph("dim-trajectory-plot", "400px"),
            _graph("sv-spectrum-plot", "400px"),
            # --- Flatness (summary + per-epoch, click-to-navigate) ---
            _graph("flatness-trajectory-plot", "400px"),
            _graph("perturbation-plot", "400px"),
        ],
        style={
            "flex": "1",
            "padding": "20px",
            "overflowY": "auto",
            "height": "100vh",
        },
    )


def create_visualization_layout(initial: dict | None = None) -> html.Div:
    """Create the visualization page layout (sidebar + plots)."""
    initial = initial or {}
    return html.Div(
        children=[
            create_sidebar(
                initial_family=initial.get("family_name"),
                initial_variant=initial.get("variant_name"),
            ),
            create_collapsed_sidebar(),
            create_content_area(),
        ],
        style=_FLEX_WRAPPER_STYLE,
    )


def create_page_sidebar(
    prefix: str,
    initial_family: str | None = None,
    initial_variant: str | None = None,
    initial_epoch_idx: int = 0,
    extra_controls: list | None = None,
) -> html.Div:
    """Shared collapsible sidebar factory for visualization pages.

    Uses shared toggle IDs (sidebar, sidebar-toggle, etc.) so the existing
    toggle_sidebar callback works on all pages without modification.
    Data component IDs are prefixed to avoid cross-page collision.
    """
    controls: list = [
        html.Div(
            className="sidebar-header",
            children=[
                html.H5("Controls", className="mb-0"),
                dbc.Button(
                    "\u2190",
                    id="sidebar-toggle",
                    size="sm",
                    color="secondary",
                    outline=True,
                    className="ms-auto",
                ),
            ],
            style={"display": "flex", "alignItems": "center"},
        ),
        html.Hr(),
        dbc.Label("Family", className="fw-bold"),
        dcc.Dropdown(
            id=f"{prefix}family-dropdown",
            placeholder="Select family...",
            value=initial_family,
        ),
        html.Br(),
        dbc.Label("Variant", className="fw-bold"),
        dcc.Dropdown(
            id=f"{prefix}variant-dropdown",
            placeholder="Select variant...",
            value=initial_variant,
        ),
        html.Br(),
        dbc.Label("Epoch", className="fw-bold"),
        dcc.Slider(
            id=f"{prefix}epoch-slider",
            min=0,
            max=1,
            step=1,
            value=initial_epoch_idx,
            marks=None,
            tooltip={"placement": "bottom", "always_visible": False},
        ),
        html.Div(
            id=f"{prefix}epoch-display",
            children="Epoch 0",
            className="text-muted small mb-3",
        ),
    ]
    if extra_controls:
        controls += [html.Hr()] + extra_controls
    controls.append(
        html.Div(id=f"{prefix}status", children="No variant selected", className="text-muted small mt-3")
    )
    return html.Div(id="sidebar", children=controls, style=_SIDEBAR_STYLE)


def create_collapsed_page_sidebar() -> html.Div:
    """Collapsed sidebar strip (shared toggle IDs, works with toggle_sidebar callback)."""
    return html.Div(
        id="sidebar-collapsed",
        children=[
            dbc.Button(
                "\u2192",
                id="sidebar-expand",
                size="sm",
                color="secondary",
                outline=True,
            ),
            html.Div(
                id="collapsed-status",
                children="",
                className="small text-muted mt-2",
                style={"writingMode": "vertical-lr", "textOrientation": "mixed"},
            ),
        ],
        style=_COLLAPSED_SIDEBAR_STYLE,
    )


def create_layout() -> html.Div:
    """Create the full application layout with navbar and URL routing."""
    return html.Div(
        [
            dcc.Store(
                id="selection-store",
                storage_type="session",
                data={"family_name": None, "variant_name": None, "epoch": None},
            ),
            dcc.Location(id="url", refresh=False),
            create_navbar(),
            html.Div(id="page-content"),
        ]
    )
