"""Layout components for the Dash dashboard.

Defines the sidebar (controls) and content area (plots) layout.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from visualization.renderers.landscape_flatness import FLATNESS_METRICS


def create_sidebar() -> html.Div:
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
            dcc.Dropdown(id="family-dropdown", placeholder="Select family..."),
            html.Br(),

            # Variant selector
            dbc.Label("Variant", className="fw-bold"),
            dcc.Dropdown(id="variant-dropdown", placeholder="Select variant..."),
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

            # Flatness metric selector
            dbc.Label("Flatness Metric", className="fw-bold"),
            dcc.Dropdown(
                id="flatness-metric-dropdown",
                options=[
                    {"label": display, "value": key}
                    for key, display in FLATNESS_METRICS.items()
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
        style={
            "width": "280px",
            "minWidth": "280px",
            "padding": "20px",
            "backgroundColor": "#f8f9fa",
            "overflowY": "auto",
            "borderRight": "1px solid #dee2e6",
            "height": "100vh",
            "position": "sticky",
            "top": "0",
        },
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
        style={
            "width": "40px",
            "minWidth": "40px",
            "padding": "10px 5px",
            "backgroundColor": "#f8f9fa",
            "borderRight": "1px solid #dee2e6",
            "height": "100vh",
            "position": "sticky",
            "top": "0",
            "display": "none",  # Hidden by default, shown when sidebar collapses
        },
    )


def create_content_area() -> html.Div:
    """Create the scrollable content area with plot containers."""
    return html.Div(
        id="content-area",
        children=[
            html.H4("Training Dynamics Analysis", className="mb-3"),

            # Loss curves (summary, click-to-navigate)
            dcc.Graph(
                id="loss-plot",
                config={"displayModeBar": True},
                style={"height": "350px"},
            ),

            # Specialization trajectory (summary, click-to-navigate)
            dcc.Graph(
                id="spec-trajectory-plot",
                config={"displayModeBar": True},
                style={"height": "350px"},
            ),

            # Flatness trajectory (summary, click-to-navigate)
            dcc.Graph(
                id="flatness-trajectory-plot",
                config={"displayModeBar": True},
                style={"height": "350px"},
            ),

            html.Hr(),

            # Dominant frequencies (per-epoch)
            dcc.Graph(
                id="freq-plot",
                config={"displayModeBar": True},
                style={"height": "400px"},
            ),

            # Neuron activation heatmap (per-epoch)
            dcc.Graph(
                id="activation-plot",
                config={"displayModeBar": True},
                style={"height": "600px"},
            ),

            # Frequency clusters (per-epoch)
            dcc.Graph(
                id="clusters-plot",
                config={"displayModeBar": True},
                style={"height": "450px"},
            ),
        ],
        style={
            "flex": "1",
            "padding": "20px",
            "overflowY": "auto",
            "height": "100vh",
        },
    )


def create_layout() -> html.Div:
    """Create the full application layout."""
    return html.Div(
        children=[
            create_sidebar(),
            create_collapsed_sidebar(),
            create_content_area(),
        ],
        style={
            "display": "flex",
            "height": "100vh",
            "overflow": "hidden",
        },
    )
