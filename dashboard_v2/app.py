"""Dash dashboard application for training dynamics analysis.

REQ_035: Visualization with sidebar, click-to-navigate, selective rendering.
REQ_040: Training and Analysis Run pages with site-level navigation.

Launch: python -m dashboard_v2.app
Runs on port 8050 (separate from Gradio on 7860).
"""

import dash_bootstrap_components as dbc
from dash import Dash

from dashboard_v2.callbacks import register_callbacks
from dashboard_v2.layout import create_layout
from dashboard_v2.navigation import register_routing_callbacks
from dashboard_v2.pages.analysis_run import register_analysis_callbacks
from dashboard_v2.pages.neuron_dynamics import register_neuron_dynamics_callbacks
from dashboard_v2.pages.repr_geometry import register_repr_geometry_callbacks
from dashboard_v2.pages.summary import register_summary_callbacks
from dashboard_v2.pages.training import register_training_callbacks


def create_app() -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )
    app.title = "Training Dynamics Workbench"
    app.layout = create_layout()
    register_routing_callbacks(app)
    register_callbacks(app)
    register_summary_callbacks(app)
    register_neuron_dynamics_callbacks(app)
    register_repr_geometry_callbacks(app)
    register_training_callbacks(app)
    register_analysis_callbacks(app)
    return app


def main() -> None:
    """Entry point for the Dash dashboard."""
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
