"""Dash dashboard application for training dynamics analysis.

REQ_035 Phase 1: Spike — validates sidebar layout, click-to-navigate,
selective rendering with Patch(), and per-output callbacks.

Launch: python -m dashboard_v2.app
Runs on port 8050 (separate from Gradio on 7860).
"""

import dash_bootstrap_components as dbc
from dash import Dash

from dashboard_v2.callbacks import register_callbacks
from dashboard_v2.layout import create_layout


def create_app() -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )
    app.title = "Training Dynamics Workbench"
    app.layout = create_layout()
    register_callbacks(app)
    return app


def main() -> None:
    """Entry point for the Dash dashboard."""
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
