"""Site-level navigation for the Dash dashboard.

Provides a top navbar with links to Visualization, Training, and Analysis Run
pages, plus URL-driven routing callback.

REQ_040: Migrate Training & Analysis Run Management to Dash.
"""

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, html

from dashboard_v2.version import __version__


def create_navbar() -> dbc.NavbarSimple:
    """Create the top navigation bar."""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Visualization", href="/")),
            dbc.NavItem(dbc.NavLink("Summary", href="/summary")),
            dbc.NavItem(dbc.NavLink("Neuron Dynamics", href="/neuron-dynamics")),
            dbc.NavItem(dbc.NavLink("Training", href="/training")),
            dbc.NavItem(dbc.NavLink("Analysis Run", href="/analysis-run")),
        ],
        brand=f"Training Dynamics Workbench v{__version__}",
        brand_href="/",
        color="dark",
        dark=True,
        sticky="top",
    )


def register_routing_callbacks(app: Dash) -> None:
    """Register URL routing callback."""
    from dashboard_v2.layout import create_visualization_layout
    from dashboard_v2.pages.analysis_run import create_analysis_run_layout
    from dashboard_v2.pages.neuron_dynamics import create_neuron_dynamics_layout
    from dashboard_v2.pages.summary import create_summary_layout
    from dashboard_v2.pages.training import create_training_layout

    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
    )
    def display_page(pathname: str | None) -> html.Div:
        if pathname == "/training":
            return create_training_layout()
        elif pathname == "/analysis-run":
            return create_analysis_run_layout()
        elif pathname == "/summary":
            return create_summary_layout()
        elif pathname == "/neuron-dynamics":
            return create_neuron_dynamics_layout()
        return create_visualization_layout()
