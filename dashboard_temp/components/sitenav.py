"""Site-level navigation for the Dash dashboard.

Provides a top navbar with links to Visualization, Training, and Analysis Run
pages, plus URL-driven routing callback.
"""

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html, set_props

from dashboard_v2.version import __version__


def create_sitenav() -> dbc.NavbarSimple:
    """Create the top navigation bar."""
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Summary", href="/summary")),
            dbc.NavItem(dbc.NavLink("Neuron Dynamics", href="/neuron-dynamics")),
            dbc.NavItem(dbc.NavLink("Repr Geometry", href="/repr-geometry")),
        ],
        brand=f"MechInterp Scope v{__version__}",
        brand_href="/",
        color="dark",
        dark=True,
        sticky="fixed",
        fluid=True,
    )


def register_sitenav_callbacks(app: Dash) -> None:
    print("register_routing_callbacks")
    """Register URL routing callback."""
    from dashboard_temp.pages.neuron_dynamics import (
        create_neuron_dynamics_page_layout,
        create_neuron_dynamics_page_nav,
    )
    from dashboard_temp.pages.repr_geometry import (
        create_repr_geometry_page_layout,
        create_repr_geometry_page_nav,
    )
    from dashboard_temp.pages.summary import create_summary_page_layout

    @app.callback(
        Output("page_left_nav", "children"),
        Input("url", "pathname"),
    )
    def display_page_nav(pathname: str | None) -> html.Div:
        print("display_page_nav")
        if pathname == "/neuron-dynamics":
            return create_neuron_dynamics_page_nav()
        elif pathname == "/repr-geometry":
            return create_repr_geometry_page_nav()
        else:
            return html.Div()

    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname"),
        State("variant-selector-store", "data"),
    )
    def display_page(pathname: str | None, variant_data: dict | None) -> html.Div:
        print("display_page")
        set_props("variant-selector-store", {"data": {"stale_data": "1"}})
        if pathname == "/neuron-dynamics":
            return create_neuron_dynamics_page_layout(variant_data)
        elif pathname == "/repr-geometry":
            return create_repr_geometry_page_layout(variant_data)
        elif pathname == "/summary":
            return create_summary_page_layout(variant_data)
        else:
            return create_summary_page_layout(variant_data)
