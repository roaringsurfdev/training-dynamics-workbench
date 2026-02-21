from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc

def create_default_layout(initial: dict | None = None) -> html.Div:
    return html.Div(
        children="Default Page"
    )