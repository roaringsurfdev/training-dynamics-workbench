import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, html

from dashboard.components.analysis_page import AnalysisPageGraphManager

# TODO: Standardize _VIEW_LIST to use a shared schema across pages
_VIEW_LIST = {
    "loss-plot": {"view_name": "loss_curve", "view_type": "epoch_selector"},
    "freq-over-time-plot": {
        "view_name": "dominant_frequencies_over_time",
        "view_type": "epoch_selector",
    },
    "spec-trajectory-plot": {
        "view_name": "specialization_trajectory",
        "view_type": "default_graph",
    },
    "spec-freq-plot": {
        "view_name": "specialization_by_frequency",
        "view_type": "epoch_selector",
    },
    "attn-spec-plot": {
        "view_name": "attention_specialization_trajectory",
        "view_type": "default_graph",
    },
    "attn-dom-freq-plot": {
        "view_name": "attention_dominant_frequencies",
        "view_type": "default_graph",
    },
    "trajectory-3d-plot": {"view_name": "trajectory_3d", "view_type": "default_graph"},
    "trajectory-plot": {"view_name": "parameter_trajectory", "view_type": "default_graph"},
    "trajectory-pc1-pc3-plot": {
        "view_name": "trajectory_pc1_pc3",
        "view_type": "default_graph",
    },
    "trajectory-pc2-pc3-plot": {
        "view_name": "trajectory_pc2_pc3",
        "view_type": "default_graph",
    },
    "velocity-plot": {"view_name": "parameter_velocity", "view_type": "default_graph"},
    "dim-trajectory-plot": {
        "view_name": "dimensionality_trajectory",
        "view_type": "default_graph",
    },
}

_graph_manager = AnalysisPageGraphManager(_VIEW_LIST, "sum")


def create_summary_page_nav() -> html.Div:
    print("create_summary_page_nav")
    return html.Div()


def create_summary_page_layout() -> html.Div:
    print("create_summary_page_layout")
    # set_props("variant-selector-store", {"data": {"stale_data": "1"}})
    return html.Div(
        children=[
            # Loss curve (full width)
            dbc.Row(dbc.Col(_graph_manager.create_graph("loss-plot", "300px"))),
            # Embedding Fourier over time (full width)
            dbc.Row(dbc.Col(_graph_manager.create_graph("freq-over-time-plot", "350px"))),
            # Neuron specialization | Attention head specialization
            dbc.Row(
                [
                    dbc.Col(
                        _graph_manager.create_graph("spec-trajectory-plot", "350px"),
                        width=7,
                    ),
                    dbc.Col(
                        _graph_manager.create_graph("attn-spec-plot", "350px"), width=5
                    ),
                ]
            ),
            # Specialized neurons by frequency (full width)
            dbc.Row(dbc.Col(_graph_manager.create_graph("spec-freq-plot", "400px"))),
            # Attention dominant frequencies (full width)
            dbc.Row(dbc.Col(_graph_manager.create_graph("attn-dom-freq-plot", "300px"))),
            # Trajectory 3D (full width)
            dbc.Row(dbc.Col(_graph_manager.create_graph("trajectory-3d-plot", "550px"))),
            # PC1/PC2 | PC1/PC3 | PC2/PC3
            dbc.Row(
                [
                    dbc.Col(
                        _graph_manager.create_graph("trajectory-plot", "400px"), width=4
                    ),
                    dbc.Col(
                        _graph_manager.create_graph("trajectory-pc1-pc3-plot", "400px"),
                        width=4,
                    ),
                    dbc.Col(
                        _graph_manager.create_graph("trajectory-pc2-pc3-plot", "400px"),
                        width=4,
                    ),
                ]
            ),
            # Component velocity | Effective dimensionality
            dbc.Row(
                [
                    dbc.Col(_graph_manager.create_graph("velocity-plot", "350px"), width=6),
                    dbc.Col(
                        _graph_manager.create_graph("dim-trajectory-plot", "350px"), width=6
                    ),
                ]
            ),
        ],
    )


def register_summary_page_callbacks(app: Dash) -> None:
    """Register all callbacks for the Summary page."""
    print("register_summary_page_callbacks")

    @app.callback(
        *[Output(pid, "figure") for pid in _graph_manager.get_graph_output_list()],
        Input("variant-selector-store", "modified_timestamp"),
        State("variant-selector-store", "data"),
    )
    def on_summary_data_change(modified_timestamp: str | None, variant_data: dict | None):
        print("on_summary_data_change")
        return _graph_manager.update_graphs(variant_data)
