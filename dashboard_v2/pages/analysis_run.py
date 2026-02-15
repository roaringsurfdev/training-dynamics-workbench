"""Analysis Run page for the Dash dashboard.

Ports the Gradio Analysis tab run-trigger UI: family/variant selection,
analysis pipeline execution with real-time progress tracking.

REQ_040: Migrate Training & Analysis Run Management to Dash.
"""

from __future__ import annotations

import threading
import traceback

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html, no_update

from dashboard_v2.components.family_selector import get_family_choices, get_variant_choices
from dashboard_v2.state import analysis_progress, get_registry, refresh_registry


def create_analysis_run_layout() -> html.Div:
    """Create the Analysis Run page layout."""
    registry = get_registry()
    family_choices = get_family_choices(registry)
    family_options = [{"label": display, "value": name} for display, name in family_choices]
    default_family = family_options[0]["value"] if family_options else None

    return html.Div(
        children=[
            html.H4("Analysis Run", className="mb-4"),
            dbc.Row(
                [
                    # Left column: selection
                    dbc.Col(
                        [
                            dbc.Label("Model Family", className="fw-bold"),
                            dcc.Dropdown(
                                id="analysis-run-family-dropdown",
                                options=family_options,
                                value=default_family,
                                clearable=False,
                            ),
                            html.Br(),
                            dbc.Label("Variant", className="fw-bold"),
                            dcc.Dropdown(
                                id="analysis-run-variant-dropdown",
                                placeholder="Select variant...",
                            ),
                            html.Br(),
                            dbc.Button(
                                "Refresh Variants",
                                id="analysis-run-refresh-btn",
                                color="secondary",
                                outline=True,
                                size="sm",
                                className="w-100 mb-3",
                            ),
                        ],
                        md=5,
                    ),
                    # Right column: status + button
                    dbc.Col(
                        [
                            html.H6("Status"),
                            dbc.Progress(
                                id="analysis-run-progress-bar",
                                value=0,
                                striped=True,
                                animated=True,
                                className="mb-2",
                                style={"display": "none"},
                            ),
                            html.Div(
                                id="analysis-run-status",
                                children="Select a variant to analyze",
                                className="mb-3",
                                style={
                                    "whiteSpace": "pre-wrap",
                                    "fontFamily": "monospace",
                                    "fontSize": "0.85rem",
                                    "backgroundColor": "#f8f9fa",
                                    "padding": "12px",
                                    "borderRadius": "4px",
                                    "minHeight": "150px",
                                    "maxHeight": "400px",
                                    "overflowY": "auto",
                                },
                            ),
                            dbc.Button(
                                "Run Analysis",
                                id="analysis-run-start-btn",
                                color="primary",
                                className="w-100",
                            ),
                        ],
                        md=7,
                    ),
                ],
                className="g-4",
            ),
            # Progress polling interval (disabled by default)
            dcc.Interval(
                id="analysis-run-interval",
                interval=500,
                disabled=True,
            ),
        ],
        style={"padding": "20px", "maxWidth": "1000px"},
    )


def _run_analysis_thread(family_name: str, variant_name: str) -> None:
    """Execute analysis pipeline in a background thread."""
    from miscope.analysis import AnalysisPipeline
    from miscope.analysis.analyzers import (
        AttentionFreqAnalyzer,
        AttentionPatternsAnalyzer,
        DominantFrequenciesAnalyzer,
        EffectiveDimensionalityAnalyzer,
        LandscapeFlatnessAnalyzer,
        NeuronActivationsAnalyzer,
        NeuronFreqClustersAnalyzer,
        ParameterSnapshotAnalyzer,
        ParameterTrajectoryPCA,
    )

    try:
        analysis_progress.update(0.05, "Initializing...")

        registry = get_registry()
        family = registry.get_family(family_name)
        variants = registry.get_variants(family)

        variant = None
        for v in variants:
            if v.name == variant_name:
                variant = v
                break

        if variant is None:
            analysis_progress.finish(f"Variant '{variant_name}' not found")
            return

        analysis_progress.update(0.1, "Starting analysis pipeline...")

        def progress_callback(pct: float, desc: str) -> None:
            ui_progress = 0.1 + (pct * 0.9)
            analysis_progress.update(ui_progress, desc)

        pipeline = AnalysisPipeline(variant)
        pipeline.register(AttentionFreqAnalyzer())
        pipeline.register(AttentionPatternsAnalyzer())
        pipeline.register(DominantFrequenciesAnalyzer())
        pipeline.register(NeuronActivationsAnalyzer())
        pipeline.register(NeuronFreqClustersAnalyzer())
        pipeline.register(ParameterSnapshotAnalyzer())
        pipeline.register(EffectiveDimensionalityAnalyzer())
        pipeline.register(LandscapeFlatnessAnalyzer())
        pipeline.register_cross_epoch(ParameterTrajectoryPCA())
        pipeline.run(progress_callback=progress_callback)

        refresh_registry()

        analysis_progress.finish(f"Analysis complete!\nArtifacts saved to {variant.artifacts_dir}")

    except Exception as e:
        analysis_progress.finish(f"Analysis failed: {e}\n\n{traceback.format_exc()}")


def register_analysis_callbacks(app: Dash) -> None:
    """Register all Analysis Run page callbacks."""

    # --- Family change → populate variant dropdown ---
    @app.callback(
        Output("analysis-run-variant-dropdown", "options"),
        Output("analysis-run-variant-dropdown", "value"),
        Input("analysis-run-family-dropdown", "value"),
    )
    def on_analysis_family_change(
        family_name: str | None,
    ) -> tuple[list, None]:
        if not family_name:
            return [], None

        registry = get_registry()
        choices = get_variant_choices(registry, family_name)
        options = [{"label": display, "value": name} for display, name in choices]
        return options, None

    # --- Refresh button → reload variants ---
    @app.callback(
        Output("analysis-run-variant-dropdown", "options", allow_duplicate=True),
        Output("analysis-run-variant-dropdown", "value", allow_duplicate=True),
        Input("analysis-run-refresh-btn", "n_clicks"),
        State("analysis-run-family-dropdown", "value"),
        prevent_initial_call=True,
    )
    def on_refresh_variants(
        n_clicks: int | None,
        family_name: str | None,
    ):  # noqa: ANN202 — Dash callbacks return no_update
        if not n_clicks or not family_name:
            return no_update, no_update

        refresh_registry()
        registry = get_registry()
        choices = get_variant_choices(registry, family_name)
        options = [{"label": display, "value": name} for display, name in choices]
        return options, None

    # --- Start Analysis button → launch thread + enable interval ---
    @app.callback(
        Output("analysis-run-interval", "disabled"),
        Output("analysis-run-start-btn", "disabled"),
        Output("analysis-run-status", "children", allow_duplicate=True),
        Output("analysis-run-progress-bar", "style", allow_duplicate=True),
        Input("analysis-run-start-btn", "n_clicks"),
        State("analysis-run-family-dropdown", "value"),
        State("analysis-run-variant-dropdown", "value"),
        prevent_initial_call=True,
    )
    def on_start_analysis(
        n_clicks: int | None,
        family_name: str | None,
        variant_name: str | None,
    ) -> tuple:
        if not n_clicks:
            return no_update, no_update, no_update, no_update

        if not family_name or not variant_name:
            return no_update, no_update, "Please select a family and variant", no_update

        state = analysis_progress.get_state()
        if state["running"]:
            return no_update, no_update, "Analysis already in progress...", no_update

        analysis_progress.start()

        thread = threading.Thread(
            target=_run_analysis_thread,
            args=(family_name, variant_name),
            daemon=True,
        )
        thread.start()

        return False, True, "Starting analysis...", {"display": "block"}

    # --- Interval → poll progress ---
    @app.callback(
        Output("analysis-run-progress-bar", "value"),
        Output("analysis-run-progress-bar", "label"),
        Output("analysis-run-status", "children"),
        Output("analysis-run-interval", "disabled", allow_duplicate=True),
        Output("analysis-run-start-btn", "disabled", allow_duplicate=True),
        Output("analysis-run-progress-bar", "style"),
        Input("analysis-run-interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def poll_analysis_progress(n_intervals: int) -> tuple:
        state = analysis_progress.get_state()
        pct = int(state["progress"] * 100)

        if state["running"]:
            return (
                pct,
                f"{pct}%",
                state["message"],
                False,
                True,
                {"display": "block"},
            )

        # Job finished
        return (
            100,
            "100%",
            state["result"] if state["result"] else state["message"],
            True,
            False,
            {"display": "none"},
        )
