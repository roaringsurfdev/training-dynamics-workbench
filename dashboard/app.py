"""Main Gradio dashboard application.

REQ_007: Training controls
REQ_008: Analysis and synchronized visualizations
REQ_009: Loss curves with epoch indicator
REQ_020: Checkpoint epoch-index display
"""

import json
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
import torch

from analysis import AnalysisPipeline, ArtifactLoader
from analysis.analyzers import (
    DominantFrequenciesAnalyzer,
    NeuronActivationsAnalyzer,
    NeuronFreqClustersAnalyzer,
)
from dashboard.components.loss_curves import render_loss_curves_with_indicator
from dashboard.state import DashboardState
from dashboard.utils import (
    discover_trained_models,
    get_model_choices,
    parse_checkpoint_epochs,
    validate_training_params,
)
from dashboard.version import __version__
from ModuloAdditionSpecification import ModuloAdditionSpecification
from visualization import (
    render_dominant_frequencies,
    render_freq_clusters,
    render_neuron_heatmap,
)


def create_empty_plot(message: str = "No data") -> go.Figure:
    """Create an empty Plotly figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(template="plotly_white", height=300)
    return fig


# ============================================================================
# Training Tab Handlers (REQ_007)
# ============================================================================


def run_training(
    modulus: int,
    model_seed: int,
    data_seed: int,
    train_fraction: float,
    num_epochs: int,
    checkpoint_str: str,
    save_path: str,
    progress=gr.Progress(),
) -> str:
    """Execute training with the given parameters."""
    # Validate inputs
    is_valid, error_msg = validate_training_params(
        int(modulus),
        int(model_seed),
        int(data_seed),
        train_fraction,
        int(num_epochs),
        checkpoint_str,
        save_path,
    )
    if not is_valid:
        return f"Error: {error_msg}"

    try:
        # Parse checkpoint epochs
        checkpoint_epochs = parse_checkpoint_epochs(checkpoint_str)
        if not checkpoint_epochs:
            checkpoint_epochs = None  # Use defaults

        progress(0, desc="Initializing model...")

        spec = ModuloAdditionSpecification(
            model_dir=save_path,
            prime=int(modulus),
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=int(model_seed),
            data_seed=int(data_seed),
            training_fraction=train_fraction,
        )

        progress(0.1, desc="Starting training...")

        # Training (synchronous for MVP)
        spec.train(num_epochs=int(num_epochs), checkpoint_epochs=checkpoint_epochs)

        progress(1.0, desc="Training complete!")

        return (
            f"Training complete!\n"
            f"Model saved to: {spec.full_dir}\n"
            f"Checkpoints: {spec.checkpoint_epochs}\n"
            f"Final train loss: {spec.train_losses[-1]:.6f}\n"
            f"Final test loss: {spec.test_losses[-1]:.6f}"
        )

    except Exception as e:
        return f"Training failed: {e}"


# ============================================================================
# Analysis Tab Handlers (REQ_008)
# ============================================================================


def refresh_models(base_path: str) -> gr.Dropdown:
    """Refresh the model dropdown with discovered models."""
    models = discover_trained_models(base_path)
    choices = get_model_choices(models)
    return gr.Dropdown(choices=choices, value=None)


def load_model_data(model_path: str | None, state: DashboardState):
    """Load model metadata and artifacts when a model is selected."""
    if not model_path:
        state.clear_artifacts()
        return (
            state,
            gr.Slider(minimum=0, maximum=1, value=0),
            create_empty_plot("Select a model"),
            create_empty_plot("Select a model"),
            create_empty_plot("Select a model"),
            create_empty_plot("Select a model"),
            "No model selected",
            "Epoch 0 (Index 0)",  # REQ_020: initial epoch display
            gr.Slider(minimum=0, maximum=511, value=0, step=1),
        )

    state.selected_model_path = model_path
    state.clear_artifacts()

    # Load metadata for loss curves
    metadata_path = Path(model_path) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        state.train_losses = metadata.get("train_losses", [])
        state.test_losses = metadata.get("test_losses", [])

    # Load config
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            state.model_config = json.load(f)

    # Check for artifacts
    artifacts_dir = Path(model_path) / "artifacts"
    if artifacts_dir.exists():
        try:
            loader = ArtifactLoader(str(artifacts_dir))
            available = loader.get_available_analyzers()

            if "dominant_frequencies" in available:
                state.dominant_freq_artifact = loader.load("dominant_frequencies")
                state.available_epochs = list(state.dominant_freq_artifact["epochs"])

            if "neuron_activations" in available:
                state.neuron_activations_artifact = loader.load("neuron_activations")
                state.n_neurons = state.neuron_activations_artifact["activations"].shape[1]

            if "neuron_freq_norm" in available:
                state.freq_clusters_artifact = loader.load("neuron_freq_norm")

        except Exception:
            pass  # Artifacts not available yet

    # Generate initial plots
    state.current_epoch_idx = 0
    plots = generate_all_plots(state)

    # Configure slider
    max_idx = max(0, len(state.available_epochs) - 1)

    status = f"Loaded: p={(state.model_config or {}).get('prime', '?')}"
    if state.available_epochs:
        status += f", {len(state.available_epochs)} checkpoints"
    else:
        status += " (no analysis yet)"

    # REQ_020: format initial epoch display with index
    initial_epoch = state.get_current_epoch()
    epoch_display_text = format_epoch_display(initial_epoch, 0)

    return (
        state,
        gr.Slider(minimum=0, maximum=max_idx, value=0, step=1),
        plots[0],  # loss
        plots[1],  # freq
        plots[2],  # activation
        plots[3],  # clusters
        status,
        epoch_display_text,
        gr.Slider(minimum=0, maximum=state.n_neurons - 1, value=0, step=1),
    )


def run_analysis(model_path: str | None, state: DashboardState, progress=gr.Progress()):
    """Run analysis pipeline on the selected model."""
    if not model_path:
        return "No model selected", state

    try:
        progress(0, desc="Initializing analysis...")

        # Load model spec
        config_path = Path(model_path) / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        spec = ModuloAdditionSpecification(
            model_dir=str(Path(model_path).parent.parent),
            prime=config.get("prime", config.get("n_ctx")),
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=config.get("model_seed", config.get("seed", 999)),
        )

        progress(0.1, desc="Starting analysis pipeline...")

        # Create progress callback that maps pipeline progress (0-1) to UI progress (0.1-1.0)
        def pipeline_progress(pct: float, desc: str):
            # Map pipeline 0-1 to UI 0.1-1.0 (reserving 0-0.1 for init)
            ui_progress = 0.1 + (pct * 0.9)
            progress(ui_progress, desc=desc)

        # Run analysis with progress callback
        pipeline = AnalysisPipeline(spec)
        pipeline.register(DominantFrequenciesAnalyzer())
        pipeline.register(NeuronActivationsAnalyzer())
        pipeline.register(NeuronFreqClustersAnalyzer())
        pipeline.run(progress_callback=pipeline_progress)

        progress(1.0, desc="Analysis complete!")

        return f"Analysis complete! Artifacts saved to {spec.artifacts_dir}", state

    except Exception as e:
        return f"Analysis failed: {e}", state


def generate_all_plots(state: DashboardState):
    """Generate all 4 visualization plots for current state."""
    epoch_idx = state.current_epoch_idx
    epoch = state.get_current_epoch()

    # Loss curves
    loss_fig = render_loss_curves_with_indicator(
        state.train_losses,
        state.test_losses,
        current_epoch=epoch,
        checkpoint_epochs=state.available_epochs,
    )

    # Dominant frequencies
    if state.dominant_freq_artifact is not None:
        freq_fig = render_dominant_frequencies(
            state.dominant_freq_artifact,
            epoch_idx=epoch_idx,
            threshold=1.0,
        )
    else:
        freq_fig = create_empty_plot("Run analysis first")

    # Neuron activation
    if state.neuron_activations_artifact is not None:
        activation_fig = render_neuron_heatmap(
            state.neuron_activations_artifact,
            epoch_idx=epoch_idx,
            neuron_idx=state.selected_neuron,
        )
    else:
        activation_fig = create_empty_plot("Run analysis first")

    # Frequency clusters
    if state.freq_clusters_artifact is not None:
        clusters_fig = render_freq_clusters(
            state.freq_clusters_artifact,
            epoch_idx=epoch_idx,
        )
    else:
        clusters_fig = create_empty_plot("Run analysis first")

    return loss_fig, freq_fig, activation_fig, clusters_fig


def format_epoch_display(epoch: int, index: int) -> str:
    """Format epoch display string with index (REQ_020)."""
    return f"Epoch {epoch} (Index {index})"


def update_visualizations(epoch_idx: int, neuron_idx: int, state: DashboardState):
    """Update all visualizations when slider or neuron changes."""
    state.current_epoch_idx = int(epoch_idx)
    state.selected_neuron = int(neuron_idx)

    plots = generate_all_plots(state)
    epoch = state.get_current_epoch()

    epoch_display = format_epoch_display(epoch, int(epoch_idx))
    return plots[0], plots[1], plots[2], plots[3], epoch_display, state


def update_activation_only(epoch_idx: int, neuron_idx: int, state: DashboardState):
    """Update only the activation heatmap when neuron changes."""
    state.selected_neuron = int(neuron_idx)

    if state.neuron_activations_artifact is not None:
        fig = render_neuron_heatmap(
            state.neuron_activations_artifact,
            epoch_idx=int(epoch_idx),
            neuron_idx=int(neuron_idx),
        )
    else:
        fig = create_empty_plot("Run analysis first")

    return fig, state


# ============================================================================
# Main App
# ============================================================================


def create_app() -> gr.Blocks:
    """Create and configure the Gradio application."""
    with gr.Blocks(title="Training Dynamics Workbench") as app:
        # Shared state
        state = gr.State(DashboardState())

        gr.Markdown(f"# Training Dynamics Workbench <small>v{__version__}</small>")
        gr.Markdown("Explore grokking dynamics in modular addition models")

        with gr.Tabs():
            # ================================================================
            # TRAINING TAB (REQ_007)
            # ================================================================
            with gr.TabItem("Training"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Training Parameters")
                        modulus = gr.Number(
                            label="Modulus (p)",
                            value=113,
                            precision=0,
                            info="Prime number for modular arithmetic",
                        )
                        model_seed = gr.Number(label="Model Seed", value=999, precision=0)
                        data_seed = gr.Number(label="Data Seed", value=598, precision=0)
                        train_fraction = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.3,
                            step=0.05,
                            label="Training Fraction",
                        )
                        num_epochs = gr.Number(label="Total Epochs", value=25000, precision=0)
                        checkpoint_str = gr.Textbox(
                            label="Checkpoint Epochs (comma-separated)",
                            value="0, 100, 500, 1000, 2000, 5000, 5500, 6000, 10000, 15000, 20000, 24999",
                            info="Leave empty for default schedule",
                        )
                        save_path = gr.Textbox(
                            label="Save Path", value="results/", info="Results directory"
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Status")
                        training_status = gr.Textbox(
                            label="Training Status",
                            value="Ready to train",
                            lines=8,
                            interactive=False,
                        )
                        train_btn = gr.Button("Start Training", variant="primary")

                train_btn.click(
                    fn=run_training,
                    inputs=[
                        modulus,
                        model_seed,
                        data_seed,
                        train_fraction,
                        num_epochs,
                        checkpoint_str,
                        save_path,
                    ],
                    outputs=[training_status],
                )

            # ================================================================
            # ANALYSIS TAB (REQ_008 + REQ_009)
            # ================================================================
            with gr.TabItem("Analysis"):
                with gr.Row():
                    with gr.Column(scale=3):
                        model_dropdown = gr.Dropdown(
                            label="Select Trained Model",
                            choices=[],
                            interactive=True,
                        )
                    with gr.Column(scale=1):
                        results_path = gr.Textbox(
                            label="Results Path", value="results/", visible=False
                        )
                        refresh_btn = gr.Button("Refresh Models")
                    with gr.Column(scale=1):
                        analyze_btn = gr.Button("Run Analysis", variant="primary")

                analysis_status = gr.Textbox(
                    label="Status", value="Select a model", interactive=False
                )

                gr.Markdown("### Global Epoch Control")
                with gr.Row():
                    epoch_slider = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=1,
                        value=0,
                        label="Checkpoint Index",
                        interactive=True,
                    )
                    epoch_display = gr.Textbox(
                        label="Current Epoch", value="Epoch 0", interactive=False
                    )

                gr.Markdown("### Loss Curves")
                loss_plot = gr.Plot(label="Train/Test Loss")

                gr.Markdown("### Analysis Visualizations")
                with gr.Row():
                    with gr.Column(scale=1):
                        freq_plot = gr.Plot(label="Dominant Frequencies")
                    with gr.Column(scale=1):
                        neuron_slider = gr.Slider(
                            minimum=0,
                            maximum=511,
                            step=1,
                            value=0,
                            label="Neuron Index",
                        )
                        activation_plot = gr.Plot(label="Neuron Activation Heatmap")

                clusters_plot = gr.Plot(label="Neuron Frequency Clusters")

                # ============================================================
                # Event Handlers
                # ============================================================

                # Refresh models list
                refresh_btn.click(
                    fn=refresh_models,
                    inputs=[results_path],
                    outputs=[model_dropdown],
                )

                # Load model when selected
                model_dropdown.change(
                    fn=load_model_data,
                    inputs=[model_dropdown, state],
                    outputs=[
                        state,
                        epoch_slider,
                        loss_plot,
                        freq_plot,
                        activation_plot,
                        clusters_plot,
                        analysis_status,
                        epoch_display,  # REQ_020
                        neuron_slider,
                    ],
                )

                # Run analysis
                analyze_btn.click(
                    fn=run_analysis,
                    inputs=[model_dropdown, state],
                    outputs=[analysis_status, state],
                ).then(
                    fn=load_model_data,
                    inputs=[model_dropdown, state],
                    outputs=[
                        state,
                        epoch_slider,
                        loss_plot,
                        freq_plot,
                        activation_plot,
                        clusters_plot,
                        analysis_status,
                        epoch_display,  # REQ_020
                        neuron_slider,
                    ],
                )

                # Synchronized slider updates (REQ_008 core feature)
                epoch_slider.change(
                    fn=update_visualizations,
                    inputs=[epoch_slider, neuron_slider, state],
                    outputs=[
                        loss_plot,
                        freq_plot,
                        activation_plot,
                        clusters_plot,
                        epoch_display,
                        state,
                    ],
                )

                # Neuron selector (only updates activation plot)
                neuron_slider.change(
                    fn=update_activation_only,
                    inputs=[epoch_slider, neuron_slider, state],
                    outputs=[activation_plot, state],
                )

        # Initial model list refresh
        app.load(
            fn=refresh_models,
            inputs=[results_path],
            outputs=[model_dropdown],
        )

    return app


def main():
    """Launch the dashboard."""
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
