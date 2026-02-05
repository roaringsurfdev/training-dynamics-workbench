"""Main Gradio dashboard application.

REQ_007: Training controls
REQ_008: Analysis and synchronized visualizations
REQ_009: Loss curves with epoch indicator
REQ_020: Checkpoint epoch-index display
REQ_021d: Dashboard integration with Model Families
REQ_021e: Training integration with Model Families
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
from dashboard.components import (
    get_family_choices,
    get_variant_choices,
    render_loss_curves_with_indicator,
)
from dashboard.state import DashboardState
from dashboard.utils import (
    discover_trained_models,
    get_model_choices,
    parse_checkpoint_epochs,
    validate_training_params,
)
from dashboard.version import __version__
from families import FamilyRegistry, TrainingResult
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
# Training Tab Handlers (REQ_007 + REQ_021e)
# ============================================================================


def on_training_family_change(family_name: str | None):
    """Handle family selection change in training tab.

    Updates domain parameter inputs based on selected family.

    Args:
        family_name: Selected family name

    Returns:
        Tuple of (variant_preview, prime_value, seed_value)
    """
    if not family_name:
        return "Select a family", 113, 999

    registry = get_registry()
    family = registry.get_family(family_name)
    defaults = family.get_default_params()

    prime = defaults.get("prime", 113)
    seed = defaults.get("seed", 999)

    # Generate variant preview
    preview_params = {"prime": prime, "seed": seed}
    variant_name = family.get_variant_directory_name(preview_params)

    return f"Variant: {variant_name}", prime, seed


def update_variant_preview(family_name: str | None, prime: int, seed: int) -> str:
    """Update variant name preview when parameters change.

    Args:
        family_name: Selected family name
        prime: Prime value
        seed: Seed value

    Returns:
        Variant preview string
    """
    if not family_name:
        return "Select a family"

    try:
        registry = get_registry()
        family = registry.get_family(family_name)
        params = {"prime": int(prime), "seed": int(seed)}
        variant_name = family.get_variant_directory_name(params)
        return f"Variant: {variant_name}"
    except Exception:
        return "Invalid parameters"


def run_family_training(
    family_name: str | None,
    prime: int,
    seed: int,
    data_seed: int,
    train_fraction: float,
    num_epochs: int,
    checkpoint_str: str,
    progress=gr.Progress(),
) -> str:
    """Execute training using the family abstraction (REQ_021e).

    Args:
        family_name: Selected model family
        prime: Prime/modulus parameter
        seed: Model initialization seed
        data_seed: Data split seed
        train_fraction: Fraction for training
        num_epochs: Total training epochs
        checkpoint_str: Comma-separated checkpoint epochs
        progress: Gradio progress tracker

    Returns:
        Status message with training results
    """
    if not family_name:
        return "Error: Please select a model family"

    try:
        progress(0, desc="Initializing...")

        registry = get_registry()
        family = registry.get_family(family_name)

        # Build domain parameters
        params = {"prime": int(prime), "seed": int(seed)}

        # Create variant through registry
        variant = registry.create_variant(family, params)

        # Parse checkpoint epochs
        checkpoint_epochs = parse_checkpoint_epochs(checkpoint_str)
        if not checkpoint_epochs:
            checkpoint_epochs = None  # Use family defaults

        progress(0.05, desc="Creating variant...")

        # Create progress callback for training
        def training_progress(pct: float, desc: str):
            # Map training progress (0-1) to UI progress (0.1-1.0)
            ui_progress = 0.1 + (pct * 0.9)
            progress(ui_progress, desc=desc)

        progress(0.1, desc="Starting training...")

        # Train via variant (uses family.create_model() and family.generate_training_dataset())
        result: TrainingResult = variant.train(
            num_epochs=int(num_epochs),
            checkpoint_epochs=checkpoint_epochs,
            training_fraction=train_fraction,
            data_seed=int(data_seed),
            progress_callback=training_progress,
        )

        progress(1.0, desc="Training complete!")

        # Refresh registry to pick up new variant
        refresh_registry()

        return (
            f"Training complete!\n"
            f"Variant: {variant.name}\n"
            f"Saved to: {result.variant_dir}\n"
            f"Checkpoints: {len(result.checkpoint_epochs)}\n"
            f"Final train loss: {result.final_train_loss:.6f}\n"
            f"Final test loss: {result.final_test_loss:.6f}\n\n"
            f"Variant now available in Analysis tab (click Refresh)"
        )

    except Exception as e:
        import traceback
        return f"Training failed: {e}\n\n{traceback.format_exc()}"


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
    """Execute training with the given parameters (legacy function).

    Note: This is kept for backward compatibility. New training should use
    run_family_training() which goes through the family abstraction.
    """
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


# Global registry instance (initialized once per app)
_registry: FamilyRegistry | None = None


def get_registry() -> FamilyRegistry:
    """Get or create the global FamilyRegistry instance."""
    global _registry
    if _registry is None:
        _registry = FamilyRegistry(
            model_families_dir=Path("model_families"),
            results_dir=Path("results"),
        )
    return _registry


def refresh_registry() -> None:
    """Force registry to reload from filesystem."""
    global _registry
    _registry = FamilyRegistry(
        model_families_dir=Path("model_families"),
        results_dir=Path("results"),
    )


# ============================================================================
# Family/Variant Handlers (REQ_021d)
# ============================================================================


def on_family_change(family_name: str | None, state: DashboardState):
    """Handle family dropdown selection change.

    Updates variant dropdown with variants from the selected family.
    """
    state.clear_selection()

    if not family_name:
        return (
            state,
            gr.Dropdown(choices=[], value=None),
            "Select a family",
        )

    state.selected_family_name = family_name
    registry = get_registry()
    variant_choices = get_variant_choices(registry, family_name)

    family = registry.get_family(family_name)
    status = f"Family: {family.display_name}"
    if variant_choices:
        status += f" ({len(variant_choices)} variants)"
    else:
        status += " (no variants found)"

    return (
        state,
        gr.Dropdown(choices=variant_choices, value=None),
        status,
    )


def on_variant_change(variant_name: str | None, family_name: str | None, state: DashboardState):
    """Handle variant dropdown selection change.

    Loads variant data and artifacts for visualization.

    Args:
        variant_name: Selected variant name
        family_name: Selected family name (from dropdown, not state)
        state: Dashboard state
    """
    # Use family_name from dropdown (fixes issue where state.selected_family_name is None on page load)
    effective_family_name = family_name or state.selected_family_name

    if not variant_name or not effective_family_name:
        state.clear_artifacts()
        return (
            state,
            gr.Slider(minimum=0, maximum=1, value=0),
            create_empty_plot("Select a variant"),
            create_empty_plot("Select a variant"),
            create_empty_plot("Select a variant"),
            create_empty_plot("Select a variant"),
            "No variant selected",
            "Epoch 0 (Index 0)",
            gr.Slider(minimum=0, maximum=511, value=0, step=1),
        )

    # Update state with both family and variant
    state.selected_family_name = effective_family_name
    state.selected_variant_name = variant_name
    registry = get_registry()
    family = registry.get_family(effective_family_name)
    variants = registry.get_variants(family)

    # Find the selected variant
    variant = None
    for v in variants:
        if v.name == variant_name:
            variant = v
            break

    if variant is None:
        state.clear_artifacts()
        return (
            state,
            gr.Slider(minimum=0, maximum=1, value=0),
            create_empty_plot("Variant not found"),
            create_empty_plot("Variant not found"),
            create_empty_plot("Variant not found"),
            create_empty_plot("Variant not found"),
            "Variant not found",
            "Epoch 0 (Index 0)",
            gr.Slider(minimum=0, maximum=511, value=0, step=1),
        )

    # Use variant's directory as model_path for compatibility
    model_path = str(variant.variant_dir)
    state.selected_model_path = model_path
    state.clear_artifacts()

    # Load metadata for loss curves
    if variant.metadata_path.exists():
        with open(variant.metadata_path) as f:
            metadata = json.load(f)
        state.train_losses = metadata.get("train_losses", [])
        state.test_losses = metadata.get("test_losses", [])

    # Load config
    if variant.config_path.exists():
        with open(variant.config_path) as f:
            state.model_config = json.load(f)

    # Check for artifacts
    artifacts_dir = variant.artifacts_dir
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

    # Build status message
    status_parts = [f"Variant: {variant.name}"]
    status_parts.append(f"State: {variant.state.value}")
    if state.available_epochs:
        status_parts.append(f"{len(state.available_epochs)} checkpoints")
    status = " | ".join(status_parts)

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


def run_analysis_for_variant(
    variant_name: str | None,
    family_name: str | None,
    state: DashboardState,
    progress=gr.Progress(),
):
    """Run analysis pipeline on the selected variant."""
    # Use family_name from dropdown (fixes issue where state.selected_family_name is None on page load)
    effective_family_name = family_name or state.selected_family_name

    if not variant_name or not effective_family_name:
        return "No variant selected", state

    try:
        progress(0, desc="Initializing analysis...")

        registry = get_registry()
        family = registry.get_family(effective_family_name)
        variants = registry.get_variants(family)

        # Find the selected variant
        variant = None
        for v in variants:
            if v.name == variant_name:
                variant = v
                break

        if variant is None:
            return "Variant not found", state

        progress(0.1, desc="Starting analysis pipeline...")

        def pipeline_progress(pct: float, desc: str):
            ui_progress = 0.1 + (pct * 0.9)
            progress(ui_progress, desc=desc)

        # Pipeline now takes Variant directly (no adapter needed)
        pipeline = AnalysisPipeline(variant)
        pipeline.register(DominantFrequenciesAnalyzer())
        pipeline.register(NeuronActivationsAnalyzer())
        pipeline.register(NeuronFreqClustersAnalyzer())
        pipeline.run(progress_callback=pipeline_progress)

        progress(1.0, desc="Analysis complete!")

        return f"Analysis complete! Artifacts saved to {variant.artifacts_dir}", state

    except Exception as e:
        import traceback
        return f"Analysis failed: {e}\n\n{traceback.format_exc()}", state


def refresh_variants(family_name: str | None) -> gr.Dropdown:
    """Refresh variant dropdown for the current family."""
    if not family_name:
        return gr.Dropdown(choices=[], value=None)

    refresh_registry()  # Reload from filesystem
    registry = get_registry()
    variant_choices = get_variant_choices(registry, family_name)
    return gr.Dropdown(choices=variant_choices, value=None)


# ============================================================================
# Legacy Model Discovery (kept for backward compatibility)
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


def update_visualizations(
    epoch_idx: int | None, neuron_idx: int | None, state: DashboardState
):
    """Update all visualizations when slider or neuron changes."""
    # Guard against None values (BUG_003: slider fires event when input is cleared)
    if epoch_idx is None or neuron_idx is None:
        epoch_idx = epoch_idx if epoch_idx is not None else state.current_epoch_idx
        neuron_idx = neuron_idx if neuron_idx is not None else state.selected_neuron

    state.current_epoch_idx = int(epoch_idx)
    state.selected_neuron = int(neuron_idx)

    plots = generate_all_plots(state)
    epoch = state.get_current_epoch()

    epoch_display = format_epoch_display(epoch, int(epoch_idx))
    return plots[0], plots[1], plots[2], plots[3], epoch_display, state


def update_activation_only(
    epoch_idx: int | None, neuron_idx: int | None, state: DashboardState
):
    """Update only the activation heatmap when neuron changes."""
    # Guard against None values (BUG_003: slider fires event when input is cleared)
    if neuron_idx is None:
        neuron_idx = state.selected_neuron
    if epoch_idx is None:
        epoch_idx = state.current_epoch_idx

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
    # Initialize family and variant choices from registry (used by both Training and Analysis tabs)
    registry = get_registry()
    family_choices = get_family_choices(registry)

    # Initialize variant choices for default family (fixes BUG_004)
    default_family = family_choices[0][1] if family_choices else None
    initial_variant_choices = get_variant_choices(registry, default_family) if default_family else []

    with gr.Blocks(title="Training Dynamics Workbench") as app:
        # Shared state
        state = gr.State(DashboardState())

        gr.Markdown(f"# Training Dynamics Workbench <small>v{__version__}</small>")
        gr.Markdown("Explore grokking dynamics in modular addition models")

        with gr.Tabs():
            # ================================================================
            # TRAINING TAB (REQ_007 + REQ_021e)
            # ================================================================
            with gr.TabItem("Training"):
                # Family Selection (REQ_021e)
                gr.Markdown("### Model Family")
                with gr.Row():
                    with gr.Column(scale=2):
                        training_family_dropdown = gr.Dropdown(
                            label="Model Family",
                            choices=family_choices,
                            value=family_choices[0][1] if family_choices else None,
                            interactive=True,
                        )
                    with gr.Column(scale=3):
                        variant_preview = gr.Textbox(
                            label="Variant Preview",
                            value="Select a family",
                            interactive=False,
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Domain Parameters")
                        training_prime = gr.Number(
                            label="Prime (p)",
                            value=113,
                            precision=0,
                            info="Modulus for the addition task",
                        )
                        training_seed = gr.Number(
                            label="Seed",
                            value=999,
                            precision=0,
                            info="Random seed for model initialization",
                        )

                        gr.Markdown("### Training Parameters")
                        training_data_seed = gr.Number(
                            label="Data Seed",
                            value=598,
                            precision=0,
                            info="Random seed for train/test split",
                        )
                        training_fraction = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.3,
                            step=0.05,
                            label="Training Fraction",
                        )
                        training_epochs = gr.Number(
                            label="Total Epochs",
                            value=25000,
                            precision=0,
                        )
                        training_checkpoint_str = gr.Textbox(
                            label="Checkpoint Epochs (comma-separated)",
                            value="",
                            info="Leave empty for default schedule",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Status")
                        training_status = gr.Textbox(
                            label="Training Status",
                            value="Ready to train",
                            lines=10,
                            interactive=False,
                        )
                        train_btn = gr.Button("Start Training", variant="primary")

                # Event handlers for Training Tab (REQ_021e)
                training_family_dropdown.change(
                    fn=on_training_family_change,
                    inputs=[training_family_dropdown],
                    outputs=[variant_preview, training_prime, training_seed],
                )

                # Update variant preview when parameters change
                training_prime.change(
                    fn=update_variant_preview,
                    inputs=[training_family_dropdown, training_prime, training_seed],
                    outputs=[variant_preview],
                )
                training_seed.change(
                    fn=update_variant_preview,
                    inputs=[training_family_dropdown, training_prime, training_seed],
                    outputs=[variant_preview],
                )

                # Training button uses family-based training
                train_btn.click(
                    fn=run_family_training,
                    inputs=[
                        training_family_dropdown,
                        training_prime,
                        training_seed,
                        training_data_seed,
                        training_fraction,
                        training_epochs,
                        training_checkpoint_str,
                    ],
                    outputs=[training_status],
                )

            # ================================================================
            # ANALYSIS TAB (REQ_008 + REQ_009 + REQ_021d)
            # ================================================================
            with gr.TabItem("Analysis"):
                # Family/Variant Selection (REQ_021d)
                gr.Markdown("### Model Selection")
                with gr.Row():
                    with gr.Column(scale=2):
                        # Uses family_choices initialized at top of create_app()
                        family_dropdown = gr.Dropdown(
                            label="Model Family",
                            choices=family_choices,
                            value=family_choices[0][1] if family_choices else None,
                            interactive=True,
                        )
                    with gr.Column(scale=3):
                        variant_dropdown = gr.Dropdown(
                            label="Select Variant",
                            choices=initial_variant_choices,
                            interactive=True,
                        )
                    with gr.Column(scale=1):
                        refresh_btn = gr.Button("Refresh")
                    with gr.Column(scale=1):
                        analyze_btn = gr.Button("Run Analysis", variant="primary")

                analysis_status = gr.Textbox(
                    label="Status", value="Select a family and variant", interactive=False
                )

                # Hidden components for backward compatibility
                results_path = gr.Textbox(
                    label="Results Path", value="results/", visible=False
                )
                model_dropdown = gr.Dropdown(
                    label="Select Trained Model (Legacy)",
                    choices=[],
                    interactive=True,
                    visible=False,
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
                # Event Handlers (REQ_021d)
                # ============================================================

                # Family selection changes variant list
                family_dropdown.change(
                    fn=on_family_change,
                    inputs=[family_dropdown, state],
                    outputs=[state, variant_dropdown, analysis_status],
                )

                # Variant selection loads data
                variant_dropdown.change(
                    fn=on_variant_change,
                    inputs=[variant_dropdown, family_dropdown, state],
                    outputs=[
                        state,
                        epoch_slider,
                        loss_plot,
                        freq_plot,
                        activation_plot,
                        clusters_plot,
                        analysis_status,
                        epoch_display,
                        neuron_slider,
                    ],
                )

                # Refresh button reloads variants for current family
                refresh_btn.click(
                    fn=refresh_variants,
                    inputs=[family_dropdown],
                    outputs=[variant_dropdown],
                )

                # Run analysis on selected variant
                analyze_btn.click(
                    fn=run_analysis_for_variant,
                    inputs=[variant_dropdown, family_dropdown, state],
                    outputs=[analysis_status, state],
                ).then(
                    fn=on_variant_change,
                    inputs=[variant_dropdown, family_dropdown, state],
                    outputs=[
                        state,
                        epoch_slider,
                        loss_plot,
                        freq_plot,
                        activation_plot,
                        clusters_plot,
                        analysis_status,
                        epoch_display,
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

        # Note: Variants are now initialized at creation time (fixes BUG_004)
        # The init_variants/app.load pattern caused issues on page reload

    return app


def main():
    """Launch the dashboard."""
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
