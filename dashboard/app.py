"""Main Gradio dashboard application.

REQ_007: Training controls
REQ_008: Analysis and synchronized visualizations
REQ_009: Loss curves with epoch indicator
REQ_020: Checkpoint epoch-index display
REQ_021d: Dashboard integration with Model Families
REQ_021e: Training integration with Model Families
REQ_021f: Per-epoch artifact loading
REQ_025: Attention head pattern visualization
REQ_026: Attention head frequency specialization
REQ_027: Neuron frequency specialization summary statistics
"""

import json
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go

from analysis import AnalysisPipeline, ArtifactLoader
from analysis.analyzers import (
    AttentionFreqAnalyzer,
    AttentionPatternsAnalyzer,
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
from dashboard.utils import parse_checkpoint_epochs
from dashboard.version import __version__
from families import FamilyRegistry, TrainingResult
from visualization import (
    render_attention_freq_heatmap,
    render_attention_heads,
    render_attention_specialization_trajectory,
    render_dominant_frequencies,
    render_freq_clusters,
    render_neuron_heatmap,
    render_specialization_by_frequency,
    render_specialization_trajectory,
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

    Loads variant metadata and discovers available artifacts (REQ_021f).
    Artifact data is NOT loaded into state — loaded per-epoch on demand.

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
            create_empty_plot("Select a variant"),
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
            create_empty_plot("Variant not found"),
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

    # Load config (includes d_mlp for neuron count)
    if variant.config_path.exists():
        with open(variant.config_path) as f:
            config = json.load(f)
        state.model_config = config
        state.n_neurons = config.get("d_mlp", 512)

    # Discover available artifacts without loading data (REQ_021f)
    artifacts_dir = variant.artifacts_dir
    if artifacts_dir.exists():
        try:
            loader = ArtifactLoader(str(artifacts_dir))
            available = loader.get_available_analyzers()
            state.artifacts_dir = str(artifacts_dir)
            state.available_analyzers = available

            # Discover available epochs from any analyzer
            for analyzer_name in available:
                epochs = loader.get_epochs(analyzer_name)
                if epochs:
                    state.available_epochs = epochs
                    break

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
        plots[4],  # neuron specialization trajectory (REQ_027)
        plots[5],  # specialization by frequency (REQ_027)
        plots[6],  # attention (REQ_025)
        plots[7],  # attention freq heatmap (REQ_026)
        plots[8],  # attention specialization trajectory (REQ_026)
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
        pipeline.register(AttentionFreqAnalyzer())
        pipeline.register(AttentionPatternsAnalyzer())
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


def generate_all_plots(state: DashboardState):
    """Generate all visualization plots for current state.

    Loads per-epoch artifact data on demand via ArtifactLoader (REQ_021f).
    """
    epoch = state.get_current_epoch()

    # Loss curves
    loss_fig = render_loss_curves_with_indicator(
        state.train_losses,
        state.test_losses,
        current_epoch=epoch,
        checkpoint_epochs=state.available_epochs,
    )

    # Load per-epoch data on demand
    if state.artifacts_dir and state.available_epochs:
        loader = ArtifactLoader(state.artifacts_dir)

        # Dominant frequencies
        if "dominant_frequencies" in state.available_analyzers:
            try:
                epoch_data = loader.load_epoch("dominant_frequencies", epoch)
                freq_fig = render_dominant_frequencies(
                    epoch_data,
                    epoch=epoch,
                    threshold=1.0,
                )
            except FileNotFoundError:
                freq_fig = create_empty_plot("No data for this epoch")
        else:
            freq_fig = create_empty_plot("Run analysis first")

        # Neuron activation
        if "neuron_activations" in state.available_analyzers:
            try:
                epoch_data = loader.load_epoch("neuron_activations", epoch)
                activation_fig = render_neuron_heatmap(
                    epoch_data,
                    epoch=epoch,
                    neuron_idx=state.selected_neuron,
                )
            except FileNotFoundError:
                activation_fig = create_empty_plot("No data for this epoch")
        else:
            activation_fig = create_empty_plot("Run analysis first")

        # Frequency clusters
        if "neuron_freq_norm" in state.available_analyzers:
            try:
                epoch_data = loader.load_epoch("neuron_freq_norm", epoch)
                clusters_fig = render_freq_clusters(epoch_data, epoch=epoch)
            except FileNotFoundError:
                clusters_fig = create_empty_plot("No data for this epoch")
        else:
            clusters_fig = create_empty_plot("Run analysis first")

        # Neuron specialization trajectory (REQ_027, cross-epoch)
        if "neuron_freq_norm" in state.available_analyzers and loader.has_summary(
            "neuron_freq_norm"
        ):
            try:
                summary_data = loader.load_summary("neuron_freq_norm")
                spec_traj_fig = render_specialization_trajectory(summary_data, current_epoch=epoch)
                spec_freq_fig = render_specialization_by_frequency(
                    summary_data, current_epoch=epoch
                )
            except FileNotFoundError:
                spec_traj_fig = create_empty_plot("No summary data")
                spec_freq_fig = create_empty_plot("No summary data")
        else:
            spec_traj_fig = create_empty_plot("Run analysis first")
            spec_freq_fig = create_empty_plot("Run analysis first")

        # Attention patterns (REQ_025)
        if "attention_patterns" in state.available_analyzers:
            try:
                epoch_data = loader.load_epoch("attention_patterns", epoch)
                attention_fig = render_attention_heads(
                    epoch_data,
                    epoch=epoch,
                    to_position=state.selected_to_position,
                    from_position=state.selected_from_position,
                )
            except FileNotFoundError:
                attention_fig = create_empty_plot("No data for this epoch")
        else:
            attention_fig = create_empty_plot("Run analysis first")

        # Attention frequency heatmap (REQ_026, per-epoch)
        if "attention_freq" in state.available_analyzers:
            try:
                epoch_data = loader.load_epoch("attention_freq", epoch)
                attn_freq_fig = render_attention_freq_heatmap(epoch_data, epoch=epoch)
            except FileNotFoundError:
                attn_freq_fig = create_empty_plot("No data for this epoch")
        else:
            attn_freq_fig = create_empty_plot("Run analysis first")

        # Attention specialization trajectory (REQ_026, cross-epoch)
        if "attention_freq" in state.available_analyzers and loader.has_summary("attention_freq"):
            try:
                summary_data = loader.load_summary("attention_freq")
                attn_spec_fig = render_attention_specialization_trajectory(
                    summary_data, current_epoch=epoch
                )
            except FileNotFoundError:
                attn_spec_fig = create_empty_plot("No summary data")
        else:
            attn_spec_fig = create_empty_plot("Run analysis first")
    else:
        freq_fig = create_empty_plot("Run analysis first")
        activation_fig = create_empty_plot("Run analysis first")
        clusters_fig = create_empty_plot("Run analysis first")
        spec_traj_fig = create_empty_plot("Run analysis first")
        spec_freq_fig = create_empty_plot("Run analysis first")
        attention_fig = create_empty_plot("Run analysis first")
        attn_freq_fig = create_empty_plot("Run analysis first")
        attn_spec_fig = create_empty_plot("Run analysis first")

    return (
        loss_fig,
        freq_fig,
        activation_fig,
        clusters_fig,
        spec_traj_fig,
        spec_freq_fig,
        attention_fig,
        attn_freq_fig,
        attn_spec_fig,
    )


def format_epoch_display(epoch: int, index: int) -> str:
    """Format epoch display string with index (REQ_020)."""
    return f"Epoch {epoch} (Index {index})"


def update_visualizations(epoch_idx: int | None, neuron_idx: int | None, state: DashboardState):
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
    return (
        plots[0],
        plots[1],
        plots[2],
        plots[3],
        plots[4],
        plots[5],
        plots[6],
        plots[7],
        plots[8],
        epoch_display,
        state,
    )


def update_activation_only(epoch_idx: int | None, neuron_idx: int | None, state: DashboardState):
    """Update only the activation heatmap when neuron changes."""
    # Guard against None values (BUG_003: slider fires event when input is cleared)
    if neuron_idx is None:
        neuron_idx = state.selected_neuron
    if epoch_idx is None:
        epoch_idx = state.current_epoch_idx

    state.selected_neuron = int(neuron_idx)

    if (
        state.artifacts_dir
        and "neuron_activations" in state.available_analyzers
        and state.available_epochs
    ):
        try:
            epoch = state.get_current_epoch()
            loader = ArtifactLoader(state.artifacts_dir)
            epoch_data = loader.load_epoch("neuron_activations", epoch)
            fig = render_neuron_heatmap(
                epoch_data,
                epoch=epoch,
                neuron_idx=int(neuron_idx),
            )
        except FileNotFoundError:
            fig = create_empty_plot("No data for this epoch")
    else:
        fig = create_empty_plot("Run analysis first")

    return fig, state


def update_attention_only(position_pair: str | None, epoch_idx: int | None, state: DashboardState):
    """Update only the attention plot when position pair changes (REQ_025)."""
    if epoch_idx is None:
        epoch_idx = state.current_epoch_idx

    if position_pair:
        parts = position_pair.split(",")
        if len(parts) == 2:
            state.selected_to_position = int(parts[0])
            state.selected_from_position = int(parts[1])

    if (
        state.artifacts_dir
        and "attention_patterns" in state.available_analyzers
        and state.available_epochs
    ):
        try:
            epoch = state.get_current_epoch()
            loader = ArtifactLoader(state.artifacts_dir)
            epoch_data = loader.load_epoch("attention_patterns", epoch)
            fig = render_attention_heads(
                epoch_data,
                epoch=epoch,
                to_position=state.selected_to_position,
                from_position=state.selected_from_position,
            )
        except FileNotFoundError:
            fig = create_empty_plot("No data for this epoch")
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
    initial_variant_choices = (
        get_variant_choices(registry, default_family) if default_family else []
    )

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

                # Neuron Specialization (REQ_027)
                gr.Markdown("### Neuron Frequency Specialization")
                with gr.Row():
                    spec_traj_plot = gr.Plot(label="Neuron Specialization Trajectory")
                    spec_freq_plot = gr.Plot(label="Specialization by Frequency")

                # Attention Patterns (REQ_025)
                gr.Markdown("### Attention Patterns")
                position_pair_dropdown = gr.Dropdown(
                    label="Attention Relationship",
                    choices=[
                        ("= attending to a", "2,0"),
                        ("= attending to b", "2,1"),
                        ("b attending to a", "1,0"),
                        ("b attending to b", "1,1"),
                        ("a attending to a", "0,0"),
                        ("a attending to b", "0,1"),
                    ],
                    value="2,0",
                    interactive=True,
                )
                attention_plot = gr.Plot(label="Attention Head Patterns")

                # Attention Frequency Specialization (REQ_026)
                gr.Markdown("### Attention Head Frequency Specialization")
                with gr.Row():
                    attn_freq_plot = gr.Plot(label="Attention Head Frequency Decomposition")
                    attn_spec_plot = gr.Plot(label="Head Specialization Trajectory")

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
                        spec_traj_plot,
                        spec_freq_plot,
                        attention_plot,
                        attn_freq_plot,
                        attn_spec_plot,
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
                        spec_traj_plot,
                        spec_freq_plot,
                        attention_plot,
                        attn_freq_plot,
                        attn_spec_plot,
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
                        spec_traj_plot,
                        spec_freq_plot,
                        attention_plot,
                        attn_freq_plot,
                        attn_spec_plot,
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

                # Position pair selector (only updates attention plot, REQ_025)
                position_pair_dropdown.change(
                    fn=update_attention_only,
                    inputs=[position_pair_dropdown, epoch_slider, state],
                    outputs=[attention_plot, state],
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
