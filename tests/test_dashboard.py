"""Tests for dashboard components and utilities."""
# pyright: reportArgumentType=false

import json
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import pytest

from dashboard_v2.components.loss_curves import render_loss_curves_with_indicator
from dashboard_v2.utils import (
    discover_trained_models,
    get_model_choices,
    parse_checkpoint_epochs,
    validate_training_params,
)


class TestLossCurvesRenderer:
    """Tests for REQ_009: Loss curves with epoch indicator."""

    def test_render_with_data(self):
        """Renders loss curves with valid data."""
        train_losses = [1.0, 0.5, 0.3, 0.2, 0.1]
        test_losses = [1.2, 0.6, 0.35, 0.25, 0.15]

        fig = render_loss_curves_with_indicator(train_losses, test_losses, current_epoch=2)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # train and test traces

    def test_render_with_checkpoints(self):
        """Renders with checkpoint markers."""
        train_losses = [1.0, 0.5, 0.3, 0.2, 0.1]
        test_losses = [1.2, 0.6, 0.35, 0.25, 0.15]

        fig = render_loss_curves_with_indicator(
            train_losses, test_losses, current_epoch=2, checkpoint_epochs=[0, 2, 4]
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # train, test, checkpoints

    def test_render_empty_state(self):
        """Renders placeholder with no data."""
        fig = render_loss_curves_with_indicator(None, None, current_epoch=0)

        assert isinstance(fig, go.Figure)
        # Check for annotation
        assert len(fig.layout.annotations) > 0

    def test_log_scale(self):
        """Respects log scale parameter."""
        train_losses = [1.0, 0.5, 0.3]
        test_losses = [1.2, 0.6, 0.35]

        fig = render_loss_curves_with_indicator(
            train_losses, test_losses, current_epoch=1, log_scale=True
        )
        assert fig.layout.yaxis.type == "log"

        fig = render_loss_curves_with_indicator(
            train_losses, test_losses, current_epoch=1, log_scale=False
        )
        assert fig.layout.yaxis.type == "linear"


class TestModelDiscovery:
    """Tests for model discovery utilities."""

    @pytest.fixture
    def mock_results_dir(self):
        """Create a mock results directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model directory structure
            model_dir = Path(tmpdir) / "modulo_addition" / "modulo_addition_p17_seed42"
            model_dir.mkdir(parents=True)

            # Create config.json
            config = {"prime": 17, "model_seed": 42, "n_ctx": 3}
            with open(model_dir / "config.json", "w") as f:
                json.dump(config, f)

            # Create metadata.json
            metadata = {"train_losses": [1.0, 0.5], "test_losses": [1.2, 0.6]}
            with open(model_dir / "metadata.json", "w") as f:
                json.dump(metadata, f)

            yield tmpdir

    def test_discover_models(self, mock_results_dir):
        """Discovers models in directory."""
        models = discover_trained_models(mock_results_dir)

        assert len(models) == 1
        assert models[0]["config"]["prime"] == 17
        assert "p=17" in models[0]["display_name"]

    def test_discover_empty_directory(self):
        """Returns empty list for non-existent directory."""
        models = discover_trained_models("/nonexistent/path")
        assert models == []

    def test_get_model_choices(self, mock_results_dir):
        """Converts models to dropdown choices."""
        models = discover_trained_models(mock_results_dir)
        choices = get_model_choices(models)

        assert len(choices) == 1
        assert isinstance(choices[0], tuple)
        assert len(choices[0]) == 2  # (display_name, path)


class TestParseCheckpointEpochs:
    """Tests for checkpoint epoch parsing."""

    def test_parse_valid(self):
        """Parses comma-separated integers."""
        result = parse_checkpoint_epochs("0, 100, 500, 1000")
        assert result == [0, 100, 500, 1000]

    def test_parse_with_duplicates(self):
        """Removes duplicates and sorts."""
        result = parse_checkpoint_epochs("100, 0, 100, 500")
        assert result == [0, 100, 500]

    def test_parse_empty(self):
        """Returns empty list for empty string."""
        result = parse_checkpoint_epochs("")
        assert result == []

    def test_parse_invalid(self):
        """Raises ValueError for invalid input."""
        with pytest.raises(ValueError):
            parse_checkpoint_epochs("0, abc, 100")


class TestValidateTrainingParams:
    """Tests for training parameter validation."""

    def test_valid_params(self):
        """Accepts valid parameters."""
        is_valid, msg = validate_training_params(
            modulus=17,
            model_seed=42,
            data_seed=598,
            train_fraction=0.3,
            num_epochs=100,
            checkpoint_str="0, 50, 99",
            save_path="results/",
        )

        assert is_valid
        assert msg == ""

    def test_invalid_modulus(self):
        """Rejects modulus < 2."""
        is_valid, msg = validate_training_params(
            modulus=1,
            model_seed=42,
            data_seed=598,
            train_fraction=0.3,
            num_epochs=100,
            checkpoint_str="0, 50",
            save_path="results/",
        )

        assert not is_valid
        assert "Modulus" in msg

    def test_invalid_train_fraction(self):
        """Rejects train_fraction outside (0, 1)."""
        is_valid, msg = validate_training_params(
            modulus=17,
            model_seed=42,
            data_seed=598,
            train_fraction=1.5,
            num_epochs=100,
            checkpoint_str="0, 50",
            save_path="results/",
        )

        assert not is_valid
        assert "fraction" in msg.lower()

    def test_checkpoint_exceeds_epochs(self):
        """Rejects checkpoint >= num_epochs."""
        is_valid, msg = validate_training_params(
            modulus=17,
            model_seed=42,
            data_seed=598,
            train_fraction=0.3,
            num_epochs=100,
            checkpoint_str="0, 50, 100",  # 100 >= 100
            save_path="results/",
        )

        assert not is_valid
        assert "100" in msg


class TestVersioning:
    """Tests for REQ_010: Application Versioning."""

    def test_version_importable_from_version_module(self):
        """Can import __version__ from dashboard_v2.version."""
        from dashboard_v2.version import __version__

        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_version_format_semantic(self):
        """Version follows MAJOR.MINOR.BUILD format."""
        from dashboard_v2.version import __version__

        parts = __version__.split(".")
        assert len(parts) == 3, f"Version should have 3 parts: {__version__}"

        # All parts should be numeric
        for part in parts:
            assert part.isdigit(), f"Version parts should be numeric: {__version__}"

    def test_version_is_mvp(self):
        """Version starts with 0.x.x for MVP phase."""
        from dashboard_v2.version import __version__

        assert __version__.startswith("0."), f"MVP version should start with 0.x: {__version__}"


class TestFamilySelectorComponent:
    """Tests for REQ_021d: Family selector component functions."""

    @pytest.fixture
    def mock_family_dir(self):
        """Create a mock model_families directory with a family.json."""
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model_families directory
            families_dir = Path(tmpdir) / "model_families"
            family_dir = families_dir / "test_family"
            family_dir.mkdir(parents=True)

            # Create family.json
            family_json = {
                "name": "test_family",
                "display_name": "Test Family",
                "description": "A test family",
                "architecture": {"n_layers": 1, "n_heads": 4},
                "domain_parameters": {
                    "prime": {"type": "int", "default": 17},
                    "seed": {"type": "int", "default": 42},
                },
                "analyzers": [],
                "visualizations": [],
                "analysis_dataset": {"type": "test"},
                "variant_pattern": "test_family_p{prime}_seed{seed}",
            }
            with open(family_dir / "family.json", "w") as f:
                json.dump(family_json, f)

            # Create results directory with a variant
            results_dir = Path(tmpdir) / "results" / "test_family"
            variant_dir = results_dir / "test_family_p17_seed42"
            checkpoints_dir = variant_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True)

            # Create a checkpoint file (marks variant as TRAINED)
            dummy_state = {"dummy": torch.zeros(1)}
            save_file(dummy_state, checkpoints_dir / "checkpoint_epoch_00000.safetensors")

            # Create metadata.json
            with open(variant_dir / "metadata.json", "w") as f:
                json.dump({"train_losses": [1.0]}, f)

            # Create config.json
            with open(variant_dir / "config.json", "w") as f:
                json.dump({"prime": 17, "seed": 42}, f)

            yield tmpdir

    def test_get_family_choices(self, mock_family_dir):
        """get_family_choices returns list of (display_name, name) tuples."""
        from dashboard_v2.components.family_selector import get_family_choices
        from miscope.families import FamilyRegistry

        registry = FamilyRegistry(
            model_families_dir=Path(mock_family_dir) / "model_families",
            results_dir=Path(mock_family_dir) / "results",
        )

        choices = get_family_choices(registry)

        assert len(choices) == 1
        assert choices[0] == ("Test Family", "test_family")

    def test_get_variant_choices(self, mock_family_dir):
        """get_variant_choices returns list of variant choices."""
        from dashboard_v2.components.family_selector import get_variant_choices
        from miscope.families import FamilyRegistry

        registry = FamilyRegistry(
            model_families_dir=Path(mock_family_dir) / "model_families",
            results_dir=Path(mock_family_dir) / "results",
        )

        choices = get_variant_choices(registry, "test_family")

        assert len(choices) == 1
        # Check display name contains state indicator and params
        display_name, name = choices[0]
        assert "prime=17" in display_name
        assert "seed=42" in display_name
        assert name == "test_family_p17_seed42"

    def test_get_variant_choices_empty_family(self, mock_family_dir):
        """get_variant_choices returns empty list for unknown family."""
        from dashboard_v2.components.family_selector import get_variant_choices
        from miscope.families import FamilyRegistry

        registry = FamilyRegistry(
            model_families_dir=Path(mock_family_dir) / "model_families",
            results_dir=Path(mock_family_dir) / "results",
        )

        choices = get_variant_choices(registry, "nonexistent_family")

        assert choices == []

    def test_get_state_indicator(self, mock_family_dir):
        """get_state_indicator returns correct symbols for states."""
        from dashboard_v2.components.family_selector import get_state_indicator
        from miscope.families import FamilyRegistry

        registry = FamilyRegistry(
            model_families_dir=Path(mock_family_dir) / "model_families",
            results_dir=Path(mock_family_dir) / "results",
        )

        variants = registry.get_variants("test_family")
        assert len(variants) == 1
        variant = variants[0]

        indicator = get_state_indicator(variant)
        assert indicator == "●"  # Trained state

    def test_format_variant_params(self, mock_family_dir):
        """format_variant_params creates readable parameter string."""
        from dashboard_v2.components.family_selector import format_variant_params
        from miscope.families import FamilyRegistry

        registry = FamilyRegistry(
            model_families_dir=Path(mock_family_dir) / "model_families",
            results_dir=Path(mock_family_dir) / "results",
        )

        variants = registry.get_variants("test_family")
        variant = variants[0]

        params_str = format_variant_params(variant)
        assert "prime=17" in params_str
        assert "seed=42" in params_str

    def test_get_available_actions(self, mock_family_dir):
        """get_available_actions returns actions for variant state."""
        from dashboard_v2.components.family_selector import get_available_actions
        from miscope.families import FamilyRegistry

        registry = FamilyRegistry(
            model_families_dir=Path(mock_family_dir) / "model_families",
            results_dir=Path(mock_family_dir) / "results",
        )

        variants = registry.get_variants("test_family")
        variant = variants[0]

        actions = get_available_actions(variant)
        assert "Analyze" in actions
