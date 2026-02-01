"""
Tests for REQ_001 (Configurable Checkpoint Epochs) and REQ_002 (Safetensors Model Persistence)

Tests are organized by Conditions of Satisfaction from each requirement.
"""
import json
import os
import shutil
import tempfile

import pytest
import torch

from ModuloAdditionSpecification import (
    ModuloAdditionSpecification,
    DEFAULT_CHECKPOINT_EPOCHS,
)


@pytest.fixture
def temp_results_dir():
    """Create a temporary directory for test results."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def model_spec(temp_results_dir):
    """Create a ModuloAdditionSpecification instance for testing."""
    return ModuloAdditionSpecification(
        model_dir=temp_results_dir,
        prime=17,  # Small prime for fast tests
        device='cpu',
        seed=42,
    )


# =============================================================================
# REQ_001: Configurable Checkpoint Epochs
# =============================================================================

class TestREQ001_ConfigurableCheckpointEpochs:
    """Tests for REQ_001 Conditions of Satisfaction."""

    def test_training_accepts_list_of_epoch_numbers(self, model_spec):
        """CoS: Training accepts a list of epoch numbers for checkpointing."""
        custom_epochs = [0, 25, 50, 75, 99]
        model = model_spec.train(num_epochs=100, checkpoint_epochs=custom_epochs)

        assert model is not None
        assert model_spec.checkpoint_epochs == custom_epochs

    def test_checkpoints_saved_only_at_specified_epochs(self, model_spec):
        """CoS: Checkpoints are saved only at the specified epochs."""
        custom_epochs = [10, 30, 50]
        model_spec.train(num_epochs=60, checkpoint_epochs=custom_epochs)

        # Check that only the specified checkpoints exist
        available = model_spec.get_available_checkpoints()
        assert available == custom_epochs

        # Verify no extra checkpoints were created
        checkpoint_files = os.listdir(model_spec.checkpoints_dir)
        assert len(checkpoint_files) == len(custom_epochs)

    def test_default_checkpoint_behavior_when_list_not_provided(self, model_spec):
        """CoS: If checkpoint list is not provided, falls back to reasonable default behavior."""
        # Train with fewer epochs than the default schedule covers
        model_spec.train(num_epochs=150, checkpoint_epochs=None)

        # Should use DEFAULT_CHECKPOINT_EPOCHS filtered to < 150
        expected_epochs = [e for e in DEFAULT_CHECKPOINT_EPOCHS if e < 150]
        assert model_spec.checkpoint_epochs == expected_epochs

    def test_checkpoint_list_configurable_per_training_run(self, temp_results_dir):
        """CoS: Checkpoint list can be configured per training run."""
        # First run with one checkpoint schedule
        spec1 = ModuloAdditionSpecification(
            model_dir=temp_results_dir,
            prime=17,
            device='cpu',
            seed=1,
        )
        spec1.train(num_epochs=50, checkpoint_epochs=[10, 20, 30])
        assert spec1.checkpoint_epochs == [10, 20, 30]

        # Second run with different checkpoint schedule
        spec2 = ModuloAdditionSpecification(
            model_dir=temp_results_dir,
            prime=17,
            device='cpu',
            seed=2,
        )
        spec2.train(num_epochs=50, checkpoint_epochs=[5, 15, 25, 35, 45])
        assert spec2.checkpoint_epochs == [5, 15, 25, 35, 45]

    def test_training_loop_handles_arbitrary_checkpoint_spacing(self, model_spec):
        """CoS: Training loop efficiently handles arbitrary checkpoint spacing."""
        # Test with irregular spacing
        irregular_epochs = [0, 1, 5, 10, 50, 51, 52, 99]
        model_spec.train(num_epochs=100, checkpoint_epochs=irregular_epochs)

        available = model_spec.get_available_checkpoints()
        assert available == irregular_epochs

    def test_epochs_beyond_num_epochs_are_filtered(self, model_spec):
        """Epochs >= num_epochs should be automatically filtered out."""
        epochs_with_invalid = [10, 50, 100, 200, 500]
        model_spec.train(num_epochs=100, checkpoint_epochs=epochs_with_invalid)

        # Only epochs < 100 should be saved
        assert model_spec.checkpoint_epochs == [10, 50]

    def test_default_checkpoint_schedule_structure(self):
        """Verify default checkpoint schedule has expected structure."""
        # Should have dense checkpoints in grokking region (5000-6000)
        grokking_checkpoints = [e for e in DEFAULT_CHECKPOINT_EPOCHS if 5000 <= e < 6000]
        early_checkpoints = [e for e in DEFAULT_CHECKPOINT_EPOCHS if e < 1000]

        # Grokking region should have more checkpoints per 1000 epochs
        grokking_density = len(grokking_checkpoints) / 1000
        early_density = len(early_checkpoints) / 1000

        assert grokking_density > early_density, "Grokking region should have denser checkpoints"


# =============================================================================
# REQ_002: Safetensors Model Persistence
# =============================================================================

class TestREQ002_SafetensorsPersistence:
    """Tests for REQ_002 Conditions of Satisfaction."""

    def test_checkpoint_saved_as_separate_safetensors_file(self, model_spec):
        """CoS: Each checkpoint saved as separate safetensors file immediately upon creation."""
        model_spec.train(num_epochs=50, checkpoint_epochs=[10, 20, 30])

        # Check that each checkpoint is a separate .safetensors file
        for epoch in [10, 20, 30]:
            checkpoint_path = os.path.join(
                model_spec.checkpoints_dir,
                f"checkpoint_epoch_{epoch:05d}.safetensors"
            )
            assert os.path.exists(checkpoint_path), f"Checkpoint at epoch {epoch} not found"
            assert checkpoint_path.endswith('.safetensors')

    def test_checkpoints_written_to_disk_during_training(self, model_spec):
        """CoS: Checkpoints written to disk during training (not accumulated in memory)."""
        # This is verified by the fact that checkpoints exist on disk after training
        # and that the class no longer has a model_checkpoints list accumulated during training
        model_spec.train(num_epochs=50, checkpoint_epochs=[10, 20, 30])

        # Checkpoints should exist on disk
        assert len(os.listdir(model_spec.checkpoints_dir)) == 3

        # The spec should not have accumulated checkpoints in memory
        # (model_checkpoints is only set when loading legacy format)
        assert not hasattr(model_spec, 'model_checkpoints') or model_spec.model_checkpoints is None

    def test_directory_structure_organizes_checkpoints_and_metadata(self, model_spec):
        """CoS: Directory structure organizes checkpoints and metadata clearly."""
        model_spec.train(num_epochs=50, checkpoint_epochs=[10, 20])

        # Verify expected directory structure exists
        assert os.path.isdir(model_spec.full_dir)
        assert os.path.isdir(model_spec.checkpoints_dir)
        assert os.path.isdir(model_spec.artifacts_dir)

        # Verify files are in expected locations
        assert os.path.exists(model_spec.model_path)
        assert os.path.exists(model_spec.metadata_path)
        assert os.path.exists(model_spec.config_path)

        # Verify checkpoints are in checkpoints subdirectory
        checkpoint_files = os.listdir(model_spec.checkpoints_dir)
        assert len(checkpoint_files) == 2
        assert all(f.endswith('.safetensors') for f in checkpoint_files)

    def test_training_metadata_saved_separately_as_json(self, model_spec):
        """CoS: Training metadata (train_losses, test_losses, train_indices, test_indices) saved separately."""
        model_spec.train(num_epochs=50, checkpoint_epochs=[10])

        # Verify metadata file exists and is valid JSON
        assert os.path.exists(model_spec.metadata_path)

        with open(model_spec.metadata_path, 'r') as f:
            metadata = json.load(f)

        # Verify required fields are present
        assert 'train_losses' in metadata
        assert 'test_losses' in metadata
        assert 'train_indices' in metadata
        assert 'test_indices' in metadata
        assert 'checkpoint_epochs' in metadata

        # Verify data integrity
        assert len(metadata['train_losses']) == 50
        assert len(metadata['test_losses']) == 50
        assert metadata['checkpoint_epochs'] == [10]

    def test_model_configuration_saved_in_readable_format(self, model_spec):
        """CoS: Model configuration saved in readable format."""
        model_spec.train(num_epochs=50, checkpoint_epochs=[10])

        # Verify config file exists and is valid JSON
        assert os.path.exists(model_spec.config_path)

        with open(model_spec.config_path, 'r') as f:
            config = json.load(f)

        # Verify key configuration fields are present
        assert 'n_layers' in config
        assert 'n_heads' in config
        assert 'd_model' in config
        assert 'prime' in config
        assert config['prime'] == 17

    def test_can_load_individual_checkpoint_by_epoch_number(self, model_spec):
        """CoS: Can load individual checkpoint by epoch number."""
        model_spec.train(num_epochs=50, checkpoint_epochs=[10, 20, 30])

        # Load specific checkpoint
        checkpoint = model_spec.load_checkpoint(20)

        # Verify it's a valid state dict
        assert isinstance(checkpoint, dict)
        assert any('W_' in key or 'b_' in key for key in checkpoint.keys())

    def test_load_checkpoint_raises_for_nonexistent_epoch(self, model_spec):
        """load_checkpoint should raise FileNotFoundError for missing checkpoints."""
        model_spec.train(num_epochs=50, checkpoint_epochs=[10, 30])

        with pytest.raises(FileNotFoundError):
            model_spec.load_checkpoint(20)  # Epoch 20 was not checkpointed

    def test_backward_compatible_with_legacy_pickle_format(self, temp_results_dir):
        """CoS: Backward compatible: can still load old pickle-based checkpoints for analysis."""
        # Create a legacy-format file
        spec = ModuloAdditionSpecification(
            model_dir=temp_results_dir,
            prime=17,
            device='cpu',
            seed=42,
        )
        model = spec.create_model()

        # Save in legacy format
        legacy_data = {
            "model": model.state_dict(),
            "config": model.cfg,
            "checkpoints": [model.state_dict()],
            "checkpoint_epochs": [0],
            "test_losses": [1.0] * 100,
            "train_losses": [1.0] * 100,
            "train_indices": torch.tensor([0, 1, 2]),
            "test_indices": torch.tensor([3, 4, 5]),
        }
        torch.save(legacy_data, spec.legacy_path)

        # Now load using load_from_file - should detect and load legacy format
        spec2 = ModuloAdditionSpecification(
            model_dir=temp_results_dir,
            prime=17,
            device='cpu',
            seed=42,
        )
        loaded_model = spec2.load_from_file()

        assert loaded_model is not None
        assert len(spec2.train_losses) == 100
        assert spec2.checkpoint_epochs == [0]

    def test_load_from_file_prefers_new_format_over_legacy(self, model_spec):
        """When both formats exist, load_from_file should prefer new safetensors format."""
        # Train to create new format
        model_spec.train(num_epochs=50, checkpoint_epochs=[10])

        # Create a legacy file with different data
        legacy_data = {
            "model": model_spec.model.state_dict(),
            "config": model_spec.model.cfg,
            "checkpoints": [],
            "checkpoint_epochs": [999],  # Different from new format
            "test_losses": [0.0] * 100,
            "train_losses": [0.0] * 100,
            "train_indices": torch.tensor([0]),
            "test_indices": torch.tensor([1]),
        }
        os.makedirs(os.path.dirname(model_spec.legacy_path), exist_ok=True)
        torch.save(legacy_data, model_spec.legacy_path)

        # Load - should get new format data (checkpoint_epochs = [10])
        spec2 = ModuloAdditionSpecification(
            model_dir=model_spec.model_dir,
            prime=17,
            device='cpu',
            seed=42,
        )
        spec2.load_from_file()

        assert spec2.checkpoint_epochs == [10], "Should load from new format, not legacy"

    def test_final_model_saved_as_safetensors(self, model_spec):
        """Final model should be saved as safetensors file."""
        model_spec.train(num_epochs=50, checkpoint_epochs=[10])

        assert os.path.exists(model_spec.model_path)
        assert model_spec.model_path.endswith('.safetensors')

    def test_get_available_checkpoints_returns_sorted_list(self, model_spec):
        """get_available_checkpoints should return sorted list of epoch numbers."""
        # Save checkpoints in non-sorted order (internally they'll be created in order anyway)
        model_spec.train(num_epochs=100, checkpoint_epochs=[50, 10, 30, 70])

        available = model_spec.get_available_checkpoints()
        assert available == [10, 30, 50, 70]
        assert available == sorted(available)

    def test_directory_naming_includes_prime_and_seed(self, temp_results_dir):
        """Directory naming should follow pattern: {model_name}_p{prime}_seed{seed}."""
        spec = ModuloAdditionSpecification(
            model_dir=temp_results_dir,
            prime=113,
            device='cpu',
            seed=999,
        )

        assert 'modulo_addition_p113_seed999' in spec.full_dir
        assert 'modulo_addition_p113_seed999' in spec.model_path


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests covering both requirements together."""

    def test_full_training_and_reload_cycle(self, model_spec):
        """Test complete cycle: train, save, reload, verify."""
        # Train
        custom_epochs = [0, 25, 49]
        model = model_spec.train(num_epochs=50, checkpoint_epochs=custom_epochs)
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Create new spec and reload
        spec2 = ModuloAdditionSpecification(
            model_dir=model_spec.model_dir,
            prime=17,
            device='cpu',
            seed=42,
        )
        reloaded_model = spec2.load_from_file()

        # Verify model weights match
        for key in original_state:
            assert torch.allclose(original_state[key], reloaded_model.state_dict()[key])

        # Verify metadata matches
        assert spec2.checkpoint_epochs == custom_epochs
        assert len(spec2.train_losses) == 50

    def test_checkpoint_weights_differ_across_epochs(self, model_spec):
        """Verify checkpoints at different epochs have different weights."""
        model_spec.train(num_epochs=100, checkpoint_epochs=[10, 50, 90])

        checkpoint_10 = model_spec.load_checkpoint(10)
        checkpoint_50 = model_spec.load_checkpoint(50)
        checkpoint_90 = model_spec.load_checkpoint(90)

        # Find a weight tensor that should change during training (not IGNORE which is always -inf)
        weight_keys = [k for k in checkpoint_10.keys() if 'W_' in k]
        key = weight_keys[0]

        # At least some checkpoints should differ (training should update weights)
        assert not torch.allclose(checkpoint_10[key], checkpoint_90[key]), \
            f"Weights for {key} should change between epoch 10 and 90"
