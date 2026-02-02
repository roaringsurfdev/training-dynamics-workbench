"""Utility functions for the dashboard."""

import json
from pathlib import Path
from typing import Any


def discover_trained_models(base_path: str = "results/") -> list[dict[str, Any]]:
    """Scan results directory for trained models.

    Args:
        base_path: Root directory to scan for models.

    Returns:
        List of dicts with:
            - path: Full path to model directory
            - display_name: Human-readable name for dropdown
            - config: Model configuration dict
            - has_artifacts: Whether analysis has been run
    """
    models = []
    base = Path(base_path)

    if not base.exists():
        return models

    # Pattern: results/{model_type}/{model_type}_p{prime}_seed{seed}/
    for model_type_dir in base.iterdir():
        if not model_type_dir.is_dir():
            continue

        for run_dir in model_type_dir.iterdir():
            if not run_dir.is_dir():
                continue

            config_path = run_dir / "config.json"
            metadata_path = run_dir / "metadata.json"

            if config_path.exists() and metadata_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)

                    # Check for artifacts
                    artifacts_dir = run_dir / "artifacts"
                    has_artifacts = artifacts_dir.exists() and any(artifacts_dir.glob("*.npz"))

                    # Generate display name
                    prime = config.get("prime", config.get("n_ctx", "?"))
                    seed = config.get("model_seed", config.get("seed", "?"))
                    display_name = f"p={prime}, seed={seed}"
                    if has_artifacts:
                        display_name += " [analyzed]"

                    models.append(
                        {
                            "path": str(run_dir),
                            "display_name": display_name,
                            "config": config,
                            "has_artifacts": has_artifacts,
                        }
                    )
                except (json.JSONDecodeError, OSError):
                    # Skip malformed entries
                    continue

    return sorted(models, key=lambda m: m["display_name"])


def get_model_choices(models: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Convert model list to Gradio dropdown choices.

    Args:
        models: List from discover_trained_models().

    Returns:
        List of (display_name, path) tuples for gr.Dropdown.
    """
    return [(m["display_name"], m["path"]) for m in models]


def parse_checkpoint_epochs(text: str) -> list[int]:
    """Parse comma-separated checkpoint epochs from text input.

    Args:
        text: Comma-separated string of epoch numbers.

    Returns:
        Sorted list of unique epoch numbers.

    Raises:
        ValueError: If parsing fails.
    """
    if not text.strip():
        return []

    epochs = []
    for part in text.split(","):
        part = part.strip()
        if part:
            epochs.append(int(part))

    return sorted(set(epochs))


def validate_training_params(
    modulus: int,
    model_seed: int,
    data_seed: int,
    train_fraction: float,
    num_epochs: int,
    checkpoint_str: str,
    save_path: str,
) -> tuple[bool, str]:
    """Validate training parameters.

    Returns:
        (is_valid, error_message) tuple.
    """
    if modulus < 2:
        return False, "Modulus must be >= 2"

    if not isinstance(model_seed, int) or model_seed < 0:
        return False, "Model seed must be a non-negative integer"

    if not isinstance(data_seed, int) or data_seed < 0:
        return False, "Data seed must be a non-negative integer"

    if not 0.0 < train_fraction < 1.0:
        return False, "Training fraction must be between 0 and 1"

    if num_epochs < 1:
        return False, "Number of epochs must be >= 1"

    try:
        epochs = parse_checkpoint_epochs(checkpoint_str)
        if epochs and max(epochs) >= num_epochs:
            return False, f"Checkpoint epoch {max(epochs)} >= num_epochs {num_epochs}"
    except ValueError as e:
        return False, f"Invalid checkpoint epochs: {e}"

    if not save_path.strip():
        return False, "Save path cannot be empty"

    return True, ""
