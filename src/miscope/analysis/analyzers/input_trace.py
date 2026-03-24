"""Input Trace Analyzer — REQ_075.

Per-checkpoint predictions on training pairs, enabling residue-class
graduation analysis and pair-level learning structure visualization.
"""

from typing import Any

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache


class InputTraceAnalyzer:
    """Records per-pair predictions on the training set at each checkpoint.

    The probe passed by the pipeline is the full p² analysis grid — this
    analyzer reconstructs the training subset from context['params'] and
    runs a separate forward pass on those pairs only.

    Per-epoch artifact keys:
        predictions   int16   (n_pairs,)    argmax of output logits
        correct       bool    (n_pairs,)    predictions == true labels
        confidence    float16 (n_pairs,)    max softmax probability
        pair_indices  int16   (n_pairs, 2)  (a, b) for each training pair
    """

    name = "input_trace"
    description = "Per-pair predictions on training set at each checkpoint"

    def analyze(
        self,
        model: HookedTransformer,
        probe: torch.Tensor,
        cache: ActivationCache,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Run predictions on training pairs.

        Args:
            model: The model loaded with checkpoint weights
            probe: Full p² analysis grid (unused — training pairs reconstructed
                   from context['params'] to match the exact training split)
            cache: Activation cache (unused — separate forward pass on training pairs)
            context: Family-provided analysis context containing 'params' with
                     'prime', 'data_seed', 'training_fraction'

        Returns:
            Dict with 'predictions', 'correct', 'confidence', 'pair_indices'
        """
        params = context["params"]
        p = int(params["prime"])
        data_seed = int(params.get("data_seed", 598))
        training_fraction = float(params.get("training_fraction", 0.3))
        device = next(model.parameters()).device

        train_probe, train_pairs, true_labels = _build_training_probe(
            p, data_seed, training_fraction, device
        )

        with torch.no_grad():
            logits = model(train_probe)  # (n_pairs, n_ctx, p)

        last_logits = logits[:, -1, :]  # (n_pairs, p)
        probs = last_logits.softmax(dim=-1)
        predictions = last_logits.argmax(dim=-1)
        correct = predictions == true_labels
        confidence = probs.max(dim=-1).values

        return {
            "predictions": predictions.cpu().numpy().astype(np.int16),
            "correct": correct.cpu().numpy(),
            "confidence": confidence.cpu().numpy().astype(np.float16),
            "pair_indices": train_pairs.cpu().numpy().astype(np.int16),
        }

    def get_summary_keys(self) -> list[str]:
        """Declare per-epoch summary statistics accumulated across checkpoints."""
        return ["residue_class_accuracy", "overall_accuracy"]

    def compute_summary(
        self, result: dict[str, np.ndarray], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute per-epoch aggregate accuracy metrics.

        Args:
            result: Per-epoch artifact dict from analyze()
            context: Family-provided analysis context

        Returns:
            Dict with 'residue_class_accuracy' (p,) and 'overall_accuracy' scalar
        """
        p = int(context["params"]["prime"])
        correct = result["correct"]
        pair_indices = result["pair_indices"].astype(np.int32)

        residue = (pair_indices[:, 0] + pair_indices[:, 1]) % p
        residue_class_accuracy = np.zeros(p, dtype=np.float32)
        for c in range(p):
            mask = residue == c
            if mask.any():
                residue_class_accuracy[c] = correct[mask].mean()

        overall_accuracy = float(correct.mean())

        return {
            "residue_class_accuracy": residue_class_accuracy,
            "overall_accuracy": np.float32(overall_accuracy),
        }


def _build_training_probe(
    p: int,
    data_seed: int,
    training_fraction: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct the training set probe using the same split logic as training.

    Uses the same RNG state as generate_training_dataset to guarantee that
    the reconstructed split is identical to the one the model was trained on.

    Args:
        p: The prime modulus
        data_seed: Random seed for train/test split
        training_fraction: Fraction of pairs used for training
        device: Device for tensors

    Returns:
        Tuple of (train_probe, train_pairs, true_labels)
            train_probe:  (n_pairs, 3) — [a, b, p] inputs for model
            train_pairs:  (n_pairs, 2) — [a, b] for each training pair
            true_labels:  (n_pairs,)   — (a + b) % p for each training pair
    """
    a_all = torch.arange(p).repeat_interleave(p)  # (p^2,)
    b_all = torch.arange(p).repeat(p)             # (p^2,)

    torch.manual_seed(data_seed)
    indices = torch.randperm(p * p)
    cutoff = int(p * p * training_fraction)
    train_indices = indices[:cutoff]

    a_train = a_all[train_indices]
    b_train = b_all[train_indices]
    train_pairs = torch.stack([a_train, b_train], dim=1)  # (n_pairs, 2)

    equals_col = torch.full((len(train_pairs), 1), p, dtype=torch.long)
    train_probe = torch.cat(
        [train_pairs.long(), equals_col], dim=1
    ).to(device)  # (n_pairs, 3)

    true_labels = (a_train + b_train) % p
    true_labels = true_labels.to(device)

    return train_probe, train_pairs, true_labels
