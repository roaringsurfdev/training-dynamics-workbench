"""Learned-Embedding MLP family for modular addition.

Architecture: two separate learned embeddings for inputs a and b,
summed into a shared d_embed-dimensional representation, then passed
through a single hidden layer with ReLU activation.

The sum embedding forces the network to encode a + b in a shared space,
creating representational pressure analogous to the transformer's
residual stream — but without the attention routing mechanism.

This is rung 2 of the architecture ladder:
  1. One-hot MLP  (no learned repr, no attention)  → no geometry
  2. Learned-embedding MLP (learned repr, no attention) → ?
  3. 1-layer Transformer (learned repr + attention)  → geometry

Input encoding: integer indices a, b ∈ {0, ..., p-1} (not one-hot).
The embedding layers handle encoding; output is logits over p classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from miscope.analysis.library import get_fourier_basis
from miscope.families.json_family import JsonModelFamily


class LearnedEmbeddingMLP(nn.Module):
    """Two-layer MLP with separate learned embeddings for a and b, summed.

    Architecture:
        embed_a: Embedding(p, d_embed)
        embed_b: Embedding(p, d_embed)
        hidden:  Linear(d_embed, d_hidden, bias=False) → ReLU
        output:  Linear(d_hidden, p, bias=False) → logits

    The two embeddings are summed before the hidden layer, forcing the
    model to encode the joint (a, b) representation in d_embed dimensions.
    """

    def __init__(
        self,
        vocab_size: int,
        d_embed: int,
        d_hidden: int,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.d_hidden = d_hidden

        self.embed_a = nn.Embedding(vocab_size, d_embed)
        self.embed_b = nn.Embedding(vocab_size, d_embed)
        self.W_in = nn.Linear(d_embed, d_hidden, bias=False)
        self.relu = nn.ReLU()
        self.W_out = nn.Linear(d_hidden, vocab_size, bias=False)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            a: Integer indices of shape (batch,) for first input
            b: Integer indices of shape (batch,) for second input

        Returns:
            Logits of shape (batch, vocab_size)
        """
        x = self.embed_a(a) + self.embed_b(b)
        hidden = self.relu(self.W_in(x))
        return self.W_out(hidden)


class LearnedEmbMLPActivationBundle:
    """ActivationBundle implementation for LearnedEmbeddingMLP.

    Captures hidden activations via a forward hook registered during
    run_forward_pass(). Exposes them via the ActivationBundle protocol.

    Weight access:
        W_in   — hidden layer weight (d_hidden, d_embed)
        W_out  — output layer weight (vocab_size, d_hidden)
        embed_a — input a embedding weights (vocab_size, d_embed)
        embed_b — input b embedding weights (vocab_size, d_embed)

    Note: W_E is intentionally NOT exposed. The dispatch logic used by all
    analyzers treats presence of W_E as a transformer signal. These embeddings
    use separate 'embed_a'/'embed_b' keys so the architecture is correctly
    identified as MLP-class (no W_E) while still exposing embedding weights.
    """

    _WEIGHT_LOOKUP = {
        "W_in": lambda m: m.W_in.weight,
        "W_out": lambda m: m.W_out.weight,
        "embed_a": lambda m: m.embed_a.weight,
        "embed_b": lambda m: m.embed_b.weight,
    }

    def __init__(
        self,
        model: LearnedEmbeddingMLP,
        hidden_acts: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        self._model = model
        self._hidden_acts = hidden_acts
        self._logits = logits

    def mlp_post(self, layer: int, position: int) -> torch.Tensor:
        """Return post-ReLU hidden activations.

        Args:
            layer: Must be 0 (the only hidden layer).
            position: Ignored — MLP has no sequence dimension.

        Returns:
            Tensor of shape (batch, d_hidden).
        """
        if layer != 0:
            raise ValueError(
                f"LearnedEmbMLPActivationBundle has one hidden layer (layer 0), got layer={layer}"
            )
        return self._hidden_acts

    def weight(self, name: str) -> torch.Tensor:
        """Return a named weight matrix.

        Supported: W_in, W_out, W_E.
        Raises KeyError for unsupported weight names.
        """
        if name not in self._WEIGHT_LOOKUP:
            raise KeyError(f"Weight '{name}' not available in LearnedEmbMLPActivationBundle")
        return self._WEIGHT_LOOKUP[name](self._model)

    def attention_pattern(self, layer: int) -> torch.Tensor:
        raise NotImplementedError("LearnedEmbMLPActivationBundle has no attention patterns")

    def residual_stream(self, layer: int, position: int, location: str) -> torch.Tensor:
        raise NotImplementedError("LearnedEmbMLPActivationBundle has no residual stream")

    def logits(self, position: int) -> torch.Tensor:
        """Return output logits.

        Args:
            position: Ignored — MLP has no sequence dimension.

        Returns:
            Tensor of shape (batch, vocab_size).
        """
        return self._logits

    def supports_site(self, extractor: str) -> bool:
        """Supports 'mlp' extraction only."""
        return extractor == "mlp"


class LearnedEmbMLPFamily(JsonModelFamily):
    """ModelFamily for the learned-embedding MLP on modular addition.

    Same task as the one-hot MLP and 1-layer transformer, but with
    learned embeddings and no attention or residual stream. The two
    input embeddings (a, b) are summed before the hidden layer.

    Domain parameters: prime, seed, data_seed
    Input: integer indices a, b ∈ {0, ..., p-1}
    Output: logits over p residue classes
    """

    def create_model(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> LearnedEmbeddingMLP:
        p = params["prime"]
        seed = params.get("seed", self.get_default_params().get("seed", 999))
        d_embed = self.architecture.get("d_embed", 16)
        d_hidden = self.architecture.get("d_hidden", 512)

        model = LearnedEmbeddingMLP(vocab_size=p, d_embed=d_embed, d_hidden=d_hidden, seed=seed)
        if device is not None:
            model = model.to(device)
        return model

    def generate_analysis_dataset(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate full (a, b) integer index grid for analysis.

        Returns:
            Tuple of (a_vals, b_vals), each shape (p²,) — integer indices.
        """
        p = params["prime"]
        a_vals = torch.arange(p).repeat_interleave(p)
        b_vals = torch.arange(p).repeat(p)
        if device is not None:
            a_vals = a_vals.to(device)
            b_vals = b_vals.to(device)
        return a_vals, b_vals

    def generate_training_dataset(
        self,
        params: dict[str, Any],
        training_fraction: float = 0.75,
        data_seed: int = 598,
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate train/test split for training.

        Returns:
            Tuple of (train_a, train_b, train_labels, test_a, test_b,
                     test_labels, train_indices, test_indices)
            Note: 8-tuple vs 6-tuple in one-hot MLP — returns separate a/b.
        """
        p = params["prime"]
        a_vals, b_vals = self.generate_analysis_dataset(params, device=device)
        labels = (a_vals + b_vals) % p

        torch.manual_seed(data_seed)
        indices = torch.randperm(p * p)
        if device is not None:
            indices = indices.to(device)
        cutoff = int(p * p * training_fraction)
        train_idx = indices[:cutoff]
        test_idx = indices[cutoff:]

        return (
            a_vals[train_idx],
            b_vals[train_idx],
            labels[train_idx],
            a_vals[test_idx],
            b_vals[test_idx],
            labels[test_idx],
            train_idx,
            test_idx,
        )

    def run_forward_pass(
        self,
        model: LearnedEmbeddingMLP,
        probe: tuple[torch.Tensor, torch.Tensor],
    ) -> LearnedEmbMLPActivationBundle:
        """Run a forward pass and return a LearnedEmbMLPActivationBundle.

        Args:
            model: LearnedEmbeddingMLP instance
            probe: Tuple of (a_vals, b_vals) integer index tensors

        Returns:
            LearnedEmbMLPActivationBundle with hidden activations and logits
        """
        a_vals, b_vals = probe
        captured: dict[str, torch.Tensor] = {}

        def _hook(module: nn.Module, inp: Any, output: torch.Tensor) -> None:
            captured["hidden"] = output

        hook = model.relu.register_forward_hook(_hook)
        try:
            with torch.inference_mode():
                logits = model(a_vals, b_vals)
        finally:
            hook.remove()

        return LearnedEmbMLPActivationBundle(model, captured["hidden"], logits)

    def prepare_analysis_context(
        self,
        params: dict[str, Any],
        device: str | torch.device,
    ) -> dict[str, Any]:
        """Prepare precomputed values for modular addition analysis."""
        p = params["prime"]
        fourier_basis, _ = get_fourier_basis(p, device)

        def loss_fn(model: LearnedEmbeddingMLP, probe: tuple[torch.Tensor, torch.Tensor]) -> float:
            a_vals, b_vals = probe
            labels = (a_vals + b_vals) % p
            with torch.no_grad():
                logits = model(a_vals, b_vals)
            log_probs = logits.log_softmax(dim=-1)
            loss = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1).mean()
            return loss.item()

        a_vals = torch.arange(p).repeat_interleave(p)
        b_vals = torch.arange(p).repeat(p)
        labels = ((a_vals + b_vals) % p).numpy()

        return {
            "params": params,
            "fourier_basis": fourier_basis,
            "loss_fn": loss_fn,
            "labels": labels,
        }

    def make_probe(
        self,
        params: dict[str, Any],
        inputs: list[list[int]],
        device: str | torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct integer index probe tensors from (a, b) input pairs.

        Returns:
            Tuple of (a_tensor, b_tensor), each shape (n,).
        """
        a_list = [pair[0] for pair in inputs]
        b_list = [pair[1] for pair in inputs]
        a_tensor = torch.tensor(a_list, dtype=torch.long)
        b_tensor = torch.tensor(b_list, dtype=torch.long)
        if device is not None:
            a_tensor = a_tensor.to(device)
            b_tensor = b_tensor.to(device)
        return a_tensor, b_tensor

    def get_training_config(self) -> dict[str, Any]:
        """Return default training hyperparameters.

        Calibrated to match He et al. (2602.16849) §3.3 settings used for
        the one-hot 2L MLP. Embedding parameters may behave differently
        under weight decay — monitor training curves and adjust if needed.
        d_embed=16 is chosen to sit below the transformer's observed W_E
        participation ratio (23.45) while staying above the theoretical
        floor (2|F|=8 for 4 frequencies at p=113).
        """
        return {
            "learning_rate": 1e-4,
            "weight_decay": 2.0,
            "betas": (0.9, 0.98),
            "num_epochs": 50000,
            "default_checkpoint_epochs": sorted(
                list(
                    set(
                        [
                            *range(0, 1500, 100),
                            *range(1500, 12000, 500),
                            *range(12000, 22000, 100),
                            *range(22000, 50000, 500),
                        ]
                    )
                )
            ),
        }


def load_learned_emb_mlp_family(
    model_families_dir: Path | str = "model_families",
) -> LearnedEmbMLPFamily:
    """Load the learned-embedding MLP family from the standard location."""
    family_json = (
        Path(model_families_dir) / "modulo_addition_learned_emb_mlp" / "family.json"
    )
    family = LearnedEmbMLPFamily.from_json(family_json)
    assert isinstance(family, LearnedEmbMLPFamily)
    return family
