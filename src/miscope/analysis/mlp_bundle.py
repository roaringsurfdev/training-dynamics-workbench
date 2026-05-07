"""MLPBundle — ActivationBundle adapter for the HookedMLP families.

Wraps a (HookedModel, ActivationCache, logits) triple from a one-hot or
embedding-MLP forward pass into the legacy ``ActivationBundle`` protocol
so analyzers that haven't migrated to the canonical-name surface yet
(REQ_114) keep working. The bundle does no compute of its own — every
read is a pass-through to the canonical cache or
``model.get_weight(canonical_name)``.

Replaces the deleted ``ModuloAddition2LMLPActivationBundle`` and
``ModuloAdditionEmbedMLPActivationBundle`` (REQ_113).

Activations on these architectures have no sequence dimension — a one-
hot MLP receives ``(batch, 2p)`` and the embedding MLP receives
``(batch, 2)``. Hidden activations are ``(batch, d_hidden)``. The
bundle's ``mlp_post(layer, position)`` method ignores ``position`` to
match the legacy semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from miscope.core import architecture as canonical_hooks
from miscope.core import weights as canonical_weights

if TYPE_CHECKING:
    from miscope.architectures import ActivationCache, HookedModel


# Translation: legacy weight names → canonical paths. The legacy MLP
# bundle accepted ``W_in``, ``W_out``, ``embed_a``, ``embed_b``; this
# table covers all of them so analyzer code that still uses TL-style
# names keeps reading the right tensor.
_LEGACY_TO_CANONICAL_WEIGHT: dict[str, str] = {
    "W_in": canonical_weights.block_mlp_weight(0, canonical_hooks.MLP_IN),
    "W_out": canonical_weights.block_mlp_weight(0, canonical_hooks.MLP_OUT),
    "embed_a": canonical_weights.EMBED_A_PATH,
    "embed_b": canonical_weights.EMBED_B_PATH,
}


def _canonical_mlp_out(layer: int) -> str:
    return canonical_hooks.hook(
        canonical_hooks.BLOCKS, layer, canonical_hooks.MLP, canonical_hooks.HOOK_OUT
    )


class MLPBundle:
    """Legacy ``ActivationBundle`` view over a HookedMLP forward pass.

    Args:
        model: ``HookedOneHotMLP`` or ``HookedEmbeddingMLP`` instance.
        cache: Canonical-name-keyed cache from ``model.run_with_cache``.
        logits: Output logits tensor of shape ``(batch, vocab_size)``.
    """

    def __init__(
        self,
        model: HookedModel,
        cache: ActivationCache,
        logits: torch.Tensor,
    ) -> None:
        self._model = model
        self._cache = cache
        self._logits = logits

    # ------------------------------------------------------------------
    # ActivationBundle protocol
    # ------------------------------------------------------------------

    def mlp_post(self, layer: int, position: int) -> torch.Tensor:
        """Post-activation MLP neurons.

        Returns ``(batch, d_hidden)``. ``position`` is ignored — these
        architectures have no sequence dimension.
        """
        return self._cache[_canonical_mlp_out(layer)]

    def residual_stream(
        self,
        layer: int,
        position: int,
        location: str,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "MLPBundle has no residual stream (no-residual architecture)."
        )

    def attention_pattern(self, layer: int) -> torch.Tensor:
        raise NotImplementedError(
            "MLPBundle has no attention patterns (no-attention architecture)."
        )

    def weight(self, name: str) -> torch.Tensor:
        canonical = _LEGACY_TO_CANONICAL_WEIGHT.get(name)
        if canonical is None:
            raise KeyError(
                f"Weight {name!r} not available in MLPBundle. "
                f"Legacy names supported: {sorted(_LEGACY_TO_CANONICAL_WEIGHT)}."
            )
        try:
            return self._model.get_weight(canonical)
        except KeyError:
            # Underlying model doesn't publish this canonical weight
            # (e.g., embed_a/embed_b on a one-hot MLP). Surface it as a
            # KeyError on the legacy name so the caller sees a single,
            # actionable message.
            raise KeyError(
                f"Weight {name!r} not available on this architecture. "
                f"Underlying canonical path {canonical!r} is not "
                f"published by {type(self._model).__name__}."
            ) from None

    def logits(self, position: int) -> torch.Tensor:
        """Output logits. ``position`` is ignored — no sequence dimension."""
        return self._logits

    def supports_site(self, extractor: str) -> bool:
        """Supports ``'mlp'`` extraction only — no residual stream, no attention."""
        return extractor == "mlp"

    # ------------------------------------------------------------------
    # Escape hatch
    # ------------------------------------------------------------------

    @property
    def raw_model(self) -> HookedModel:
        return self._model
