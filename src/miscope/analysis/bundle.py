"""TransformerLensBundle — ActivationBundle implementation for transformer models.

Wraps a (model, cache, logits) triple into the ActivationBundle protocol
so legacy analyzers can access activations and weights through a stable
surface during the REQ_112 / REQ_114 migration.

REQ_112 dual-mode operation
---------------------------
Production code paths (via ``family.run_forward_pass``) construct the
bundle from a :class:`miscope.architectures.HookedTransformer` and a
canonical-name :class:`miscope.architectures.ActivationCache`. Tests
that pre-date REQ_112 still construct the bundle from a raw TL
``HookedTransformer`` and a TL ``ActivationCache``. Both shapes work:
the bundle detects which mode it's in at construction time and routes
reads accordingly. After REQ_114 migrates all analyzers (and their
tests) to consume the canonical-name surface directly, this bundle and
its dual-mode logic are retired.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from miscope.architectures import ActivationCache as MiActivationCache
from miscope.architectures import HookedModel
from miscope.core import architecture as canonical_hooks
from miscope.core import weights as canonical_weights

if TYPE_CHECKING:
    from miscope.architectures import HookedTransformer

# Legacy-name → canonical-name translation (used in canonical mode).
_LEGACY_TO_CANONICAL_WEIGHT: dict[str, str] = {
    "W_E": canonical_weights.EMBED_WEIGHT,
    "W_pos": canonical_weights.POS_EMBED_WEIGHT,
    "W_U": canonical_weights.UNEMBED_WEIGHT,
    "W_Q": canonical_weights.block_attn_weight(0, canonical_hooks.ATTN_Q),
    "W_K": canonical_weights.block_attn_weight(0, canonical_hooks.ATTN_K),
    "W_V": canonical_weights.block_attn_weight(0, canonical_hooks.ATTN_V),
    "W_O": canonical_weights.block_attn_weight(0, canonical_hooks.ATTN_O),
    "W_in": canonical_weights.block_mlp_weight(0, canonical_hooks.MLP_IN),
    "W_out": canonical_weights.block_mlp_weight(0, canonical_hooks.MLP_OUT),
}

_LEGACY_RESID_TO_CANONICAL: dict[str, str] = {
    "resid_pre": canonical_hooks.hook(
        canonical_hooks.BLOCKS, 0, canonical_hooks.HOOK_IN
    ),
    "resid_post": canonical_hooks.hook(
        canonical_hooks.BLOCKS, 0, canonical_hooks.HOOK_OUT
    ),
    "attn_out": canonical_hooks.hook(
        canonical_hooks.BLOCKS, 0, canonical_hooks.ATTN, canonical_hooks.HOOK_OUT
    ),
}


# Legacy-mode weight lookup (used when the bundle holds a raw TL model
# without ``get_weight``). Mirrors the pre-REQ_112 mechanism so legacy
# tests continue to pass without migration.
_LEGACY_WEIGHT_LOOKUP: dict[str, Any] = {
    "W_E": lambda m: m.embed.W_E,
    "W_pos": lambda m: m.pos_embed.W_pos,
    "W_Q": lambda m: m.blocks[0].attn.W_Q,
    "W_K": lambda m: m.blocks[0].attn.W_K,
    "W_V": lambda m: m.blocks[0].attn.W_V,
    "W_O": lambda m: m.blocks[0].attn.W_O,
    "W_in": lambda m: m.blocks[0].mlp.W_in,
    "W_out": lambda m: m.blocks[0].mlp.W_out,
    "W_U": lambda m: m.unembed.W_U,
}


def _canonical_mlp_out(layer: int) -> str:
    return canonical_hooks.hook(
        canonical_hooks.BLOCKS, layer, canonical_hooks.MLP, canonical_hooks.HOOK_OUT
    )


def _canonical_attn_pattern(layer: int) -> str:
    return canonical_hooks.hook(
        canonical_hooks.BLOCKS,
        layer,
        canonical_hooks.ATTN,
        canonical_hooks.HOOK_PATTERN,
    )


def _layer_substituted(canonical: str, layer: int) -> str:
    """Rewrite the layer index in a canonical block path."""
    if layer == 0:
        return canonical
    parts = canonical.split(".")
    parts[1] = str(layer)
    return ".".join(parts)


class TransformerLensBundle:
    """Zero-cost wrapper over (model, cache, logits) for transformer models.

    Args:
        model: Either ``miscope.architectures.HookedTransformer`` (canonical
            mode, production path) or a raw ``transformer_lens.HookedTransformer``
            (legacy mode, exercised by pre-REQ_112 tests).
        cache: Either ``miscope.architectures.ActivationCache`` (canonical
            mode) or a TL ``ActivationCache`` (legacy mode).
        logits: Raw logits tensor (batch, seq_len, vocab) from the same forward
            pass. May be ``None`` for legacy tests that bypass forward.
    """

    def __init__(
        self,
        model: Any,
        cache: Any,
        logits: torch.Tensor | None,
    ) -> None:
        self._model = model
        self._cache = cache
        self._logits = logits
        # Canonical mode requires both a HookedModel and our ActivationCache.
        # Either-or doesn't make sense (a HookedModel always produces a
        # canonical cache; a TL model always produces a TL cache).
        self._canonical_mode = isinstance(model, HookedModel) and isinstance(
            cache, MiActivationCache
        )

    # ------------------------------------------------------------------
    # ActivationBundle protocol
    # ------------------------------------------------------------------

    def mlp_post(self, layer: int, position: int) -> torch.Tensor:
        """Post-activation MLP neurons. Returns (batch, d_mlp)."""
        if self._canonical_mode:
            acts = self._cache[_canonical_mlp_out(layer)]
        else:
            acts = self._cache["post", layer, "mlp"]
        return acts[:, position, :]

    def residual_stream(
        self,
        layer: int,
        position: int,
        location: str,
    ) -> torch.Tensor:
        """Residual stream at a site. Returns (batch, d_model)."""
        if self._canonical_mode:
            canonical = _LEGACY_RESID_TO_CANONICAL.get(location)
            if canonical is None:
                raise KeyError(
                    f"Unknown residual-stream location {location!r}. "
                    f"Expected one of {sorted(_LEGACY_RESID_TO_CANONICAL)}."
                )
            acts = self._cache[_layer_substituted(canonical, layer)]
        else:
            acts = self._cache[location, layer]
        return acts[:, position, :]

    def attention_pattern(self, layer: int) -> torch.Tensor:
        """Attention weights (post-softmax). Returns (batch, n_heads, seq_to, seq_from)."""
        if self._canonical_mode:
            return self._cache[_canonical_attn_pattern(layer)]
        return self._cache["pattern", layer]

    def weight(self, name: str) -> torch.Tensor:
        """Named weight matrix. Raises KeyError for unknown names."""
        if self._canonical_mode:
            canonical = _LEGACY_TO_CANONICAL_WEIGHT.get(name)
            if canonical is None:
                raise KeyError(
                    f"Unknown weight name: {name!r}. "
                    f"Expected one of {sorted(_LEGACY_TO_CANONICAL_WEIGHT)}"
                )
            return self._model.get_weight(canonical)
        legacy_lookup = _LEGACY_WEIGHT_LOOKUP.get(name)
        if legacy_lookup is None:
            raise KeyError(
                f"Unknown weight name: {name!r}. "
                f"Expected one of {sorted(_LEGACY_WEIGHT_LOOKUP)}"
            )
        return legacy_lookup(self._model)

    def logits(self, position: int) -> torch.Tensor:
        """Logits at a token position. Returns (batch, vocab_size)."""
        return self._logits[:, position, :]

    def supports_site(self, extractor: str) -> bool:
        """TransformerLens bundles support all extraction types."""
        return True

    # ------------------------------------------------------------------
    # Escape hatch for analyzers requiring direct model access
    # (e.g. LandscapeFlatnessAnalyzer which perturbs model parameters)
    # ------------------------------------------------------------------

    @property
    def raw_model(self) -> HookedTransformer:
        """Direct access to the underlying model. Use only when the bundle
        protocol is insufficient (e.g., parameter perturbation)."""
        return self._model
