"""Canonical component and hook names for miscope architectures.

This module is the platform's single source of truth for component names
and hook point names. Analyzers, renderers, and other consumers reference
the constants and helpers defined here; raw string literals like
``"blocks.0.mlp.hook_pre"`` are reserved for adapter implementations that
translate canonical names to a backend's spelling.

Source spec
-----------
The vocabulary mirrors the TransformerLens 3.0 Model Structure page:
https://transformerlensorg.github.io/TransformerLens/content/model_structure.html

Spec adopted: 2026-04-17 (TL 3.0 release).

Why adopt verbatim
------------------
TL 3.0's hierarchy — uniform ``hook_in`` / ``hook_out`` suffixes, explicit
Q / K / V / O sub-paths, separate ``ln1`` / ``attn`` / ``ln2`` / ``mlp``
block decomposition — is a clean conceptual design. Inventing a parallel
vocabulary would be churn for no gain and would make researcher onboarding
harder (they already know TL's names).

Adoption here is independent of the underlying TL runtime version. The
``TransformerLensAdapter`` holds a translation table from canonical names
to TL 2.x legacy hook strings; if the runtime moves to TL 3.x or off TL
entirely, the canonical names in this module do not change.
"""

# Top-level components
EMBED = "embed"
POS_EMBED = "pos_embed"
LN_FINAL = "ln_final"
UNEMBED = "unembed"
BLOCKS = "blocks"

# Block-internal components
LN1 = "ln1"
LN2 = "ln2"
ATTN = "attn"
MLP = "mlp"

# Attention sub-components
ATTN_Q = "q"
ATTN_K = "k"
ATTN_V = "v"
ATTN_O = "o"
ATTN_QKV = "qkv"

# MLP sub-components
MLP_IN = "in"
MLP_PRE = "pre"
MLP_OUT = "out"

# Hook point suffixes
HOOK_IN = "hook_in"
HOOK_OUT = "hook_out"
HOOK_PRE = "hook_pre"
HOOK_PATTERN = "hook_pattern"
HOOK_ATTN_SCORES = "hook_attn_scores"
HOOK_HIDDEN_STATES = "hook_hidden_states"


def block(layer: int) -> str:
    """Path to a block, e.g. ``block(0) == "blocks.0"``."""
    return f"{BLOCKS}.{layer}"


def block_component(layer: int, component: str) -> str:
    """Path to a component within a block, e.g. ``block_component(0, ATTN) == "blocks.0.attn"``."""
    return f"{block(layer)}.{component}"


def block_attn_subcomponent(layer: int, sub: str) -> str:
    """Path to an attention sub-component, e.g. ``block_attn_subcomponent(0, ATTN_Q) == "blocks.0.attn.q"``."""
    return f"{block(layer)}.{ATTN}.{sub}"


def block_mlp_subcomponent(layer: int, sub: str) -> str:
    """Path to an MLP sub-component, e.g. ``block_mlp_subcomponent(0, MLP_PRE) == "blocks.0.mlp.pre"``."""
    return f"{block(layer)}.{MLP}.{sub}"


def hook(*parts: str | int) -> str:
    """Compose a canonical hook path from arbitrary parts.

    Example:
        ``hook(BLOCKS, 0, ATTN, ATTN_Q, HOOK_OUT) == "blocks.0.attn.q.hook_out"``
    """
    return ".".join(str(p) for p in parts)
