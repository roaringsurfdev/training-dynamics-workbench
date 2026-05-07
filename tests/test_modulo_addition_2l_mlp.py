"""Tests for ModuloAddition2LMLPFamily and the underlying HookedOneHotMLP.

Migrated under REQ_113: standalone ``ModuloAddition2LMLP`` /
``ModuloAddition2LMLPActivationBundle`` are retired in favor of
``HookedOneHotMLP`` (canonical-name surface) and ``MLPBundle`` (legacy
ActivationBundle adapter).
"""

from __future__ import annotations

import pytest
import torch

from miscope.analysis.mlp_bundle import MLPBundle
from miscope.analysis.protocols import ActivationBundle
from miscope.architectures import (
    ActivationCache,
    HookedModel,
    HookedOneHotMLP,
    HookedOneHotMLPConfig,
)
from miscope.families.implementations.modulo_addition_2l_mlp import (
    ModuloAddition2LMLPFamily,
    load_modulo_addition_2l_mlp_family,
)

P = 13  # Small prime for fast tests


@pytest.fixture
def model() -> HookedOneHotMLP:
    return HookedOneHotMLP(HookedOneHotMLPConfig(vocab_size=P, d_hidden=64, seed=42))


@pytest.fixture
def probe() -> torch.Tensor:
    """One-hot encoded (a, b) pairs for all P² inputs."""
    a_vals = torch.arange(P).repeat_interleave(P)
    b_vals = torch.arange(P).repeat(P)
    one_hot_a = torch.zeros(P * P, P)
    one_hot_b = torch.zeros(P * P, P)
    one_hot_a.scatter_(1, a_vals.unsqueeze(1), 1.0)
    one_hot_b.scatter_(1, b_vals.unsqueeze(1), 1.0)
    return torch.cat([one_hot_a, one_hot_b], dim=1)


@pytest.fixture
def bundle(model: HookedOneHotMLP, probe: torch.Tensor) -> MLPBundle:
    with torch.inference_mode():
        logits, cache = model.run_with_cache(probe)
    return MLPBundle(model, cache, logits)


@pytest.fixture
def family() -> ModuloAddition2LMLPFamily:
    return load_modulo_addition_2l_mlp_family("model_families")


@pytest.fixture
def params() -> dict:
    return {"prime": P, "seed": 42, "data_seed": 598}


# ── HookedOneHotMLP ──────────────────────────────────────────────────────────


class TestHookedOneHotMLP:
    def test_is_hooked_model(self, model):
        assert isinstance(model, HookedModel)

    def test_forward_output_shape(self, model, probe):
        with torch.inference_mode():
            out = model(probe)
        assert out.shape == (P * P, P)

    def test_seedable_initialization(self):
        m1 = HookedOneHotMLP(HookedOneHotMLPConfig(vocab_size=P, d_hidden=64, seed=7))
        m2 = HookedOneHotMLP(HookedOneHotMLPConfig(vocab_size=P, d_hidden=64, seed=7))
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_different_seeds_differ(self):
        m1 = HookedOneHotMLP(HookedOneHotMLPConfig(vocab_size=P, d_hidden=64, seed=1))
        m2 = HookedOneHotMLP(HookedOneHotMLPConfig(vocab_size=P, d_hidden=64, seed=2))
        differs = any(not torch.equal(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters()))
        assert differs

    def test_weight_shapes(self, model):
        assert model.W_in.weight.shape == (64, 2 * P)
        assert model.W_out.weight.shape == (P, 64)

    def test_no_bias(self, model):
        assert model.W_in.bias is None
        assert model.W_out.bias is None

    def test_canonical_weight_access(self, model):
        assert torch.equal(model.get_weight("blocks.0.mlp.in.W"), model.W_in.weight)
        assert torch.equal(model.get_weight("blocks.0.mlp.out.W"), model.W_out.weight)

    def test_no_embed_weight_published(self, model):
        for forbidden in ("embed.W_E", "embed.embed_a", "embed.embed_b"):
            with pytest.raises(KeyError):
                model.get_weight(forbidden)

    def test_run_with_cache_populates_canonical_hooks(self, model, probe):
        logits, cache = model.run_with_cache(probe)
        assert isinstance(cache, ActivationCache)
        assert cache["blocks.0.mlp.hook_pre"].shape == (P * P, 64)
        assert cache["blocks.0.mlp.hook_out"].shape == (P * P, 64)
        assert cache["unembed.hook_out"].shape == (P * P, P)
        assert torch.equal(cache["unembed.hook_out"], logits)


# ── MLPBundle (legacy ActivationBundle compatibility) ───────────────────────


class TestMLPBundleOnOneHotMLP:
    def test_implements_activation_bundle_protocol(self, bundle):
        assert isinstance(bundle, ActivationBundle)

    def test_mlp_post_layer0(self, bundle):
        hidden = bundle.mlp_post(0, -1)
        assert hidden.shape == (P * P, 64)

    def test_weight_w_in(self, bundle):
        w = bundle.weight("W_in")
        assert w.shape == (64, 2 * P)

    def test_weight_w_out(self, bundle):
        w = bundle.weight("W_out")
        assert w.shape == (P, 64)

    def test_weight_transformer_names_raise_key_error(self, bundle):
        for name in ("W_E", "W_pos", "W_Q", "W_K", "W_V", "W_O", "W_U"):
            with pytest.raises(KeyError):
                bundle.weight(name)

    def test_weight_embedding_names_raise_key_error(self, bundle):
        # One-hot architecture publishes neither embed_a nor embed_b
        for name in ("embed_a", "embed_b"):
            with pytest.raises(KeyError):
                bundle.weight(name)

    def test_attention_pattern_raises_not_implemented(self, bundle):
        with pytest.raises(NotImplementedError):
            bundle.attention_pattern(0)

    def test_residual_stream_raises_not_implemented(self, bundle):
        with pytest.raises(NotImplementedError):
            bundle.residual_stream(0, -1, "pre")

    def test_logits_shape(self, bundle):
        logits = bundle.logits(-1)
        assert logits.shape == (P * P, P)

    def test_logits_position_ignored(self, bundle):
        """logits() should return the same tensor regardless of position arg."""
        assert torch.equal(bundle.logits(0), bundle.logits(-1))
        assert torch.equal(bundle.logits(0), bundle.logits(99))

    def test_supports_site(self, bundle):
        assert bundle.supports_site("mlp")
        assert not bundle.supports_site("residual")
        assert not bundle.supports_site("attention")


# ── ModuloAddition2LMLPFamily ────────────────────────────────────────────────


class TestModuloAddition2LMLPFamily:
    def test_name(self, family):
        assert family.name == "modulo_addition_2layer_mlp"

    def test_create_model_returns_hooked_one_hot_mlp(self, family, params):
        model = family.create_model(params)
        assert isinstance(model, HookedOneHotMLP)
        assert isinstance(model, HookedModel)
        assert model.vocab_size == P

    def test_generate_analysis_dataset_shape(self, family, params):
        dataset = family.generate_analysis_dataset(params)
        assert dataset.shape == (P * P, 2 * P)
        assert dataset.dtype == torch.float32

    def test_generate_analysis_dataset_is_one_hot(self, family, params):
        dataset = family.generate_analysis_dataset(params)
        # Each row sums to 2.0 (one 1 in a-half, one 1 in b-half)
        assert torch.allclose(dataset.sum(dim=1), torch.full((P * P,), 2.0))

    def test_generate_training_dataset_shapes(self, family, params):
        train_d, train_l, test_d, test_l, tr_idx, te_idx = family.generate_training_dataset(params)
        total = P * P
        n_train = len(tr_idx)
        n_test = len(te_idx)
        assert n_train + n_test == total
        assert train_d.shape == (n_train, 2 * P)
        assert test_d.shape == (n_test, 2 * P)

    def test_generate_training_dataset_reproducible(self, family, params):
        r1 = family.generate_training_dataset(params)
        r2 = family.generate_training_dataset(params)
        assert torch.equal(r1[4], r2[4])  # train_indices

    def test_run_forward_pass_returns_mlp_bundle(self, family, params):
        model = family.create_model(params)
        probe = family.generate_analysis_dataset(params)
        bundle = family.run_forward_pass(model, probe)
        assert isinstance(bundle, MLPBundle)

    def test_run_forward_pass_bundle_shapes(self, family, params):
        model = family.create_model(params)
        probe = family.generate_analysis_dataset(params)
        bundle = family.run_forward_pass(model, probe)
        assert bundle.mlp_post(0, -1).shape == (P * P, 512)
        assert bundle.logits(-1).shape == (P * P, P)

    def test_prepare_analysis_context_has_fourier_basis(self, family, params):
        ctx = family.prepare_analysis_context(params, "cpu")
        assert "fourier_basis" in ctx
        assert "params" in ctx
        assert "loss_fn" in ctx
        assert ctx["fourier_basis"].shape == (P, P)

    def test_prepare_analysis_context_loss_fn(self, family, params):
        model = family.create_model(params)
        probe = family.generate_analysis_dataset(params)
        ctx = family.prepare_analysis_context(params, "cpu")
        loss = ctx["loss_fn"](model, probe)
        assert isinstance(loss, float)
        assert loss > 0

    def test_make_probe(self, family, params):
        probe = family.make_probe(params, [[3, 5], [0, 12]])
        assert probe.shape == (2, 2 * P)
        # Row 0: a=3 → one-hot[3]=1, b=5 → one-hot[P+5]=1
        assert probe[0, 3] == 1.0
        assert probe[0, P + 5] == 1.0
        assert probe[0].sum() == 2.0

    def test_analyzers_excludes_transformer_specific(self, family):
        transformer_only = {
            "attention_freq",
            "attention_fourier",
            "attention_patterns",
            "fourier_nucleation",
        }
        assert not transformer_only.intersection(set(family.analyzers))

    def test_analyzers_includes_neuron_activations(self, family):
        assert "neuron_activations" in family.analyzers

    def test_analyzers_includes_parameter_snapshot(self, family):
        assert "parameter_snapshot" in family.analyzers

    def test_secondary_analyzers_includes_neuron_fourier(self, family):
        assert "neuron_fourier" in family.secondary_analyzers

    def test_cross_epoch_analyzers_includes_neuron_group_pca(self, family):
        assert "neuron_group_pca" in family.cross_epoch_analyzers
