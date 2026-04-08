"""Tests for LearnedEmbeddingMLP, LearnedEmbMLPActivationBundle, and LearnedEmbMLPFamily."""

from __future__ import annotations

import pytest
import torch

from miscope.analysis.protocols import ActivationBundle
from miscope.families.implementations.learned_emb_mlp import (
    LearnedEmbMLPActivationBundle,
    LearnedEmbeddingMLP,
    LearnedEmbMLPFamily,
    load_learned_emb_mlp_family,
)

P = 13       # Small prime for fast tests
D_EMBED = 8  # Small embedding dim for tests
D_HIDDEN = 32


@pytest.fixture
def model() -> LearnedEmbeddingMLP:
    return LearnedEmbeddingMLP(vocab_size=P, d_embed=D_EMBED, d_hidden=D_HIDDEN, seed=42)


@pytest.fixture
def probe() -> tuple[torch.Tensor, torch.Tensor]:
    """Integer index (a, b) pairs for all P² inputs."""
    a_vals = torch.arange(P).repeat_interleave(P)
    b_vals = torch.arange(P).repeat(P)
    return a_vals, b_vals


@pytest.fixture
def bundle(model, probe) -> LearnedEmbMLPActivationBundle:
    a_vals, b_vals = probe
    captured: dict = {}

    def hook(m, inp, output):
        captured["hidden"] = output

    h = model.relu.register_forward_hook(hook)
    with torch.inference_mode():
        logits = model(a_vals, b_vals)
    h.remove()
    return LearnedEmbMLPActivationBundle(model, captured["hidden"], logits)


@pytest.fixture
def family() -> LearnedEmbMLPFamily:
    return load_learned_emb_mlp_family("model_families")


@pytest.fixture
def params() -> dict:
    return {"prime": P, "seed": 42, "data_seed": 598}


# ── LearnedEmbeddingMLP ───────────────────────────────────────────────────────


class TestLearnedEmbeddingMLP:
    def test_forward_output_shape(self, model, probe):
        a_vals, b_vals = probe
        with torch.inference_mode():
            out = model(a_vals, b_vals)
        assert out.shape == (P * P, P)

    def test_seedable_initialization(self):
        m1 = LearnedEmbeddingMLP(vocab_size=P, d_embed=D_EMBED, d_hidden=D_HIDDEN, seed=7)
        m2 = LearnedEmbeddingMLP(vocab_size=P, d_embed=D_EMBED, d_hidden=D_HIDDEN, seed=7)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            assert torch.equal(p1, p2)

    def test_different_seeds_differ(self):
        m1 = LearnedEmbeddingMLP(vocab_size=P, d_embed=D_EMBED, d_hidden=D_HIDDEN, seed=1)
        m2 = LearnedEmbeddingMLP(vocab_size=P, d_embed=D_EMBED, d_hidden=D_HIDDEN, seed=2)
        differs = any(not torch.equal(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters()))
        assert differs

    def test_embedding_shapes(self, model):
        assert model.embed_a.weight.shape == (P, D_EMBED)
        assert model.embed_b.weight.shape == (P, D_EMBED)

    def test_weight_shapes(self, model):
        assert model.W_in.weight.shape == (D_HIDDEN, D_EMBED)
        assert model.W_out.weight.shape == (P, D_HIDDEN)

    def test_no_bias(self, model):
        assert model.W_in.bias is None
        assert model.W_out.bias is None

    def test_sum_embedding_symmetry(self, model):
        """embed_a(3) + embed_b(5) should differ from embed_a(5) + embed_b(3)
        unless embed_a == embed_b (which they shouldn't be after seeded init)."""
        a = torch.tensor([3])
        b = torch.tensor([5])
        with torch.inference_mode():
            sum_ab = model.embed_a(a) + model.embed_b(b)
            sum_ba = model.embed_a(b) + model.embed_b(a)
        # The sums will generally differ (a and b embeddings are independent)
        assert not torch.equal(sum_ab, sum_ba)


# ── LearnedEmbMLPActivationBundle ─────────────────────────────────────────────


class TestLearnedEmbMLPActivationBundle:
    def test_implements_activation_bundle_protocol(self, bundle):
        assert isinstance(bundle, ActivationBundle)

    def test_mlp_post_layer0(self, bundle):
        hidden = bundle.mlp_post(0, -1)
        assert hidden.shape == (P * P, D_HIDDEN)

    def test_mlp_post_wrong_layer_raises(self, bundle):
        with pytest.raises(ValueError, match="layer 0"):
            bundle.mlp_post(1, -1)

    def test_weight_w_in(self, bundle):
        w = bundle.weight("W_in")
        assert w.shape == (D_HIDDEN, D_EMBED)

    def test_weight_w_out(self, bundle):
        w = bundle.weight("W_out")
        assert w.shape == (P, D_HIDDEN)

    def test_weight_embed_a(self, bundle):
        w = bundle.weight("embed_a")
        assert w.shape == (P, D_EMBED)

    def test_weight_embed_b(self, bundle):
        w = bundle.weight("embed_b")
        assert w.shape == (P, D_EMBED)

    def test_weight_w_e_raises_key_error(self, bundle):
        """W_E is intentionally not exposed — dispatch logic uses W_E presence as
        transformer signal; this architecture must not masquerade as a transformer."""
        with pytest.raises(KeyError):
            bundle.weight("W_E")

    def test_weight_unsupported_raises_key_error(self, bundle):
        for name in ("W_pos", "W_Q", "W_K", "W_V", "W_O", "W_U"):
            with pytest.raises(KeyError):
                bundle.weight(name)

    def test_attention_pattern_raises_not_implemented(self, bundle):
        with pytest.raises(NotImplementedError):
            bundle.attention_pattern(0)

    def test_residual_stream_raises_not_implemented(self, bundle):
        with pytest.raises(NotImplementedError):
            bundle.residual_stream(0, -1, "pre")

    def test_logits_shape(self, bundle):
        assert bundle.logits(-1).shape == (P * P, P)

    def test_logits_position_ignored(self, bundle):
        assert torch.equal(bundle.logits(0), bundle.logits(-1))

    def test_supports_site_mlp(self, bundle):
        assert bundle.supports_site("mlp") is True
        assert bundle.supports_site("resid_pre") is False
        assert bundle.supports_site("attn") is False


# ── LearnedEmbMLPFamily ───────────────────────────────────────────────────────


class TestLearnedEmbMLPFamily:
    def test_name(self, family):
        assert family.name == "modulo_addition_learned_emb_mlp"

    def test_create_model_returns_correct_type(self, family, params):
        model = family.create_model(params)
        assert isinstance(model, LearnedEmbeddingMLP)
        assert model.vocab_size == P

    def test_create_model_d_embed_from_architecture(self, family, params):
        model = family.create_model(params)
        # family.json specifies d_embed=16; test with the loaded family
        assert model.d_embed == family.architecture.get("d_embed", 16)

    def test_generate_analysis_dataset_shape(self, family, params):
        a_vals, b_vals = family.generate_analysis_dataset(params)
        assert a_vals.shape == (P * P,)
        assert b_vals.shape == (P * P,)
        assert a_vals.dtype == torch.long
        assert b_vals.dtype == torch.long

    def test_generate_analysis_dataset_covers_all_pairs(self, family, params):
        a_vals, b_vals = family.generate_analysis_dataset(params)
        pairs = set(zip(a_vals.tolist(), b_vals.tolist()))
        assert len(pairs) == P * P

    def test_generate_training_dataset_returns_8_tuple(self, family, params):
        result = family.generate_training_dataset(params)
        assert len(result) == 8

    def test_generate_training_dataset_shapes(self, family, params):
        train_a, train_b, train_l, test_a, test_b, test_l, tr_idx, te_idx = (
            family.generate_training_dataset(params)
        )
        assert len(tr_idx) + len(te_idx) == P * P
        assert train_a.shape == train_b.shape == train_l.shape == (len(tr_idx),)
        assert test_a.shape == test_b.shape == test_l.shape == (len(te_idx),)

    def test_generate_training_dataset_reproducible(self, family, params):
        r1 = family.generate_training_dataset(params)
        r2 = family.generate_training_dataset(params)
        assert torch.equal(r1[6], r2[6])  # train_indices

    def test_run_forward_pass_returns_bundle(self, family, params):
        model = family.create_model(params)
        probe = family.generate_analysis_dataset(params)
        bundle = family.run_forward_pass(model, probe)
        assert isinstance(bundle, LearnedEmbMLPActivationBundle)

    def test_run_forward_pass_bundle_shapes(self, family, params):
        model = family.create_model(params)
        probe = family.generate_analysis_dataset(params)
        bundle = family.run_forward_pass(model, probe)
        d_hidden = family.architecture.get("d_hidden", 512)
        assert bundle.mlp_post(0, -1).shape == (P * P, d_hidden)
        assert bundle.logits(-1).shape == (P * P, P)

    def test_prepare_analysis_context_keys(self, family, params):
        ctx = family.prepare_analysis_context(params, "cpu")
        assert "fourier_basis" in ctx
        assert "params" in ctx
        assert "loss_fn" in ctx

    def test_prepare_analysis_context_fourier_basis_shape(self, family, params):
        ctx = family.prepare_analysis_context(params, "cpu")
        assert ctx["fourier_basis"].shape == (P, P)

    def test_prepare_analysis_context_loss_fn(self, family, params):
        model = family.create_model(params)
        probe = family.generate_analysis_dataset(params)
        ctx = family.prepare_analysis_context(params, "cpu")
        loss = ctx["loss_fn"](model, probe)
        assert isinstance(loss, float)
        assert loss > 0

    def test_make_probe_returns_tuple(self, family, params):
        a_t, b_t = family.make_probe(params, [[3, 5], [0, 12]])
        assert a_t.shape == (2,)
        assert b_t.shape == (2,)
        assert a_t[0].item() == 3
        assert b_t[0].item() == 5

    def test_get_variant_directory_name(self, family, params):
        name = family.get_variant_directory_name(params)
        assert name == f"modulo_addition_learned_emb_mlp_p{P}_seed42_dseed598"

    def test_analyzers_includes_neuron_activations(self, family):
        assert "neuron_activations" in family.analyzers

    def test_analyzers_includes_parameter_snapshot(self, family):
        assert "parameter_snapshot" in family.analyzers

    def test_analyzers_excludes_transformer_specific(self, family):
        transformer_only = {
            "attention_freq",
            "attention_fourier",
            "attention_patterns",
            "fourier_nucleation",
        }
        assert not transformer_only.intersection(set(family.analyzers))

    def test_secondary_analyzers_includes_neuron_fourier(self, family):
        assert "neuron_fourier" in family.secondary_analyzers

    def test_cross_epoch_analyzers_includes_neuron_group_pca(self, family):
        assert "neuron_group_pca" in family.cross_epoch_analyzers

    def test_get_training_config_keys(self, family):
        cfg = family.get_training_config()
        for key in ("learning_rate", "weight_decay", "betas", "num_epochs",
                    "default_checkpoint_epochs"):
            assert key in cfg

    def test_get_training_config_he_et_al_settings(self, family):
        """Verify hyperparameters match He et al. (2602.16849)."""
        cfg = family.get_training_config()
        assert cfg["learning_rate"] == 1e-4
        assert cfg["weight_decay"] == 2.0
        assert cfg["num_epochs"] == 50000
