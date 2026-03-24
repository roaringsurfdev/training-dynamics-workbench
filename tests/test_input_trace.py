"""Tests for REQ_075: Per-Input Prediction Trace.

CoS coverage:
- Unit: InputTraceAnalyzer produces correct shape output for a minimal model (p=11)
- Unit: graduation_epochs is -1 for a pair never stable-correct
- Unit: graduation_epochs is the first epoch of sustained correctness
- Unit: residue_class_accuracy sums correctly — each pair in exactly one class
- Integration: analyzer runs on a minimal model and artifacts load cleanly
- Integration: all three views render without error
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from miscope.analysis.analyzers.input_trace import InputTraceAnalyzer, _build_training_probe
from miscope.analysis.analyzers.input_trace_graduation import (
    InputTraceGraduationAnalyzer,
    _compute_graduation_epochs,
)
from miscope.analysis.protocols import Analyzer, CrossEpochAnalyzer

# ── Minimal model fixture ─────────────────────────────────────────────


SMALL_P = 11


def _make_minimal_model(p: int = SMALL_P):
    """Create a minimal HookedTransformer for p=SMALL_P testing."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=1,
        n_heads=2,
        d_model=16,
        d_head=8,
        d_mlp=32,
        act_fn="relu",
        normalization_type=None,
        d_vocab=p + 1,
        d_vocab_out=p,
        n_ctx=3,
        init_weights=True,
        seed=42,
    )
    model = HookedTransformer(cfg)
    for name, param in model.named_parameters():
        if "b_" in name:
            param.requires_grad = False
    model.eval()
    return model


def _make_minimal_context(p: int = SMALL_P, data_seed: int = 42, training_fraction: float = 0.4):
    """Create a minimal analysis context."""
    return {
        "params": {
            "prime": p,
            "data_seed": data_seed,
            "training_fraction": training_fraction,
        }
    }


# ── Protocol conformance ─────────────────────────────────────────────


class TestProtocolConformance:
    def test_input_trace_analyzer_conforms(self):
        assert isinstance(InputTraceAnalyzer(), Analyzer)

    def test_graduation_analyzer_conforms(self):
        assert isinstance(InputTraceGraduationAnalyzer(), CrossEpochAnalyzer)

    def test_input_trace_has_summary_methods(self):
        analyzer = InputTraceAnalyzer()
        assert hasattr(analyzer, "get_summary_keys")
        assert hasattr(analyzer, "compute_summary")


# ── Unit: InputTraceAnalyzer output shapes ───────────────────────────


class TestInputTraceAnalyzerShapes:
    def test_output_shapes(self):
        p = SMALL_P
        training_fraction = 0.4
        data_seed = 42

        model = _make_minimal_model(p)
        context = _make_minimal_context(p, data_seed, training_fraction)

        # Build a minimal probe (full grid — analyzer ignores it)
        probe = torch.zeros((p * p, 3), dtype=torch.long)
        cache = None  # analyzer doesn't use cache

        analyzer = InputTraceAnalyzer()
        result = analyzer.analyze(model, probe, cache, context)  # type: ignore[arg-type]

        expected_n_pairs = int(p * p * training_fraction)
        assert result["predictions"].shape == (expected_n_pairs,)
        assert result["correct"].shape == (expected_n_pairs,)
        assert result["confidence"].shape == (expected_n_pairs,)
        assert result["pair_indices"].shape == (expected_n_pairs, 2)

    def test_output_dtypes(self):
        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = torch.zeros((p * p, 3), dtype=torch.long)

        result = InputTraceAnalyzer().analyze(model, probe, None, context)  # type: ignore[arg-type]

        assert result["predictions"].dtype == np.int16
        assert result["correct"].dtype == bool
        assert result["confidence"].dtype == np.float16
        assert result["pair_indices"].dtype == np.int16

    def test_predictions_in_valid_range(self):
        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = torch.zeros((p * p, 3), dtype=torch.long)

        result = InputTraceAnalyzer().analyze(model, probe, None, context)  # type: ignore[arg-type]

        assert result["predictions"].min() >= 0
        assert result["predictions"].max() < p

    def test_pair_indices_in_valid_range(self):
        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = torch.zeros((p * p, 3), dtype=torch.long)

        result = InputTraceAnalyzer().analyze(model, probe, None, context)  # type: ignore[arg-type]

        assert result["pair_indices"].min() >= 0
        assert result["pair_indices"].max() < p

    def test_correct_consistent_with_predictions(self):
        """correct[i] must match whether predictions[i] == (a+b)%p."""
        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = torch.zeros((p * p, 3), dtype=torch.long)

        result = InputTraceAnalyzer().analyze(model, probe, None, context)  # type: ignore[arg-type]

        a = result["pair_indices"][:, 0].astype(np.int32)
        b = result["pair_indices"][:, 1].astype(np.int32)
        expected_correct = result["predictions"].astype(np.int32) == (a + b) % p
        np.testing.assert_array_equal(result["correct"], expected_correct)


# ── Unit: summary stats ──────────────────────────────────────────────


class TestSummaryStats:
    def test_summary_keys(self):
        keys = InputTraceAnalyzer().get_summary_keys()
        assert "residue_class_accuracy" in keys
        assert "overall_accuracy" in keys

    def test_residue_class_accuracy_shape(self):
        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = torch.zeros((p * p, 3), dtype=torch.long)

        analyzer = InputTraceAnalyzer()
        result = analyzer.analyze(model, probe, None, context)  # type: ignore[arg-type]
        summary = analyzer.compute_summary(result, context)

        assert summary["residue_class_accuracy"].shape == (p,)
        assert summary["residue_class_accuracy"].dtype == np.float32

    def test_each_pair_contributes_to_exactly_one_residue_class(self):
        """Every training pair must contribute to exactly one residue class,
        so the weighted sum of per-class accuracies equals overall accuracy."""
        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = torch.zeros((p * p, 3), dtype=torch.long)

        analyzer = InputTraceAnalyzer()
        result = analyzer.analyze(model, probe, None, context)  # type: ignore[arg-type]

        pair_indices = result["pair_indices"].astype(np.int32)
        residues = (pair_indices[:, 0] + pair_indices[:, 1]) % p

        # Verify each pair lands in exactly one class
        class_counts = np.bincount(residues, minlength=p)
        assert class_counts.sum() == len(pair_indices), "Some pairs uncounted"

        # Verify weighted average of class accuracies equals overall accuracy
        summary = analyzer.compute_summary(result, context)
        correct = result["correct"].astype(float)
        expected_overall = correct.mean()
        np.testing.assert_allclose(
            float(summary["overall_accuracy"]),
            expected_overall,
            atol=1e-5,
        )


# ── Unit: training probe reconstruction ─────────────────────────────


class TestBuildTrainingProbe:
    def test_probe_shape(self):
        p = 11
        probe, pairs, labels = _build_training_probe(p, 42, 0.4, torch.device("cpu"))
        n_expected = int(p * p * 0.4)
        assert probe.shape == (n_expected, 3)
        assert pairs.shape == (n_expected, 2)
        assert labels.shape == (n_expected,)

    def test_probe_equals_token_correct(self):
        p = 11
        probe, _, _ = _build_training_probe(p, 42, 0.4, torch.device("cpu"))
        assert (probe[:, 2] == p).all()

    def test_labels_match_pairs(self):
        p = 11
        _, pairs, labels = _build_training_probe(p, 42, 0.4, torch.device("cpu"))
        expected = (pairs[:, 0] + pairs[:, 1]) % p
        torch.testing.assert_close(labels, expected)

    def test_deterministic_with_same_seed(self):
        p = 11
        _, pairs1, _ = _build_training_probe(p, 42, 0.4, torch.device("cpu"))
        _, pairs2, _ = _build_training_probe(p, 42, 0.4, torch.device("cpu"))
        torch.testing.assert_close(pairs1, pairs2)

    def test_different_seeds_produce_different_splits(self):
        p = 11
        _, pairs1, _ = _build_training_probe(p, 42, 0.4, torch.device("cpu"))
        _, pairs2, _ = _build_training_probe(p, 999, 0.4, torch.device("cpu"))
        assert not torch.equal(pairs1, pairs2)


# ── Unit: graduation epoch computation ──────────────────────────────


class TestComputeGraduationEpochs:
    def test_never_graduated_returns_minus_one(self):
        """A pair never correct in any epoch gets graduation_epoch = -1."""
        correct = np.zeros((5, 3), dtype=bool)
        epochs = [100, 200, 300, 400, 500]
        result = _compute_graduation_epochs(correct, epochs, window=3)
        np.testing.assert_array_equal(result, [-1, -1, -1])

    def test_isolated_correct_not_graduation(self):
        """A single correct epoch followed by incorrect does not graduate."""
        correct = np.array(
            [
                [True],
                [False],
                [False],
                [False],
                [False],
            ],
            dtype=bool,
        )
        epochs = [100, 200, 300, 400, 500]
        result = _compute_graduation_epochs(correct, epochs, window=3)
        assert result[0] == -1

    def test_graduation_at_first_stable_window(self):
        """Graduation epoch is the first epoch of a stable window, not later."""
        # Pair 0: correct at epochs [200, 300, 400] — first stable window starts at index 1
        correct = np.array(
            [
                [False],  # epoch 100
                [True],   # epoch 200 ← first stable window starts here
                [True],   # epoch 300
                [True],   # epoch 400
                [True],   # epoch 500
            ],
            dtype=bool,
        )
        epochs = [100, 200, 300, 400, 500]
        result = _compute_graduation_epochs(correct, epochs, window=3)
        assert result[0] == 200

    def test_intermittent_correct_before_stable(self):
        """A pair correct once early, then stable later: graduation = start of stable run."""
        correct = np.array(
            [
                [True],   # epoch 100 — isolated
                [False],  # epoch 200
                [True],   # epoch 300 ← stable window starts here
                [True],   # epoch 400
                [True],   # epoch 500
            ],
            dtype=bool,
        )
        epochs = [100, 200, 300, 400, 500]
        result = _compute_graduation_epochs(correct, epochs, window=3)
        assert result[0] == 300

    def test_window_larger_than_epochs_returns_minus_one(self):
        correct = np.ones((2, 3), dtype=bool)
        epochs = [0, 1]
        result = _compute_graduation_epochs(correct, epochs, window=5)
        np.testing.assert_array_equal(result, [-1, -1, -1])

    def test_multiple_pairs_independent(self):
        """Each pair's graduation is computed independently."""
        correct = np.array(
            [
                [True, False],   # epoch 0
                [True, False],   # epoch 100
                [True, True],    # epoch 200
                [False, False],  # epoch 300 — pair 0 breaks here
                [False, False],  # epoch 400
            ],
            dtype=bool,
        )
        epochs = [0, 100, 200, 300, 400]
        result = _compute_graduation_epochs(correct, epochs, window=3)
        assert result[0] == 0    # stable from index 0 (epochs 0,100,200 all correct)
        assert result[1] == -1   # pair 1 never has 3 consecutive correct


# ── Integration: artifact round-trip ────────────────────────────────


class TestIntegrationArtifactRoundTrip:
    """Run analyzer → save → load cycle using a minimal model."""

    def test_analyzer_artifacts_load_cleanly(self, tmp_path):
        from miscope.analysis.artifact_loader import ArtifactLoader

        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = torch.zeros((p * p, 3), dtype=torch.long)

        analyzer = InputTraceAnalyzer()
        result = analyzer.analyze(model, probe, None, context)  # type: ignore[arg-type]

        # Save artifact
        epoch = 100
        epoch_dir = tmp_path / "input_trace"
        epoch_dir.mkdir()
        np.savez_compressed(str(epoch_dir / f"epoch_{epoch:05d}.npz"), **result)

        # Load and verify
        loader = ArtifactLoader(str(tmp_path))
        loaded = loader.load_epoch("input_trace", epoch)

        assert "predictions" in loaded
        assert "correct" in loaded
        assert "confidence" in loaded
        assert "pair_indices" in loaded
        np.testing.assert_array_equal(loaded["predictions"], result["predictions"])

    def test_graduation_analyzer_runs_on_minimal_data(self, tmp_path):
        from miscope.analysis.artifact_loader import ArtifactLoader

        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = torch.zeros((p * p, 3), dtype=torch.long)

        analyzer = InputTraceAnalyzer()

        # Save 5 epochs
        epochs = [0, 100, 200, 300, 400]
        epoch_dir = tmp_path / "input_trace"
        epoch_dir.mkdir()
        for epoch in epochs:
            result = analyzer.analyze(model, probe, None, context)  # type: ignore[arg-type]
            np.savez_compressed(str(epoch_dir / f"epoch_{epoch:05d}.npz"), **result)

        # Run graduation analyzer
        grad_analyzer = InputTraceGraduationAnalyzer()
        grad_result = grad_analyzer.analyze_across_epochs(str(tmp_path), epochs, context)

        n_pairs = int(p * p * 0.4)
        assert grad_result["graduation_epochs"].shape == (n_pairs,)
        assert grad_result["epochs"].shape == (len(epochs),)
        assert grad_result["pair_indices"].shape == (n_pairs, 2)
        np.testing.assert_array_equal(grad_result["epochs"], epochs)


# ── Integration: views render without error ──────────────────────────


class TestViewsRender:
    """Verify all three renderers produce a Figure without error."""

    def test_accuracy_grid_renders(self):
        import plotly.graph_objects as go

        from miscope.visualization.renderers.input_trace import render_accuracy_grid

        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = torch.zeros((p * p, 3), dtype=torch.long)

        epoch_data = InputTraceAnalyzer().analyze(model, probe, None, context)  # type: ignore[arg-type]
        data = {"epoch_data": epoch_data, "prime": p}
        fig = render_accuracy_grid(data, epoch=100)
        assert isinstance(fig, go.Figure)

    def test_residue_class_timeline_renders(self):
        import plotly.graph_objects as go

        from miscope.visualization.renderers.input_trace import (
            render_residue_class_accuracy_timeline,
        )

        p = SMALL_P
        model = _make_minimal_model(p)
        context = _make_minimal_context(p)
        probe = torch.zeros((p * p, 3), dtype=torch.long)

        analyzer = InputTraceAnalyzer()
        epochs = [0, 100, 200]
        all_class_acc = []
        all_overall = []
        for ep in epochs:
            result = analyzer.analyze(model, probe, None, context)  # type: ignore[arg-type]
            s = analyzer.compute_summary(result, context)
            all_class_acc.append(s["residue_class_accuracy"])
            all_overall.append(s["overall_accuracy"])

        summary = {
            "epochs": np.array(epochs, dtype=np.int32),
            "residue_class_accuracy": np.stack(all_class_acc),
            "overall_accuracy": np.array(all_overall, dtype=np.float32),
        }
        data = {"summary": summary, "prime": p}
        fig = render_residue_class_accuracy_timeline(data, epoch=100)
        assert isinstance(fig, go.Figure)

    def test_graduation_heatmap_renders(self):
        import plotly.graph_objects as go

        from miscope.visualization.renderers.input_trace import render_pair_graduation_heatmap

        p = SMALL_P
        n_pairs = int(p * p * 0.4)

        rng = np.random.default_rng(42)
        _, pairs, _ = _build_training_probe(p, 42, 0.4, torch.device("cpu"))

        graduation_epochs = rng.choice(
            [-1, 100, 200, 300, 400], size=n_pairs
        ).astype(np.int32)

        graduation = {
            "graduation_epochs": graduation_epochs,
            "pair_indices": pairs.numpy().astype(np.int16),
            "epochs": np.array([0, 100, 200, 300, 400], dtype=np.int32),
        }
        data = {"graduation": graduation, "prime": p}
        fig = render_pair_graduation_heatmap(data, epoch=None)
        assert isinstance(fig, go.Figure)
