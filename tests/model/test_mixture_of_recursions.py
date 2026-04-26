"""Tests for src/model/mixture_of_recursions.py

Covers all specified classes with 16+ test functions.

Test dimensions used throughout:
    d_model    = 16
    vocab_size = 16
    n_layers   = 2
    n_heads    = 4
    max_depth  = 3
    B          = 2
    T          = 6
"""

import math

import pytest
import torch

from src.model.mixture_of_recursions import (
    MixtureOfRecursionsLayer,
    MoRConfig,
    MoRLanguageModel,
    RecursionAnalyzer,
    RecursionDepthRouter,
    RecursiveTransformerBlock,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_SIZE = 16
N_LAYERS = 2
N_HEADS = 4
MAX_DEPTH = 3
B = 2
T = 6


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _router() -> RecursionDepthRouter:
    return RecursionDepthRouter(D_MODEL, MAX_DEPTH)


def _block() -> RecursiveTransformerBlock:
    return RecursiveTransformerBlock(D_MODEL, N_HEADS, MAX_DEPTH)


def _mor_layer() -> MixtureOfRecursionsLayer:
    return MixtureOfRecursionsLayer(D_MODEL, N_HEADS, MAX_DEPTH)


def _lm() -> MoRLanguageModel:
    return MoRLanguageModel(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_depth=MAX_DEPTH,
    )


def _x() -> torch.Tensor:
    return torch.randn(B, T, D_MODEL)


def _ids() -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (B, T))


# ===========================================================================
# RecursionDepthRouter
# ===========================================================================


class TestRecursionDepthRouter:
    def test_depths_shape(self) -> None:
        """forward() depths tensor has shape (B, T)."""
        router = _router()
        depths, _ = router(_x())
        assert depths.shape == (B, T), f"Expected ({B}, {T}), got {depths.shape}"

    def test_probs_shape(self) -> None:
        """forward() probs tensor has shape (B, T, max_depth)."""
        router = _router()
        _, probs = router(_x())
        assert probs.shape == (B, T, MAX_DEPTH)

    def test_probs_sum_to_one(self) -> None:
        """Probability distribution sums to 1 along the depth dimension."""
        router = _router()
        router.train(False)  # inference mode; no Gumbel noise
        with torch.no_grad():
            _, probs = router(_x())
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            f"Max deviation from 1: {(sums - 1).abs().max().item():.2e}"
        )

    def test_depths_in_valid_range(self) -> None:
        """Depth indices lie in [0, max_depth - 1]."""
        router = _router()
        depths, _ = router(_x())
        assert int(depths.min().item()) >= 0
        assert int(depths.max().item()) <= MAX_DEPTH - 1

    def test_expected_depth_shape(self) -> None:
        """expected_depth() returns a (B, T) tensor."""
        router = _router()
        _, probs = router(_x())
        exp_d = router.expected_depth(probs)
        assert exp_d.shape == (B, T), f"Expected ({B}, {T}), got {exp_d.shape}"

    def test_expected_depth_in_range(self) -> None:
        """expected_depth values lie in [0, max_depth - 1]."""
        router = _router()
        router.train(False)
        with torch.no_grad():
            _, probs = router(_x())
            exp_d = router.expected_depth(probs)
        assert float(exp_d.min().item()) >= -1e-6
        assert float(exp_d.max().item()) <= MAX_DEPTH - 1 + 1e-6

    def test_probs_shape_train_mode(self) -> None:
        """Probs shape is consistent in training mode (Gumbel-softmax path)."""
        router = _router()
        router.train(True)
        _, probs = router(_x())
        assert probs.shape == (B, T, MAX_DEPTH)


# ===========================================================================
# RecursiveTransformerBlock
# ===========================================================================


class TestRecursiveTransformerBlock:
    def test_apply_n_times_output_shape(self) -> None:
        """apply_n_times preserves (B, T, d_model) shape."""
        block = _block()
        out = block.apply_n_times(_x(), n=2)
        assert out.shape == (B, T, D_MODEL)

    def test_apply_n_times_n1_vs_n2_differ(self) -> None:
        """Outputs for n=1 and n=2 are numerically different."""
        block = _block()
        block.train(False)
        x = _x()
        with torch.no_grad():
            out1 = block.apply_n_times(x, n=1)
            out2 = block.apply_n_times(x, n=2)
        assert not torch.allclose(out1, out2, atol=1e-6), "n=1 and n=2 outputs should differ"

    def test_apply_n_times_n1_shape(self) -> None:
        """apply_n_times with n=1 returns correct shape."""
        block = _block()
        out = block.apply_n_times(_x(), n=1)
        assert out.shape == (B, T, D_MODEL)

    def test_apply_with_depths_soft_shape(self) -> None:
        """apply_with_depths returns (B, T, d_model) in soft mode."""
        block = _block()
        block.train(True)
        x = _x()
        depths = torch.zeros(B, T, dtype=torch.long)
        probs = torch.softmax(torch.randn(B, T, MAX_DEPTH), dim=-1)
        out = block.apply_with_depths(x, depths, probs=probs, mode="soft")
        assert out.shape == (B, T, D_MODEL)

    def test_apply_with_depths_hard_shape(self) -> None:
        """apply_with_depths returns (B, T, d_model) in hard mode."""
        block = _block()
        block.train(False)
        x = _x()
        depths = torch.randint(0, MAX_DEPTH, (B, T))
        with torch.no_grad():
            out = block.apply_with_depths(x, depths, mode="hard")
        assert out.shape == (B, T, D_MODEL)

    def test_apply_n_times_invalid_n_raises(self) -> None:
        """apply_n_times with n < 1 raises ValueError."""
        block = _block()
        with pytest.raises(ValueError):
            block.apply_n_times(_x(), n=0)


# ===========================================================================
# MixtureOfRecursionsLayer
# ===========================================================================


class TestMixtureOfRecursionsLayer:
    def test_forward_output_shape(self) -> None:
        """forward() output has shape (B, T, d_model)."""
        layer = _mor_layer()
        out, _ = layer(_x())
        assert out.shape == (B, T, D_MODEL)

    def test_forward_depth_probs_shape(self) -> None:
        """forward() depth_probs has shape (B, T, max_depth)."""
        layer = _mor_layer()
        _, probs = layer(_x())
        assert probs.shape == (B, T, MAX_DEPTH)

    def test_depth_regularizer_loss_scalar(self) -> None:
        """depth_regularizer_loss returns a scalar tensor."""
        layer = _mor_layer()
        probs = torch.softmax(torch.randn(B, T, MAX_DEPTH), dim=-1)
        loss = layer.depth_regularizer_loss(probs)
        assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"

    def test_depth_regularizer_loss_in_range(self) -> None:
        """depth_regularizer_loss lies in [0, 1]."""
        layer = _mor_layer()
        probs = torch.softmax(torch.randn(B, T, MAX_DEPTH), dim=-1)
        val = float(layer.depth_regularizer_loss(probs).item())
        assert 0.0 <= val <= 1.0 + 1e-6, f"Loss {val} outside [0, 1]"

    def test_gradient_flows(self) -> None:
        """Gradients propagate to block parameters via backward."""
        layer = _mor_layer()
        layer.train(True)
        x = _x().requires_grad_(True)
        out, probs = layer(x)
        loss = out.sum() + layer.depth_regularizer_loss(probs)
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum().item() > 0 for p in layer.parameters()
        )
        assert has_grad, "No non-zero gradients reached layer parameters"


# ===========================================================================
# MoRLanguageModel
# ===========================================================================


class TestMoRLanguageModel:
    def test_forward_logits_shape(self) -> None:
        """forward() logits have shape (B, T, vocab_size)."""
        model = _lm()
        logits, _ = model(_ids())
        assert logits.shape == (B, T, VOCAB_SIZE)

    def test_depth_info_length(self) -> None:
        """forward() depth_info list has n_layers entries."""
        model = _lm()
        _, depth_info = model(_ids())
        assert len(depth_info) == N_LAYERS

    def test_compute_loss_returns_3_tuple(self) -> None:
        """compute_loss returns a 3-element tuple."""
        model = _lm()
        model.train(True)
        result = model.compute_loss(_ids())
        assert len(result) == 3

    def test_total_loss_finite_scalar(self) -> None:
        """total_loss is a finite scalar tensor."""
        model = _lm()
        model.train(True)
        total, _, _ = model.compute_loss(_ids())
        assert total.shape == torch.Size([]), "total_loss should be scalar"
        assert math.isfinite(total.item()), f"total_loss not finite: {total.item()}"

    def test_compute_loss_lm_positive(self) -> None:
        """lm_loss is positive for a randomly initialised model."""
        model = _lm()
        model.train(True)
        _, lm_loss, _ = model.compute_loss(_ids())
        assert lm_loss.item() > 0.0

    def test_backward_does_not_raise(self) -> None:
        """Calling backward on total_loss completes without error."""
        model = _lm()
        model.train(True)
        total, _, _ = model.compute_loss(_ids())
        total.backward()

    def test_inference_forward(self) -> None:
        """Model runs in inference (non-training) mode without error."""
        model = _lm()
        model.train(False)
        with torch.no_grad():
            logits, depth_info = model(_ids())
        assert logits.shape == (B, T, VOCAB_SIZE)
        assert len(depth_info) == N_LAYERS


# ===========================================================================
# RecursionAnalyzer
# ===========================================================================


class TestRecursionAnalyzer:
    def test_compute_flop_ratio_in_range(self) -> None:
        """compute_flop_ratio is in (0, 1] for various mean depths."""
        for mean_d in [0.0, 0.5, 1.0, float(MAX_DEPTH - 1)]:
            ratio = RecursionAnalyzer.compute_flop_ratio(mean_d, MAX_DEPTH)
            assert 0.0 < ratio <= 1.0, f"Ratio {ratio} out of (0,1] for mean_depth={mean_d}"

    def test_compute_flop_ratio_max_equals_one(self) -> None:
        """Ratio is ~1.0 when mean_depth = max_depth - 1."""
        ratio = RecursionAnalyzer.compute_flop_ratio(float(MAX_DEPTH - 1), MAX_DEPTH)
        assert abs(ratio - 1.0) < 1e-6

    def test_depth_histogram_sums_to_one(self) -> None:
        """depth_histogram fraction values sum to 1."""
        probs = torch.softmax(torch.randn(B, T, MAX_DEPTH), dim=-1)
        hist = RecursionAnalyzer.depth_histogram(probs)
        total = sum(hist.values())
        assert abs(total - 1.0) < 1e-6

    def test_depth_histogram_keys(self) -> None:
        """depth_histogram has exactly max_depth integer keys."""
        probs = torch.softmax(torch.randn(B, T, MAX_DEPTH), dim=-1)
        hist = RecursionAnalyzer.depth_histogram(probs)
        assert set(hist.keys()) == set(range(MAX_DEPTH))

    def test_mean_depth_per_layer_length(self) -> None:
        """mean_depth_per_layer returns a list of length n_layers."""
        model = _lm()
        model.train(False)
        with torch.no_grad():
            _, di1 = model(_ids())
            _, di2 = model(_ids())
        means = RecursionAnalyzer.mean_depth_per_layer([di1, di2])
        assert len(means) == N_LAYERS

    def test_mean_depth_per_layer_values_in_range(self) -> None:
        """mean_depth_per_layer values lie in [0, max_depth - 1]."""
        model = _lm()
        model.train(False)
        with torch.no_grad():
            _, di = model(_ids())
        means = RecursionAnalyzer.mean_depth_per_layer([di])
        for i, m in enumerate(means):
            assert 0.0 <= m <= MAX_DEPTH - 1 + 1e-6, (
                f"Layer {i} mean {m} outside [0, {MAX_DEPTH - 1}]"
            )


# ===========================================================================
# MoRConfig
# ===========================================================================


class TestMoRConfig:
    def test_default_d_model(self) -> None:
        assert MoRConfig().d_model == 32

    def test_default_vocab_size(self) -> None:
        assert MoRConfig().vocab_size == 64

    def test_default_n_layers(self) -> None:
        assert MoRConfig().n_layers == 2

    def test_default_n_heads(self) -> None:
        assert MoRConfig().n_heads == 4

    def test_default_max_depth(self) -> None:
        assert MoRConfig().max_depth == 4

    def test_default_depth_reg_weight(self) -> None:
        assert abs(MoRConfig().depth_reg_weight - 0.01) < 1e-9

    def test_custom_values(self) -> None:
        cfg = MoRConfig(
            d_model=64, vocab_size=128, n_layers=4, n_heads=8, max_depth=6, depth_reg_weight=0.1
        )
        assert cfg.d_model == 64
        assert cfg.vocab_size == 128
        assert cfg.n_layers == 4
        assert cfg.n_heads == 8
        assert cfg.max_depth == 6
        assert abs(cfg.depth_reg_weight - 0.1) < 1e-9

    def test_config_builds_model(self) -> None:
        """A model built from MoRConfig attributes works end-to-end."""
        cfg = MoRConfig(
            d_model=D_MODEL,
            vocab_size=VOCAB_SIZE,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            max_depth=MAX_DEPTH,
        )
        model = MoRLanguageModel(
            d_model=cfg.d_model,
            vocab_size=cfg.vocab_size,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            max_depth=cfg.max_depth,
        )
        assert len(model.layers) == cfg.n_layers
        logits, _ = model(_ids())
        assert logits.shape[-1] == cfg.vocab_size
