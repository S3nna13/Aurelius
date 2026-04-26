"""Tests for early exit inference module."""

from __future__ import annotations

import torch

from src.inference.early_exit import (
    EarlyExitConfig,
    EarlyExitWrapper,
    ExitClassifier,
    compute_confidence,
    train_exit_classifiers,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_model() -> AureliusTransformer:
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=4,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


def _make_wrapper(
    model: AureliusTransformer,
    exit_thresholds: list[float] | None = None,
) -> EarlyExitWrapper:
    cfg = EarlyExitConfig(
        exit_layers=[1, 2],
        exit_thresholds=exit_thresholds if exit_thresholds is not None else [0.9, 0.85],
        min_exit_layer=0,
        confidence_metric="max_prob",
        use_highway=True,
    )
    return EarlyExitWrapper(
        model=model,
        config=cfg,
        d_model=64,
        vocab_size=256,
    )


# ---------------------------------------------------------------------------
# 1. EarlyExitConfig defaults
# ---------------------------------------------------------------------------


def test_early_exit_config_defaults():
    cfg = EarlyExitConfig()
    assert cfg.exit_thresholds == [0.9, 0.85, 0.8]
    assert cfg.exit_layers == [4, 8, 12]
    assert cfg.min_exit_layer == 2
    assert cfg.confidence_metric == "max_prob"
    assert cfg.use_highway is True


# ---------------------------------------------------------------------------
# 2–5. compute_confidence
# ---------------------------------------------------------------------------


def test_compute_confidence_max_prob_range():
    torch.manual_seed(0)
    logits = torch.randn(8, 256)
    conf = compute_confidence(logits, "max_prob")
    assert conf.shape == (8,)
    assert (conf >= 0.0).all() and (conf <= 1.0).all()


def test_compute_confidence_entropy_range():
    torch.manual_seed(0)
    logits = torch.randn(8, 256)
    conf = compute_confidence(logits, "entropy")
    assert conf.shape == (8,)
    assert (conf >= 0.0).all() and (conf <= 1.0 + 1e-5).all()  # allow tiny fp error


def test_compute_confidence_margin_range():
    torch.manual_seed(0)
    logits = torch.randn(8, 256)
    conf = compute_confidence(logits, "margin")
    assert conf.shape == (8,)
    # margin = top1_prob - top2_prob, which is in [0, 1]
    assert (conf >= 0.0).all() and (conf <= 1.0).all()


def test_compute_confidence_output_shape():
    """All metrics must return shape (B,)."""
    B, V = 4, 128
    logits = torch.randn(B, V)
    for metric in ("max_prob", "entropy", "margin"):
        out = compute_confidence(logits, metric)
        assert out.shape == (B,), f"Failed for metric={metric}"


# ---------------------------------------------------------------------------
# 6. ExitClassifier output shape
# ---------------------------------------------------------------------------


def test_exit_classifier_output_shape():
    B, d_model, vocab_size = 4, 64, 256
    classifier = ExitClassifier(d_model, vocab_size)
    hidden = torch.randn(B, d_model)
    out = classifier(hidden)
    assert out.shape == (B, vocab_size)


# ---------------------------------------------------------------------------
# 7. EarlyExitWrapper forward output shape
# ---------------------------------------------------------------------------


def test_early_exit_wrapper_forward_output_shape():
    model = _make_model()
    wrapper = _make_wrapper(model)
    input_ids = torch.randint(0, 256, (2, 8))
    out = wrapper(input_ids)
    # forward returns (B, vocab_size) — last token logits
    assert out.shape == (2, 256)


# ---------------------------------------------------------------------------
# 8. EarlyExitWrapper forward_with_exits returns (logits, int)
# ---------------------------------------------------------------------------


def test_early_exit_wrapper_forward_with_exits_return_type():
    model = _make_model()
    wrapper = _make_wrapper(model)
    input_ids = torch.randint(0, 256, (1, 8))
    result = wrapper.forward_with_exits(input_ids)
    assert isinstance(result, tuple) and len(result) == 2
    logits, exit_layer = result
    assert isinstance(logits, torch.Tensor)
    assert isinstance(exit_layer, int)


# ---------------------------------------------------------------------------
# 9. EarlyExitWrapper forward_with_exits exit_layer in valid range
# ---------------------------------------------------------------------------


def test_early_exit_wrapper_exit_layer_valid_range():
    model = _make_model()
    wrapper = _make_wrapper(model)
    n_layers = len(model.layers)
    input_ids = torch.randint(0, 256, (1, 8))
    _logits, exit_layer = wrapper.forward_with_exits(input_ids)
    assert 0 <= exit_layer < n_layers


# ---------------------------------------------------------------------------
# 10. EarlyExitWrapper compute_exit_stats returns correct keys
# ---------------------------------------------------------------------------


def test_compute_exit_stats_keys():
    model = _make_model()
    wrapper = _make_wrapper(model)
    batch = [torch.randint(0, 256, (1, 8)) for _ in range(4)]
    stats = wrapper.compute_exit_stats(batch)
    assert "mean_exit_layer" in stats
    assert "early_exit_rate" in stats
    assert "speedup_estimate" in stats


# ---------------------------------------------------------------------------
# 11. threshold=1.0 never exits early (always runs all layers)
# ---------------------------------------------------------------------------


def test_no_early_exit_when_threshold_is_one():
    """With threshold=1.0, confidence can never meet it, so always run all layers."""
    model = _make_model()
    n_layers = len(model.layers)
    wrapper = _make_wrapper(model, exit_thresholds=[1.0, 1.0])

    for _ in range(5):
        input_ids = torch.randint(0, 256, (1, 8))
        _logits, exit_layer = wrapper.forward_with_exits(input_ids)
        assert exit_layer == n_layers - 1, (
            f"Expected exit at last layer ({n_layers - 1}), got {exit_layer}"
        )


# ---------------------------------------------------------------------------
# 12. train_exit_classifiers returns mean_kl_loss key
# ---------------------------------------------------------------------------


def test_train_exit_classifiers_returns_kl_loss():
    model = _make_model()
    wrapper = _make_wrapper(model)
    data = torch.randint(0, 256, (4, 8))
    result = train_exit_classifiers(model, wrapper, data, n_steps=3)
    assert "mean_kl_loss" in result
    assert isinstance(result["mean_kl_loss"], float)


# ---------------------------------------------------------------------------
# 13. Exit classifiers have trainable parameters (requires_grad=True)
# ---------------------------------------------------------------------------


def test_exit_classifiers_have_trainable_params():
    model = _make_model()
    wrapper = _make_wrapper(model)
    for i, clf in enumerate(wrapper.exit_classifiers):
        for name, param in clf.named_parameters():
            assert param.requires_grad, f"Exit classifier {i} param '{name}' does not require grad"
