"""Tests for adaptive_early_exit.py — 16 test functions."""

from __future__ import annotations

import math

import torch

from src.inference.adaptive_early_exit import (
    AdaptiveEarlyExitModel,
    EarlyExitConfig,
    EarlyExitLayer,
    EarlyExitProfiler,
    EarlyExitTrainer,
    ExitClassifier,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_SIZE = 16
N_LAYERS = 4
N_HEADS = 4
B = 2
T = 6


def _make_model(threshold: float = 0.8) -> AdaptiveEarlyExitModel:
    return AdaptiveEarlyExitModel(
        d_model=D_MODEL,
        vocab_size=VOCAB_SIZE,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        exit_threshold=threshold,
    )


def _make_input() -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (B, T))


# ---------------------------------------------------------------------------
# ExitClassifier tests
# ---------------------------------------------------------------------------


def test_exit_classifier_output_shape():
    """forward() returns [B, T] tensor."""
    clf = ExitClassifier(D_MODEL)
    x = torch.randn(B, T, D_MODEL)
    out = clf(x)
    assert out.shape == (B, T), f"Expected ({B}, {T}), got {out.shape}"


def test_exit_classifier_probs_in_unit_interval():
    """Exit probabilities are in [0, 1]."""
    clf = ExitClassifier(D_MODEL)
    x = torch.randn(B, T, D_MODEL)
    out = clf(x)
    assert (out >= 0.0).all(), "Some exit_probs are negative"
    assert (out <= 1.0).all(), "Some exit_probs exceed 1"


def test_exit_classifier_should_exit_returns_bool():
    """should_exit() returns a bool tensor."""
    clf = ExitClassifier(D_MODEL)
    x = torch.randn(B, T, D_MODEL)
    mask = clf.should_exit(x, threshold=0.5)
    assert mask.dtype == torch.bool, f"Expected bool, got {mask.dtype}"
    assert mask.shape == (B, T)


# ---------------------------------------------------------------------------
# EarlyExitLayer tests
# ---------------------------------------------------------------------------


def test_early_exit_layer_output_shapes():
    """EarlyExitLayer forward returns tensors of correct shapes."""
    layer = EarlyExitLayer(D_MODEL, N_HEADS, layer_idx=0, vocab_size=VOCAB_SIZE)
    x = torch.randn(B, T, D_MODEL)
    x_out, early_logits, exit_mask = layer(x, exit_threshold=0.5)

    assert x_out.shape == (B, T, D_MODEL), f"x_out shape wrong: {x_out.shape}"
    assert early_logits.shape == (B, T, VOCAB_SIZE), (
        f"early_logits shape wrong: {early_logits.shape}"
    )
    assert exit_mask.shape == (B, T), f"exit_mask shape wrong: {exit_mask.shape}"


def test_early_exit_layer_exit_mask_is_bool():
    """exit_mask returned by EarlyExitLayer is a boolean tensor."""
    layer = EarlyExitLayer(D_MODEL, N_HEADS, layer_idx=0, vocab_size=VOCAB_SIZE)
    x = torch.randn(B, T, D_MODEL)
    _, _, exit_mask = layer(x, exit_threshold=0.5)
    assert exit_mask.dtype == torch.bool, f"exit_mask dtype: {exit_mask.dtype}"


def test_early_exit_layer_threshold_zero_all_exit():
    """With threshold=0, every token should exit (exit_prob > 0 always true)."""
    torch.manual_seed(42)
    layer = EarlyExitLayer(D_MODEL, N_HEADS, layer_idx=0, vocab_size=VOCAB_SIZE)
    x = torch.randn(B, T, D_MODEL)
    _, _, exit_mask = layer(x, exit_threshold=0.0)
    assert exit_mask.all(), "Expected all tokens to exit when threshold=0"


def test_early_exit_layer_threshold_one_none_exit():
    """With threshold=1.0, no token should exit (exit_prob <= 1 always)."""
    layer = EarlyExitLayer(D_MODEL, N_HEADS, layer_idx=0, vocab_size=VOCAB_SIZE)
    x = torch.randn(B, T, D_MODEL)
    _, _, exit_mask = layer(x, exit_threshold=1.0)
    assert not exit_mask.any(), "Expected no tokens to exit when threshold=1"


# ---------------------------------------------------------------------------
# AdaptiveEarlyExitModel tests
# ---------------------------------------------------------------------------


def test_model_forward_logits_shape():
    """Model forward returns logits of shape [B, T, vocab]."""
    model = _make_model()
    ids = _make_input()
    logits, _ = model(ids)
    assert logits.shape == (B, T, VOCAB_SIZE), f"logits shape wrong: {logits.shape}"


def test_model_forward_layer_assignments_range():
    """layer_assignments are in [0, n_layers]."""
    model = _make_model()
    ids = _make_input()
    _, assignments = model(ids)
    assert assignments.shape == (B, T)
    assert (assignments >= 0).all(), "Negative layer assignment found"
    assert (assignments <= N_LAYERS).all(), (
        f"Layer assignment exceeds n_layers={N_LAYERS}: {assignments.max()}"
    )


def test_model_compute_loss_is_finite():
    """compute_loss returns a finite scalar."""
    model = _make_model()
    ids = _make_input()
    loss = model.compute_loss(ids)
    assert loss.ndim == 0, "Loss should be a scalar"
    assert math.isfinite(loss.item()), f"Loss is not finite: {loss.item()}"


def test_model_compute_loss_backward():
    """Gradients flow through compute_loss."""
    model = _make_model()
    ids = _make_input()
    loss = model.compute_loss(ids)
    loss.backward()

    grad_found = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            grad_found = True
            break
    assert grad_found, "No non-zero gradients found after backward()"


def test_model_set_threshold_changes_behavior():
    """Changing threshold via set_threshold alters layer_assignments distribution."""
    torch.manual_seed(0)
    model = _make_model(threshold=0.01)  # very low → most tokens exit early
    ids = _make_input()
    _, assignments_low = model(ids)

    model.set_threshold(0.99)  # very high → most tokens go through all layers
    _, assignments_high = model(ids)

    mean_low = assignments_low.float().mean().item()
    mean_high = assignments_high.float().mean().item()
    assert mean_low <= mean_high, (
        f"Low threshold should yield smaller mean layer. Got {mean_low} vs {mean_high}"
    )


# ---------------------------------------------------------------------------
# EarlyExitTrainer tests
# ---------------------------------------------------------------------------


def test_trainer_train_step_returns_loss_and_float():
    """train_step returns (Tensor, float)."""
    model = _make_model()
    trainer = EarlyExitTrainer(model, lr=1e-3, lambda_exit=0.1)
    ids = _make_input()
    loss, mean_exit = trainer.train_step(ids)

    assert isinstance(loss, torch.Tensor), "loss should be a Tensor"
    assert loss.ndim == 0, "loss should be scalar"
    assert isinstance(mean_exit, float), f"mean_exit should be float, got {type(mean_exit)}"


def test_trainer_mean_exit_layer_non_negative():
    """mean_exit_layer returned by train_step is >= 0."""
    model = _make_model()
    trainer = EarlyExitTrainer(model, lr=1e-3, lambda_exit=0.1)
    ids = _make_input()
    _, mean_exit = trainer.train_step(ids)
    assert mean_exit >= 0.0, f"mean_exit_layer is negative: {mean_exit}"


# ---------------------------------------------------------------------------
# EarlyExitProfiler tests
# ---------------------------------------------------------------------------


def test_profiler_flop_savings_in_unit_interval():
    """compute_flop_savings returns a value in [0, 1]."""
    profiler = EarlyExitProfiler()
    # Simulate assignments: half exit at layer 1, half at final layer
    assignments = torch.tensor([[1, N_LAYERS], [2, N_LAYERS]])
    profiler.record_batch(assignments)
    savings = profiler.compute_flop_savings(n_layers=N_LAYERS)
    assert 0.0 <= savings <= 1.0, f"FLOPs savings out of [0,1]: {savings}"


def test_profiler_layer_distribution_sums_to_one():
    """layer_distribution fractions sum to 1."""
    profiler = EarlyExitProfiler()
    assignments = torch.randint(0, N_LAYERS + 1, (B, T))
    profiler.record_batch(assignments)
    dist = profiler.layer_distribution()
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-5, f"Distribution sums to {total}, not 1.0"


# ---------------------------------------------------------------------------
# EarlyExitConfig tests
# ---------------------------------------------------------------------------


def test_early_exit_config_defaults():
    """EarlyExitConfig has the expected default values."""
    cfg = EarlyExitConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.n_layers == 4
    assert cfg.n_heads == 4
    assert abs(cfg.exit_threshold - 0.8) < 1e-9
    assert abs(cfg.lambda_exit - 0.1) < 1e-9
    assert abs(cfg.target_layer_frac - 0.5) < 1e-9
