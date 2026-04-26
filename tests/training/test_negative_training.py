"""Tests for negative_training module."""

from __future__ import annotations

import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.negative_training import (
    NegativeTrainer,
    NegativeTrainingConfig,
    NegativeTrainingResult,
    evaluate_suppression,
    negative_loss,
    positive_loss,
    project_gradient,
)

# ---------------------------------------------------------------------------
# Tiny model config for fast tests
# ---------------------------------------------------------------------------
CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

BATCH = 2
SEQ = 8


def make_model() -> AureliusTransformer:
    torch.manual_seed(42)
    return AureliusTransformer(CFG)


def make_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (inputs, targets) both shape (BATCH, SEQ) with valid token ids."""
    inputs = torch.randint(0, CFG.vocab_size, (BATCH, SEQ))
    targets = torch.randint(0, CFG.vocab_size, (BATCH, SEQ))
    return inputs, targets


# ---------------------------------------------------------------------------
# 1. negative_loss returns negative scalar
# ---------------------------------------------------------------------------
def test_negative_loss_is_negative():
    torch.manual_seed(0)
    model = make_model()
    inputs, targets = make_batch()
    loss = negative_loss(model, inputs, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, "Expected scalar tensor"
    assert loss.item() < 0, f"negative_loss should be < 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 2. positive_loss returns positive scalar
# ---------------------------------------------------------------------------
def test_positive_loss_is_positive():
    torch.manual_seed(1)
    model = make_model()
    inputs, targets = make_batch()
    loss = positive_loss(model, inputs, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, "Expected scalar tensor"
    assert loss.item() > 0, f"positive_loss should be > 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# 3. project_gradient returns list of same length and shape
# ---------------------------------------------------------------------------
def test_project_gradient_shape():
    torch.manual_seed(2)
    shapes = [(32, 16), (16,), (8, 8)]
    neg_grad = [torch.randn(*s) for s in shapes]
    pos_grad = [torch.randn(*s) for s in shapes]
    result = project_gradient(neg_grad, pos_grad)
    assert len(result) == len(neg_grad)
    for orig, proj in zip(neg_grad, result):
        assert proj.shape == orig.shape


# ---------------------------------------------------------------------------
# 4. Projected gradient has reduced dot product with pos_grad
# ---------------------------------------------------------------------------
def test_project_gradient_reduces_dot_product():
    torch.manual_seed(3)
    # Use a single large vector so the projection is measurable
    g_neg = [torch.randn(512)]
    g_pos = [torch.randn(512)]

    # Original dot product
    orig_dot = (g_neg[0] * g_pos[0]).sum().abs().item()

    projected = project_gradient(g_neg, g_pos)
    proj_dot = (projected[0] * g_pos[0]).sum().abs().item()

    assert proj_dot < orig_dot + 1e-4, (
        f"Projected dot ({proj_dot:.6f}) should be <= original dot ({orig_dot:.6f})"
    )
    # After projection, dot product should be near zero (orthogonal)
    assert proj_dot < 1e-3, (
        f"Projected gradient should be nearly orthogonal to pos_grad, got dot={proj_dot:.6f}"
    )


# ---------------------------------------------------------------------------
# 5. NegativeTrainer constructs without error
# ---------------------------------------------------------------------------
def test_negative_trainer_constructs():
    model = make_model()
    config = NegativeTrainingConfig()
    trainer = NegativeTrainer(model, config)
    assert trainer.model is model
    assert trainer.config is config


# ---------------------------------------------------------------------------
# 6. negative_step returns float
# ---------------------------------------------------------------------------
def test_negative_step_returns_float():
    torch.manual_seed(4)
    model = make_model()
    config = NegativeTrainingConfig(negative_steps=1, positive_steps=1)
    trainer = NegativeTrainer(model, config)
    inputs, targets = make_batch()
    result = trainer.negative_step(inputs, targets)
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert torch.isfinite(torch.tensor(result))


# ---------------------------------------------------------------------------
# 7. positive_step returns float
# ---------------------------------------------------------------------------
def test_positive_step_returns_float():
    torch.manual_seed(5)
    model = make_model()
    config = NegativeTrainingConfig(negative_steps=1, positive_steps=1)
    trainer = NegativeTrainer(model, config)
    inputs, targets = make_batch()
    result = trainer.positive_step(inputs, targets)
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert torch.isfinite(torch.tensor(result))


# ---------------------------------------------------------------------------
# 8. After negative_step, model weights change
# ---------------------------------------------------------------------------
def test_negative_step_changes_weights():
    torch.manual_seed(6)
    model = make_model()
    config = NegativeTrainingConfig(
        negative_lr=1e-3,
        loss_margin=100.0,  # ensure we don't skip
        gradient_projection=False,
    )
    trainer = NegativeTrainer(model, config)
    inputs, targets = make_batch()

    # Capture weight snapshot before
    param = next(model.parameters())
    before = param.data.clone()

    trainer.negative_step(inputs, targets)

    after = param.data.clone()
    assert not torch.allclose(before, after), "Weights should change after negative_step"


# ---------------------------------------------------------------------------
# 9. run returns NegativeTrainingResult
# ---------------------------------------------------------------------------
def test_run_returns_result():
    torch.manual_seed(7)
    model = make_model()
    config = NegativeTrainingConfig(negative_steps=2, positive_steps=2, gradient_projection=False)
    trainer = NegativeTrainer(model, config)

    neg_data = [make_batch()]
    pos_data = [make_batch()]

    result = trainer.run(neg_data, pos_data)
    assert isinstance(result, NegativeTrainingResult)


# ---------------------------------------------------------------------------
# 10. run negative_losses has correct length
# ---------------------------------------------------------------------------
def test_run_negative_losses_length():
    torch.manual_seed(8)
    model = make_model()
    n_neg = 3
    n_pos = 4
    config = NegativeTrainingConfig(
        negative_steps=n_neg,
        positive_steps=n_pos,
        gradient_projection=False,
    )
    trainer = NegativeTrainer(model, config)

    neg_data = [make_batch(), make_batch()]
    pos_data = [make_batch(), make_batch()]

    result = trainer.run(neg_data, pos_data)
    assert len(result.negative_losses) == n_neg, (
        f"Expected {n_neg} negative losses, got {len(result.negative_losses)}"
    )
    assert len(result.positive_losses) == n_pos, (
        f"Expected {n_pos} positive losses, got {len(result.positive_losses)}"
    )
    assert result.n_negative_steps == n_neg
    assert result.n_positive_steps == n_pos


# ---------------------------------------------------------------------------
# 11. evaluate_suppression returns dict with required keys
# ---------------------------------------------------------------------------
def test_evaluate_suppression_keys():
    torch.manual_seed(9)
    model = make_model()
    inputs, targets = make_batch()
    result = evaluate_suppression(model, inputs, targets)
    assert isinstance(result, dict)
    assert "neg_loss" in result, "Missing 'neg_loss' key"
    assert "neg_perplexity" in result, "Missing 'neg_perplexity' key"
    assert isinstance(result["neg_loss"], float)
    assert isinstance(result["neg_perplexity"], float)
    assert result["neg_loss"] > 0
    assert result["neg_perplexity"] >= 1.0


# ---------------------------------------------------------------------------
# 12. After negative training, neg_loss higher than before (suppression happened)
# ---------------------------------------------------------------------------
def test_negative_training_increases_neg_loss():
    torch.manual_seed(10)
    model = make_model()
    inputs, targets = make_batch()

    # Measure baseline
    baseline = evaluate_suppression(model, inputs, targets)
    baseline_loss = baseline["neg_loss"]

    # Run negative training with high LR to ensure measurable change
    config = NegativeTrainingConfig(
        negative_lr=5e-3,
        positive_lr=1e-6,
        negative_steps=5,
        positive_steps=0,
        gradient_projection=False,
        loss_margin=50.0,
    )
    trainer = NegativeTrainer(model, config)
    neg_data = [(inputs, targets)]
    pos_data = [(inputs, targets)]  # not used (positive_steps=0)
    trainer.run(neg_data, pos_data)

    # Measure after
    after = evaluate_suppression(model, inputs, targets)
    after_loss = after["neg_loss"]

    assert after_loss > baseline_loss, (
        f"Expected neg_loss to increase after negative training "
        f"(baseline={baseline_loss:.4f}, after={after_loss:.4f})"
    )
