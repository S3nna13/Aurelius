"""Tests for Contrastive Preference Optimization (CPO).

Tiny setup: vocab_size=256, seq_len=16, batch=2.
MockLM is self-contained — no imports from the main transformer module.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.alignment.cpo import CPOConfig, CPOTrainer, cpo_loss

# ---------------------------------------------------------------------------
# Tiny mock language model
# ---------------------------------------------------------------------------


class MockLM(nn.Module):
    """Minimal language model for testing: input_ids → logits (B, T, vocab_size)."""

    def __init__(self, vocab_size: int = 256, d_model: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # (B, T) → (B, T, V)
        x = self.embed(input_ids)  # (B, T, d_model)
        return self.proj(x)  # (B, T, vocab_size)


# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
SEQ_LEN = 16
BATCH = 2


@pytest.fixture
def model():
    torch.manual_seed(0)
    return MockLM(vocab_size=VOCAB_SIZE)


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)


@pytest.fixture
def default_config():
    return CPOConfig()


def _make_logps(batch: int = BATCH, value: float = -2.0) -> torch.Tensor:
    """Return a fixed tensor of mean log-probs for testing."""
    return torch.full((batch,), value)


def _random_logps(batch: int = BATCH, low: float = -5.0, high: float = -1.0) -> torch.Tensor:
    return torch.empty(batch).uniform_(low, high)


def _make_batch(batch: int = BATCH, seq_len: int = SEQ_LEN) -> dict:
    """Build a minimal CPOTrainer-compatible batch."""
    torch.manual_seed(42)
    chosen_ids = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
    rejected_ids = torch.randint(0, VOCAB_SIZE, (batch, seq_len))
    mask = torch.ones(batch, seq_len, dtype=torch.long)
    return {
        "chosen_input_ids": chosen_ids,
        "chosen_labels": chosen_ids.clone(),
        "rejected_input_ids": rejected_ids,
        "rejected_labels": rejected_ids.clone(),
        "chosen_attention_mask": mask,
        "rejected_attention_mask": mask.clone(),
    }


# ---------------------------------------------------------------------------
# Test 1: cpo_loss returns a scalar tensor
# ---------------------------------------------------------------------------


def test_cpo_loss_returns_scalar(default_config):
    """cpo_loss must return a 0-dim (scalar) tensor."""
    chosen = _make_logps(value=-1.5)
    rejected = _make_logps(value=-3.0)
    loss, _ = cpo_loss(chosen, rejected, default_config)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), "Loss must be finite"


# ---------------------------------------------------------------------------
# Test 2: Metrics dict has required keys
# ---------------------------------------------------------------------------


def test_cpo_loss_metrics_keys(default_config):
    """Metrics dict must contain 'sft_loss', 'cpo_loss', and 'margin'."""
    chosen = _make_logps(value=-1.5)
    rejected = _make_logps(value=-3.0)
    _, metrics = cpo_loss(chosen, rejected, default_config)
    assert "sft_loss" in metrics, "Missing key 'sft_loss'"
    assert "cpo_loss" in metrics, "Missing key 'cpo_loss'"
    assert "margin" in metrics, "Missing key 'margin'"


# ---------------------------------------------------------------------------
# Test 3: Loss is differentiable (backward passes)
# ---------------------------------------------------------------------------


def test_cpo_loss_backward(default_config):
    """total_loss.backward() must not raise and produce finite gradients."""
    chosen = _random_logps().requires_grad_(True)
    rejected = _random_logps().detach()
    loss, _ = cpo_loss(chosen, rejected, default_config)
    loss.backward()
    assert chosen.grad is not None, "Gradient must flow to chosen_logps"
    assert torch.isfinite(chosen.grad).all(), "Gradients must be finite"


# ---------------------------------------------------------------------------
# Test 4: When chosen >> rejected, contrastive loss is near zero
# ---------------------------------------------------------------------------


def test_cpo_loss_near_zero_contrastive_when_chosen_dominates():
    """When chosen log-probs >> rejected, the contrastive term should be near 0."""
    config = CPOConfig(beta=1.0, delta=0.0)
    # Large gap → sigmoid argument → +∞ → -log sigmoid → 0
    chosen = torch.full((BATCH,), -0.1)
    rejected = torch.full((BATCH,), -100.0)
    _, metrics = cpo_loss(chosen, rejected, config)
    assert metrics["cpo_loss"] < 0.01, (
        f"Expected near-zero contrastive loss, got {metrics['cpo_loss']}"
    )


# ---------------------------------------------------------------------------
# Test 5: When chosen == rejected, margin is 0
# ---------------------------------------------------------------------------


def test_cpo_loss_zero_margin_when_equal(default_config):
    """When chosen_logps == rejected_logps, the margin must be 0."""
    logps = _make_logps(value=-2.5)
    _, metrics = cpo_loss(logps, logps.clone(), default_config)
    assert abs(metrics["margin"]) < 1e-6, f"Expected margin=0, got {metrics['margin']}"


# ---------------------------------------------------------------------------
# Test 6: Higher beta → sharper (larger-magnitude) preference signal
# ---------------------------------------------------------------------------


def test_higher_beta_sharper_preference():
    """Higher beta should increase the contrastive loss magnitude when margin is negative."""
    # Chosen < rejected → margin < 0 → loss is large and grows with beta
    chosen = torch.full((BATCH,), -3.0)
    rejected = torch.full((BATCH,), -1.0)

    cfg_low = CPOConfig(beta=0.1, delta=0.0)
    cfg_high = CPOConfig(beta=2.0, delta=0.0)

    _, m_low = cpo_loss(chosen, rejected, cfg_low)
    _, m_high = cpo_loss(chosen, rejected, cfg_high)

    assert m_high["cpo_loss"] > m_low["cpo_loss"], (
        "Higher beta should produce larger contrastive loss when chosen < rejected"
    )


# ---------------------------------------------------------------------------
# Test 7: CPOTrainer.train_step returns dict with 'loss' key
# ---------------------------------------------------------------------------


def test_trainer_train_step_has_loss_key(model, optimizer):
    """train_step must return a dict that includes a 'loss' key."""
    config = CPOConfig()
    trainer = CPOTrainer(model, config, optimizer)
    batch = _make_batch()
    result = trainer.train_step(batch)
    assert "loss" in result, "train_step result must contain 'loss'"
    assert math.isfinite(result["loss"]), f"loss must be finite, got {result['loss']}"


# ---------------------------------------------------------------------------
# Test 8: label_smoothing > 0 produces different loss than label_smoothing = 0
# ---------------------------------------------------------------------------


def test_label_smoothing_changes_loss():
    """label_smoothing > 0 must yield a different cpo_loss value than label_smoothing = 0."""
    chosen = torch.full((BATCH,), -1.0)
    rejected = torch.full((BATCH,), -3.0)

    cfg_no_ls = CPOConfig(beta=0.5, delta=0.0, label_smoothing=0.0)
    cfg_ls = CPOConfig(beta=0.5, delta=0.0, label_smoothing=0.2)

    _, m_no_ls = cpo_loss(chosen, rejected, cfg_no_ls)
    _, m_ls = cpo_loss(chosen, rejected, cfg_ls)

    assert abs(m_ls["cpo_loss"] - m_no_ls["cpo_loss"]) > 1e-6, (
        "label_smoothing > 0 should produce a different cpo_loss"
    )


# ---------------------------------------------------------------------------
# Test 9: Gradient flows to model parameters
# ---------------------------------------------------------------------------


def test_gradient_flows_to_model_parameters(model, optimizer):
    """After train_step, at least one model parameter must have a non-zero gradient."""
    config = CPOConfig()
    trainer = CPOTrainer(model, config, optimizer)
    batch = _make_batch()

    # Zero out gradients before the step to start clean
    optimizer.zero_grad()
    # train_step does zero_grad+backward+step internally; we call it directly
    trainer.train_step(batch)

    # After train_step the optimizer has already stepped and zeroed — run a manual
    # backward to verify gradient connectivity exists
    chosen_ids = batch["chosen_input_ids"]
    chosen_labels = batch["chosen_labels"]
    rejected_ids = batch["rejected_input_ids"]
    rejected_labels = batch["rejected_labels"]
    mask = batch["chosen_attention_mask"]

    optimizer.zero_grad()
    chosen_logps = trainer.compute_logps(model, chosen_ids, mask, chosen_labels)
    rejected_logps = trainer.compute_logps(model, rejected_ids, mask, rejected_labels)
    loss, _ = cpo_loss(chosen_logps, rejected_logps, config)
    loss.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No non-zero gradients found in model parameters"


# ---------------------------------------------------------------------------
# Test 10: delta > 0 increases loss vs delta = 0
# ---------------------------------------------------------------------------


def test_positive_delta_increases_loss():
    """A positive delta makes the margin harder to satisfy, increasing the CPO loss."""
    chosen = torch.full((BATCH,), -1.5)
    rejected = torch.full((BATCH,), -2.5)  # margin = 1.0

    cfg_no_delta = CPOConfig(beta=1.0, delta=0.0)
    cfg_delta = CPOConfig(beta=1.0, delta=0.5)  # pushes required margin higher

    _, m_no_delta = cpo_loss(chosen, rejected, cfg_no_delta)
    _, m_delta = cpo_loss(chosen, rejected, cfg_delta)

    assert m_delta["cpo_loss"] > m_no_delta["cpo_loss"], (
        f"delta>0 should increase cpo_loss: {m_delta['cpo_loss']} vs {m_no_delta['cpo_loss']}"
    )


# ---------------------------------------------------------------------------
# Bonus test 11: train_step returns all expected metric keys
# ---------------------------------------------------------------------------


def test_trainer_train_step_full_metrics(model, optimizer):
    """train_step must return 'loss', 'sft_loss', 'cpo_loss', and 'margin'."""
    config = CPOConfig()
    trainer = CPOTrainer(model, config, optimizer)
    batch = _make_batch()
    result = trainer.train_step(batch)
    for key in ("loss", "sft_loss", "cpo_loss", "margin"):
        assert key in result, f"Missing key '{key}' in train_step result"
        assert math.isfinite(result[key]), f"'{key}' must be finite, got {result[key]}"


# ---------------------------------------------------------------------------
# Bonus test 12: sft_weight and cpo_weight scale their respective components
# ---------------------------------------------------------------------------


def test_sft_weight_scales_loss():
    """Doubling sft_weight should change total_loss by approximately that factor on the SFT term."""
    chosen = torch.full((BATCH,), -1.5)
    rejected = torch.full((BATCH,), -2.5)

    cfg1 = CPOConfig(beta=0.5, delta=0.0, sft_weight=1.0, cpo_weight=0.0)
    cfg2 = CPOConfig(beta=0.5, delta=0.0, sft_weight=2.0, cpo_weight=0.0)

    loss1, _ = cpo_loss(chosen, rejected, cfg1)
    loss2, _ = cpo_loss(chosen, rejected, cfg2)

    assert abs(loss2.item() / loss1.item() - 2.0) < 1e-5, (
        "Doubling sft_weight should double the total loss when cpo_weight=0"
    )


# ---------------------------------------------------------------------------
# Bonus test 13: compute_logps returns correct shape
# ---------------------------------------------------------------------------


def test_compute_logps_shape(model, optimizer):
    """compute_logps must return a 1-D tensor of length batch_size."""
    config = CPOConfig()
    trainer = CPOTrainer(model, config, optimizer)
    ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    logps = trainer.compute_logps(model, ids, mask, ids.clone())
    assert logps.ndim == 1
    assert logps.shape[0] == BATCH


# ---------------------------------------------------------------------------
# Bonus test 14: compute_logps values are negative (valid log-probs)
# ---------------------------------------------------------------------------


def test_compute_logps_are_negative(model, optimizer):
    """Mean log-probs per sequence should be negative (probabilities < 1)."""
    config = CPOConfig()
    trainer = CPOTrainer(model, config, optimizer)
    ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    mask = torch.ones(BATCH, SEQ_LEN, dtype=torch.long)
    logps = trainer.compute_logps(model, ids, mask, ids.clone())
    assert (logps < 0).all(), f"Log-probs should be negative, got {logps}"
