"""Tests for ORPO v2 — Odds Ratio Preference Optimization (Hong et al. 2024).

Covers ORPOConfig, ORPOLoss, and ORPOTrainer from aurelius.alignment.orpo_v2.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from aurelius.alignment.orpo_v2 import ORPOConfig, ORPOLoss, ORPOTrainer

# ---------------------------------------------------------------------------
# Shared tiny model used for gradient / trainer tests
# ---------------------------------------------------------------------------


class _TinyLM(nn.Module):
    """Minimal embedding + linear model sufficient for gradient flow tests."""

    def __init__(self, V: int = 16, d: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.proj = nn.Linear(d, V)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T) -> (B, T, V)
        return self.proj(self.embed(x).float())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

V, T, B = 16, 8, 4  # vocab, seq_len, batch


def _make_logits(batch: int = B, seq: int = T, vocab: int = V) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, seq, vocab)


def _make_labels(batch: int = B, seq: int = T, vocab: int = V) -> torch.LongTensor:
    torch.manual_seed(1)
    return torch.randint(0, vocab, (batch, seq))


def _default_loss_fn() -> ORPOLoss:
    return ORPOLoss(ORPOConfig())


# ---------------------------------------------------------------------------
# 1. ORPOConfig defaults
# ---------------------------------------------------------------------------


def test_orpo_config_defaults():
    """ORPOConfig must expose lambda_=0.1 and beta=0.1 by default."""
    cfg = ORPOConfig()
    assert cfg.lambda_ == pytest.approx(0.1)
    assert cfg.beta == pytest.approx(0.1)


def test_orpo_config_custom():
    """ORPOConfig fields must be settable."""
    cfg = ORPOConfig(lambda_=0.5, beta=0.2)
    assert cfg.lambda_ == pytest.approx(0.5)
    assert cfg.beta == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 2. log_odds
# ---------------------------------------------------------------------------


def test_log_odds_finite():
    """log_odds must return finite values for typical negative log-probs."""
    loss_fn = _default_loss_fn()
    # log(-0.7) is undefined; the test description says "log_odds(log(-0.7))
    # makes sense" — interpret as: log_odds of a value around log(0.7) ≈ -0.357.
    log_p = torch.tensor(math.log(0.7))  # ≈ -0.357
    result = loss_fn.log_odds(log_p.unsqueeze(0))
    assert torch.isfinite(result).all(), f"log_odds not finite: {result}"


def test_log_odds_output_shape():
    """log_odds must return a (B,) tensor given a (B,) input."""
    loss_fn = _default_loss_fn()
    log_p = torch.linspace(-5.0, -0.1, B)
    result = loss_fn.log_odds(log_p)
    assert result.shape == (B,), f"Expected ({B},), got {result.shape}"


def test_log_odds_monotone():
    """log_odds must be monotonically increasing: higher p → higher log-odds."""
    loss_fn = _default_loss_fn()
    log_p = torch.tensor([-3.0, -2.0, -1.0, -0.5])
    lo = loss_fn.log_odds(log_p)
    assert (lo[1:] > lo[:-1]).all(), "log_odds should be monotonically increasing"


# ---------------------------------------------------------------------------
# 3. sft_loss
# ---------------------------------------------------------------------------


def test_sft_loss_scalar_and_finite():
    """sft_loss must return a scalar finite tensor."""
    loss_fn = _default_loss_fn()
    logits = _make_logits()
    labels = _make_labels()
    loss = loss_fn.sft_loss(logits, labels)
    assert loss.ndim == 0, "sft_loss must be scalar"
    assert torch.isfinite(loss), f"sft_loss not finite: {loss.item()}"


def test_sft_loss_all_ignored():
    """sft_loss must not return NaN when all labels are -100."""
    loss_fn = _default_loss_fn()
    logits = _make_logits()
    labels = torch.full((B, T), -100, dtype=torch.long)
    loss = loss_fn.sft_loss(logits, labels)
    # Should return 0.0 (our convention) or at least not NaN/inf.
    assert not torch.isnan(loss), "sft_loss returned NaN for all-(-100) labels"
    assert torch.isfinite(loss) or loss.item() == 0.0


# ---------------------------------------------------------------------------
# 4. odds_ratio_loss
# ---------------------------------------------------------------------------


def test_odds_ratio_loss_scalar_and_finite():
    """odds_ratio_loss must return a scalar finite tensor."""
    loss_fn = _default_loss_fn()
    log_p_w = torch.tensor([-0.3, -0.4, -0.2, -0.5])
    log_p_l = torch.tensor([-1.0, -0.9, -0.8, -1.2])
    loss = loss_fn.odds_ratio_loss(log_p_w, log_p_l)
    assert loss.ndim == 0, "odds_ratio_loss must be scalar"
    assert torch.isfinite(loss), f"odds_ratio_loss not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# 5. ORPOLoss.forward
# ---------------------------------------------------------------------------


def test_forward_returns_correct_keys():
    """forward must return a metrics dict with the four required keys."""
    loss_fn = _default_loss_fn()
    chosen_logits = _make_logits()
    rejected_logits = _make_logits()
    chosen_labels = _make_labels()
    rejected_labels = _make_labels()

    _, metrics = loss_fn(chosen_logits, rejected_logits, chosen_labels, rejected_labels)

    required = {"sft_loss", "odds_ratio_loss", "total_loss", "accuracy"}
    assert required == set(metrics.keys()), (
        f"Metric keys mismatch. Expected {required}, got {set(metrics.keys())}"
    )


def test_forward_total_loss_finite():
    """total_loss from forward must be finite."""
    loss_fn = _default_loss_fn()
    total, metrics = loss_fn(_make_logits(), _make_logits(), _make_labels(), _make_labels())
    assert torch.isfinite(total), f"total_loss not finite: {total.item()}"
    assert torch.isfinite(metrics["total_loss"])


def test_forward_gradient_flows():
    """Gradient must flow back through chosen_logits to model parameters."""
    torch.manual_seed(42)
    model = _TinyLM(V=V, d=32)
    loss_fn = _default_loss_fn()

    tokens = torch.randint(0, V, (B, T))
    chosen_logits = model(tokens)  # requires_grad via model params
    rejected_logits = model(tokens).detach()  # no grad needed on rejected

    labels = _make_labels()
    total, _ = loss_fn(chosen_logits, rejected_logits, labels, labels)
    total.backward()

    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert has_grad, "No gradient flowed to model parameters"


# ---------------------------------------------------------------------------
# 6. Accuracy
# ---------------------------------------------------------------------------


def test_accuracy_in_range():
    """Accuracy must be in [0, 1]."""
    loss_fn = _default_loss_fn()
    _, metrics = loss_fn(_make_logits(), _make_logits(), _make_labels(), _make_labels())
    acc = metrics["accuracy"].item()
    assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"


def test_accuracy_perfect_when_chosen_dominates():
    """When chosen log-probs >> rejected, accuracy should be 1.0."""
    loss_fn = _default_loss_fn()

    # Craft logits so chosen gets uniformly high log-prob and rejected gets low.
    V_local = 8
    # chosen: uniform over first token
    chosen_logits = torch.zeros(B, T, V_local)
    chosen_logits[:, :, 0] = 10.0  # very high prob on token 0

    rejected_logits = torch.zeros(B, T, V_local)
    rejected_logits[:, :, 0] = -10.0  # very low prob on token 0

    # Labels all point to token 0.
    labels = torch.zeros(B, T, dtype=torch.long)

    _, metrics = loss_fn(chosen_logits, rejected_logits, labels, labels)
    assert metrics["accuracy"].item() == pytest.approx(1.0), (
        f"Expected accuracy=1.0 when chosen dominates, got {metrics['accuracy'].item()}"
    )


# ---------------------------------------------------------------------------
# 7. ORPOTrainer
# ---------------------------------------------------------------------------


def test_compute_log_probs_shape():
    """compute_log_probs must return a (B,) tensor."""
    model = _TinyLM(V=V, d=32)
    trainer = ORPOTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
        loss_fn=ORPOLoss(ORPOConfig()),
    )
    tokens = torch.randint(0, V, (B, T))
    logits = model(tokens)
    labels = _make_labels()

    log_probs = trainer.compute_log_probs(model, logits, labels)
    assert log_probs.shape == (B,), (
        f"compute_log_probs expected shape ({B},), got {log_probs.shape}"
    )


def test_train_step_returns_correct_keys():
    """train_step must return a dict with all four metric keys."""
    torch.manual_seed(7)
    model = _TinyLM(V=V, d=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = ORPOTrainer(model=model, optimizer=optimizer, loss_fn=ORPOLoss(ORPOConfig()))

    tokens = torch.randint(0, V, (B, T))
    chosen_logits = model(tokens)
    rejected_logits = model(tokens).detach()
    labels = _make_labels()

    metrics = trainer.train_step(chosen_logits, rejected_logits, labels, labels)

    required = {"sft_loss", "odds_ratio_loss", "total_loss", "accuracy"}
    assert required == set(metrics.keys()), (
        f"train_step metric keys mismatch. Expected {required}, got {set(metrics.keys())}"
    )


def test_train_step_loss_finite():
    """train_step total_loss must be finite."""
    torch.manual_seed(11)
    model = _TinyLM(V=V, d=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = ORPOTrainer(model=model, optimizer=optimizer, loss_fn=ORPOLoss(ORPOConfig()))

    tokens = torch.randint(0, V, (B, T))
    chosen_logits = model(tokens)
    rejected_logits = model(tokens).detach()
    labels = _make_labels()

    metrics = trainer.train_step(chosen_logits, rejected_logits, labels, labels)
    assert torch.isfinite(metrics["total_loss"]), (
        f"train_step returned non-finite total_loss: {metrics['total_loss'].item()}"
    )
