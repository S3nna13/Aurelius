"""Tests for src/training/sift.py"""

import math
import pytest
import torch
import torch.nn as nn

from aurelius.training.sift import InfluenceScorer, SIFTFilter, SIFTLoss, SIFTTrainer

SEED = 42
B, T, V = 4, 8, 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_model():
    return nn.Linear(16, 16)


def _scalar_loss(model, x, target):
    out = model(x)
    return ((out - target) ** 2).mean()


# ---------------------------------------------------------------------------
# InfluenceScorer
# ---------------------------------------------------------------------------

def test_influence_score_range():
    torch.manual_seed(SEED)
    model = _tiny_model()
    scorer = InfluenceScorer(model)
    x = torch.randn(4, 16)
    t = torch.randn(4, 16)
    tl = _scalar_loss(model, x, t)
    vl = _scalar_loss(model, x, t)
    score = scorer.score(tl, vl)
    assert -1.0 <= score <= 1.0


def test_influence_identical_losses_are_one():
    torch.manual_seed(SEED)
    model = _tiny_model()
    scorer = InfluenceScorer(model)
    x = torch.randn(4, 16)
    t = torch.randn(4, 16)
    # Must compute two separate loss tensors from the same graph operation
    # Use retain_graph so the second grad call doesn't fail
    loss_a = _scalar_loss(model, x, t)
    loss_b = _scalar_loss(model, x, t)
    score = scorer.score(loss_a, loss_b)
    assert abs(score - 1.0) < 1e-4


def test_influence_probe_params_subset():
    model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
    param_name = "0.weight"
    scorer = InfluenceScorer(model, probe_params=[param_name])
    x = torch.randn(2, 8)
    tl = ((model(x)) ** 2).mean()
    vl = ((model(x)) ** 2).mean()
    score = scorer.score(tl, vl)
    assert -1.0 <= score <= 1.0


def test_influence_missing_probe_param_raises():
    model = _tiny_model()
    with pytest.raises(ValueError, match="not found"):
        InfluenceScorer(model, probe_params=["nonexistent.weight"])


def test_influence_zero_grad_returns_zero():
    # A loss with zero gradient (constant) → norm = 0 → score = 0
    model = _tiny_model()
    scorer = InfluenceScorer(model)
    x = torch.randn(2, 16)
    # Constant loss wrt params — use a detached computation
    const_loss = torch.zeros(1, requires_grad=True).sum()  # grad is zero
    real_loss = ((model(x)) ** 2).mean()
    score = scorer.score(const_loss, real_loss)
    assert score == 0.0


# ---------------------------------------------------------------------------
# SIFTFilter
# ---------------------------------------------------------------------------

def test_filter_threshold_keeps_positive():
    f = SIFTFilter(threshold=0.0)
    scores = torch.tensor([0.5, -0.1, 0.3, -0.4, 0.0])
    mask = f.filter(scores)
    assert mask[0] and mask[2] and mask[4]
    assert not mask[1] and not mask[3]


def test_filter_threshold_rejects_negative():
    f = SIFTFilter(threshold=0.0)
    scores = torch.tensor([-1.0, -0.5, -0.01])
    mask = f.filter(scores)
    assert not mask.any()


def test_filter_all_zero_threshold_zero():
    f = SIFTFilter(threshold=0.0)
    scores = torch.zeros(5)
    mask = f.filter(scores)
    assert mask.all()  # 0 >= 0


def test_filter_top_k_half():
    f = SIFTFilter(top_k=0.5)
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mask = f.filter(scores)
    assert mask.sum() == math.ceil(0.5 * 4)
    assert mask[2] and mask[3]  # top 2


def test_filter_top_k_all():
    f = SIFTFilter(top_k=1.0)
    scores = torch.randn(6)
    mask = f.filter(scores)
    assert mask.all()


def test_filter_requires_1d():
    f = SIFTFilter()
    with pytest.raises(ValueError):
        f.filter(torch.randn(3, 3))


# ---------------------------------------------------------------------------
# SIFTLoss
# ---------------------------------------------------------------------------

def test_siftloss_uniform_weights_equals_mean_ce():
    torch.manual_seed(SEED)
    loss_fn = SIFTLoss()
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    loss_no_w = loss_fn(logits, targets)
    uniform_w = torch.ones(B)
    loss_w = loss_fn(logits, targets, weights=uniform_w)
    assert torch.allclose(loss_no_w, loss_w, atol=1e-5)


def test_siftloss_zero_weight_sample():
    torch.manual_seed(SEED)
    loss_fn = SIFTLoss()
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    w = torch.ones(B)
    w[0] = 0.0  # zero out first sample
    w_all = torch.ones(B)
    loss_zero = loss_fn(logits, targets, weights=w)
    # Loss should differ from uniform
    loss_all = loss_fn(logits, targets, weights=w_all)
    assert not torch.allclose(loss_zero, loss_all)


def test_siftloss_finite():
    loss_fn = SIFTLoss()
    logits = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    assert torch.isfinite(loss_fn(logits, targets))


def test_siftloss_gradient_flows():
    loss_fn = SIFTLoss()
    logits = torch.randn(B, T, V, requires_grad=True)
    targets = torch.randint(0, V, (B, T))
    loss = loss_fn(logits, targets)
    loss.backward()
    assert logits.grad is not None


# ---------------------------------------------------------------------------
# SIFTTrainer
# ---------------------------------------------------------------------------

def test_sifttrainer_returns_correct_keys():
    torch.manual_seed(SEED)
    model = nn.Linear(V, V)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = SIFTTrainer(model, opt, SIFTFilter(threshold=0.0), SIFTLoss())
    x = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    # Logits must flow through model parameters to support backward()
    logits = model(x)
    val_logits = model(x[:2])
    stats = trainer.train_step(logits, targets, val_logits, targets[:2])
    assert "loss" in stats
    assert "n_kept" in stats
    assert "fraction_kept" in stats


def test_sifttrainer_fraction_in_range():
    model = nn.Linear(V, V)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = SIFTTrainer(model, opt, SIFTFilter(threshold=0.0), SIFTLoss())
    x = torch.randn(B, T, V)
    targets = torch.randint(0, V, (B, T))
    logits = model(x)
    stats = trainer.train_step(logits, targets, model(x), targets)
    assert 0.0 <= stats["fraction_kept"] <= 1.0
