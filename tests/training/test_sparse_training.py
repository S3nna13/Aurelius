"""Tests for sparse_training: target-loss pruning and L0 hard-concrete regularization."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.sparse_training import (
    HardConcrete,
    L0LinearLayer,
    PruningResult,
    TargetLossPruningConfig,
    add_l0_regularization,
    compute_model_sparsity,
    evaluate_model_loss,
    l0_regularization_loss,
    prune_magnitude,
    target_loss_prune,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_model():
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
    )
    return model


@pytest.fixture
def all_zero_model():
    model = nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    return model


@pytest.fixture
def no_zero_model():
    torch.manual_seed(1)
    model = nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    return model


def _make_batches(n=5, seq_len=8, vocab=256):
    batches = []
    for _ in range(n):
        input_ids = torch.randint(0, vocab, (4, seq_len))
        labels = torch.randint(0, vocab, (4, seq_len))
        batches.append((input_ids, labels))
    return batches


class TinyLM(nn.Module):
    """Minimal LM for testing evaluate_model_loss."""

    def __init__(self, vocab=256, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.proj = nn.Linear(d, vocab)

    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        logits = self.proj(x)
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss, logits
        return logits


# ── 1. compute_model_sparsity ──────────────────────────────────────────────────


def test_compute_model_sparsity_all_zero(all_zero_model):
    sparsity = compute_model_sparsity(all_zero_model)
    assert sparsity == pytest.approx(1.0)


def test_compute_model_sparsity_none_zero(no_zero_model):
    sparsity = compute_model_sparsity(no_zero_model)
    assert sparsity == pytest.approx(0.0)


# ── 2. prune_magnitude ────────────────────────────────────────────────────────


def test_prune_magnitude_count(simple_model):
    n_zeroed = prune_magnitude(simple_model, fraction=0.1)
    assert isinstance(n_zeroed, int)
    assert n_zeroed > 0


def test_prune_magnitude_increases_sparsity(simple_model):
    before = compute_model_sparsity(simple_model)
    prune_magnitude(simple_model, fraction=0.2)
    after = compute_model_sparsity(simple_model)
    assert after > before


def test_prune_magnitude_zeros_smallest(simple_model):
    """After pruning, surviving weights should be the largest by magnitude."""
    all_weights = torch.cat([p.data.abs().flatten() for p in simple_model.parameters()])
    total = all_weights.numel()
    fraction = 0.3
    n_to_prune = int(total * fraction)

    prune_magnitude(simple_model, fraction=fraction)

    sorted_w, _ = all_weights.sort()
    threshold = sorted_w[n_to_prune - 1].item()

    remaining = torch.cat([p.data.abs().flatten() for p in simple_model.parameters()])
    surviving_nonzero = remaining[remaining > 0]
    assert (surviving_nonzero >= threshold - 1e-6).all()


# ── 3. evaluate_model_loss ────────────────────────────────────────────────────


def test_evaluate_model_loss_positive():
    torch.manual_seed(0)
    model = TinyLM()
    batches = _make_batches(n=3)
    loss = evaluate_model_loss(model, batches)
    assert isinstance(loss, float)
    assert loss > 0.0


# ── 4. target_loss_prune ──────────────────────────────────────────────────────


def test_target_loss_prune_returns_result():
    torch.manual_seed(0)
    model = TinyLM()
    batches = _make_batches(n=5)
    cfg = TargetLossPruningConfig(
        target_loss=1000.0,
        prune_fraction_per_step=0.05,
        n_eval_batches=3,
        max_steps=3,
    )
    result = target_loss_prune(model, batches, cfg)
    assert isinstance(result, PruningResult)
    assert hasattr(result, "final_loss")
    assert hasattr(result, "target_loss")
    assert hasattr(result, "n_steps")
    assert hasattr(result, "sparsity")
    assert hasattr(result, "converged")
    assert hasattr(result, "loss_history")
    assert hasattr(result, "sparsity_history")


def test_target_loss_prune_converges():
    """converged=True when target_loss is set above the initial model loss."""
    torch.manual_seed(0)
    model = TinyLM()
    batches = _make_batches(n=5)
    cfg = TargetLossPruningConfig(
        target_loss=1e9,
        prune_fraction_per_step=0.05,
        n_eval_batches=3,
        max_steps=10,
    )
    result = target_loss_prune(model, batches, cfg)
    assert result.converged is True


def test_target_loss_prune_loss_history_len():
    torch.manual_seed(0)
    model = TinyLM()
    batches = _make_batches(n=5)
    cfg = TargetLossPruningConfig(
        target_loss=0.0,
        prune_fraction_per_step=0.05,
        n_eval_batches=3,
        max_steps=4,
    )
    result = target_loss_prune(model, batches, cfg)
    assert len(result.loss_history) == result.n_steps


# ── 5. HardConcrete ──────────────────────────────────────────────────────────


def test_hard_concrete_output_range():
    torch.manual_seed(0)
    hc = HardConcrete(n_weights=128)
    hc.train()
    z = hc()
    assert z.shape == (128,)
    assert (z >= 0).all()
    assert (z <= 1).all()


def test_hard_concrete_l0_penalty_range():
    torch.manual_seed(0)
    n = 64
    hc = HardConcrete(n_weights=n)
    penalty = hc.l0_penalty()
    assert penalty.item() >= 0.0
    assert penalty.item() <= float(n)


def test_hard_concrete_eval_mode_binary():
    torch.manual_seed(0)
    hc = HardConcrete(n_weights=100)
    hc.eval()
    z = hc()
    unique_vals = z.unique()
    for v in unique_vals:
        assert v.item() in (0.0, 1.0)


# ── 6. L0LinearLayer ─────────────────────────────────────────────────────────


def test_l0_linear_forward_shape():
    torch.manual_seed(0)
    layer = L0LinearLayer(in_features=16, out_features=32)
    layer.train()
    x = torch.randn(4, 16)
    out = layer(x)
    assert out.shape == (4, 32)


def test_l0_linear_l0_penalty_positive():
    torch.manual_seed(0)
    layer = L0LinearLayer(in_features=16, out_features=32)
    penalty = layer.l0_penalty()
    assert penalty.item() >= 0.0


# ── 7. l0_regularization_loss ────────────────────────────────────────────────


def test_l0_regularization_loss_sum():
    torch.manual_seed(0)
    model = nn.Sequential(
        L0LinearLayer(8, 16),
        nn.ReLU(),
        L0LinearLayer(16, 8),
    )
    total = l0_regularization_loss(model, l0_lambda=1.0)
    expected = sum(m.l0_penalty().item() for m in model.modules() if isinstance(m, L0LinearLayer))
    assert total.item() == pytest.approx(expected, rel=1e-5)


# ── 8. add_l0_regularization ─────────────────────────────────────────────────


def test_add_l0_regularization_count():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )
    n_replaced = add_l0_regularization(model)
    assert n_replaced == 3
    linear_types = [type(m) for m in model.modules() if isinstance(m, (nn.Linear, L0LinearLayer))]
    assert all(t is L0LinearLayer for t in linear_types)
