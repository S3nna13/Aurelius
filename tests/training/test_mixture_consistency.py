"""Tests for src/training/mixture_consistency.py.

All tests use tiny dimensions so they run on CPU in milliseconds.
Dimensions: B=2 (batch), T=8 (seq len), V=16 (vocab), D=32 (d_model).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.training.mixture_consistency import (
    MixtureConsistencyConfig,
    MixtureConsistencyLoss,
    MixtureConsistencyTrainer,
    token_dropout_augment,
)

# ---------------------------------------------------------------------------
# Shared tiny constants
# ---------------------------------------------------------------------------

B, T, V, D = 2, 8, 16, 32


# ---------------------------------------------------------------------------
# Tiny mock model
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Linear projection from token embeddings to logits."""

    def __init__(self, vocab: int = V, d_model: int = D) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: (batch, seq)  -> logits: (batch, seq, vocab)
        return self.proj(self.embed(input_ids))


def _make_model() -> TinyModel:
    torch.manual_seed(0)
    return TinyModel(vocab=V, d_model=D)


def _make_logits(seed: int = 42) -> Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T, V)


def _base_loss_fn(logits: Tensor, labels: Tensor) -> Tensor:
    """Cross-entropy over (batch, seq) flattened."""
    return F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1))


# ---------------------------------------------------------------------------
# Test 1: MixtureConsistencyConfig defaults are correct
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = MixtureConsistencyConfig()
    assert cfg.consistency_weight == 0.1
    assert cfg.temperature == 1.0
    assert cfg.distance_fn == "kl"
    assert cfg.dropout_rate == 0.15
    assert cfg.n_augmented_views == 2


# ---------------------------------------------------------------------------
# Test 2: kl_consistency returns a scalar >= 0
# ---------------------------------------------------------------------------


def test_kl_consistency_nonnegative() -> None:
    loss_fn = MixtureConsistencyLoss(distance_fn="kl")
    p = F.softmax(_make_logits(1), dim=-1)
    q = F.softmax(_make_logits(2), dim=-1)
    val = loss_fn.kl_consistency(p, q)
    assert val.shape == torch.Size([])
    assert val.item() >= 0.0


# ---------------------------------------------------------------------------
# Test 3: kl_consistency of identical distributions == 0
# ---------------------------------------------------------------------------


def test_kl_consistency_identical_is_zero() -> None:
    loss_fn = MixtureConsistencyLoss(distance_fn="kl")
    p = F.softmax(_make_logits(7), dim=-1)
    val = loss_fn.kl_consistency(p, p.clone())
    assert val.item() < 1e-6


# ---------------------------------------------------------------------------
# Test 4: js_divergence returns value in [0, log(2)]
# ---------------------------------------------------------------------------


def test_js_divergence_in_range() -> None:
    loss_fn = MixtureConsistencyLoss(distance_fn="js")
    p = F.softmax(_make_logits(3), dim=-1)
    q = F.softmax(_make_logits(4), dim=-1)
    val = loss_fn.js_divergence(p, q)
    assert val.item() >= 0.0
    assert val.item() <= math.log(2) + 1e-6


# ---------------------------------------------------------------------------
# Test 5: cosine_consistency of identical tensors == 0
# ---------------------------------------------------------------------------


def test_cosine_consistency_identical_is_zero() -> None:
    loss_fn = MixtureConsistencyLoss(distance_fn="cosine")
    h = torch.randn(B, T, D)
    val = loss_fn.cosine_consistency(h, h.clone())
    assert abs(val.item()) < 1e-5


# ---------------------------------------------------------------------------
# Test 6: create_augmented_pair returns same-shape pair
# ---------------------------------------------------------------------------


def test_create_augmented_pair_shape() -> None:
    model = _make_model()
    trainer = MixtureConsistencyTrainer(model, base_loss_fn=_base_loss_fn)
    torch.manual_seed(99)
    input_ids = torch.randint(0, V, (B, T))
    view1, view2 = trainer.create_augmented_pair(input_ids)
    assert view1.shape == input_ids.shape
    assert view2.shape == input_ids.shape


# ---------------------------------------------------------------------------
# Test 7: Augmented pair has some changed tokens (not identical to original)
# ---------------------------------------------------------------------------


def test_augmented_pair_not_identical() -> None:
    model = _make_model()
    trainer = MixtureConsistencyTrainer(model, base_loss_fn=_base_loss_fn)
    # Use large input to ensure some tokens get dropped with high probability
    torch.manual_seed(5)
    input_ids = torch.randint(1, V, (B, 64))  # seq_len=64 for reliability
    view1, view2 = trainer.create_augmented_pair(input_ids)
    # At least one view should differ from the original
    changed = (view1 != input_ids).any() or (view2 != input_ids).any()
    assert changed


# ---------------------------------------------------------------------------
# Test 8: token_dropout_augment changes some tokens
# ---------------------------------------------------------------------------


def test_token_dropout_augment_changes_tokens() -> None:
    torch.manual_seed(11)
    ids = torch.randint(1, V, (4, 64))  # seq_len=64 for reliable change detection
    augmented = token_dropout_augment(ids, dropout_rate=0.5, unk_id=0)
    assert (augmented != ids).any()


# ---------------------------------------------------------------------------
# Test 9: token_dropout_augment with rate=0 makes no changes
# ---------------------------------------------------------------------------


def test_token_dropout_augment_zero_rate() -> None:
    ids = torch.randint(1, V, (B, T))
    augmented = token_dropout_augment(ids, dropout_rate=0.0, unk_id=0)
    assert torch.equal(augmented, ids)


# ---------------------------------------------------------------------------
# Test 10: train_step returns dict with all required keys
# ---------------------------------------------------------------------------


def test_train_step_returns_required_keys() -> None:
    model = _make_model()
    trainer = MixtureConsistencyTrainer(model, base_loss_fn=_base_loss_fn)
    torch.manual_seed(21)
    input_ids = torch.randint(0, V, (B, T))
    labels = torch.randint(0, V, (B, T))
    result = trainer.train_step(input_ids, labels)
    assert set(result.keys()) == {"loss", "base_loss", "consistency_loss", "ratio"}


# ---------------------------------------------------------------------------
# Test 11: consistency_weight=0 means consistency_loss does not affect total
# ---------------------------------------------------------------------------


def test_zero_consistency_weight_no_effect() -> None:
    model = _make_model()
    trainer = MixtureConsistencyTrainer(
        model,
        base_loss_fn=_base_loss_fn,
        consistency_weight=0.0,
    )
    torch.manual_seed(33)
    input_ids = torch.randint(0, V, (B, T))
    labels = torch.randint(0, V, (B, T))
    result = trainer.train_step(input_ids, labels)
    # With weight=0, total loss == base_loss
    assert abs(result["loss"].item() - result["base_loss"].item()) < 1e-6


# ---------------------------------------------------------------------------
# Test 12: Gradient flows through combined loss
# ---------------------------------------------------------------------------


def test_gradient_flows_through_combined_loss() -> None:
    torch.manual_seed(44)
    model = _make_model()
    trainer = MixtureConsistencyTrainer(
        model,
        base_loss_fn=_base_loss_fn,
        consistency_weight=0.1,
    )
    input_ids = torch.randint(0, V, (B, T))
    labels = torch.randint(0, V, (B, T))

    result = trainer.train_step(input_ids, labels)
    result["loss"].backward()

    # At least one parameter should have a non-zero gradient
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    assert len(grad_norms) > 0
    assert any(g > 0.0 for g in grad_norms)
