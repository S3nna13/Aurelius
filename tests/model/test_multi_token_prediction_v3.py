"""Tests for multi_token_prediction_v3.py

Tiny config: d_model=16, n_heads=2, n_layers=2, vocab_size=16,
             seq_len=8, batch=2, k=3
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.optim as optim
from aurelius.model.multi_token_prediction_v3 import (
    MTPTrainer,
    MultiTokenHead,
    MultiTokenLoss,
    MultiTokenPredictionModel,
    SpeculativeDecodingFromMTP,
)

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------
D_MODEL = 16
N_HEADS = 2
N_LAYERS = 2
VOCAB_SIZE = 16
SEQ_LEN = 8
BATCH = 2
K = 3


def make_model(k: int = K) -> MultiTokenPredictionModel:
    return MultiTokenPredictionModel(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        vocab_size=VOCAB_SIZE,
        k_predictions=k,
    )


def make_ids(seq_len: int = SEQ_LEN, batch: int = BATCH) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (batch, seq_len))


# ---------------------------------------------------------------------------
# 1. MultiTokenHead — output shape (B, T, V)
# ---------------------------------------------------------------------------
def test_mth_output_shape():
    head = MultiTokenHead(D_MODEL, VOCAB_SIZE, offset=1)
    hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    logits = head(hidden)
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# 2. MultiTokenHead — offset stored correctly
# ---------------------------------------------------------------------------
def test_mth_offset_stored():
    for off in [1, 2, 5]:
        head = MultiTokenHead(D_MODEL, VOCAB_SIZE, offset=off)
        assert head.offset == off


# ---------------------------------------------------------------------------
# 3. MultiTokenPredictionModel — returns list of K logit tensors
# ---------------------------------------------------------------------------
def test_mtpm_returns_k_tensors():
    model = make_model()
    ids = make_ids()
    logits_list = model(ids)
    assert len(logits_list) == K


# ---------------------------------------------------------------------------
# 4. MultiTokenPredictionModel — each logit tensor is (B, T, V)
# ---------------------------------------------------------------------------
def test_mtpm_logit_shapes():
    model = make_model()
    ids = make_ids()
    logits_list = model(ids)
    for logits in logits_list:
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)


# ---------------------------------------------------------------------------
# 5. MultiTokenPredictionModel — grad flows to all K heads from loss
# ---------------------------------------------------------------------------
def test_mtpm_grad_flows_to_all_heads():
    model = make_model()
    ids = make_ids()
    loss_fn = MultiTokenLoss(K)
    logits_list = model(ids)
    total_loss, _ = loss_fn(logits_list, ids)
    total_loss.backward()
    for i, head in enumerate(model.heads):
        for name, param in head.named_parameters():
            assert param.grad is not None, f"head[{i}].{name} has no grad"
            assert not torch.all(param.grad == 0), f"head[{i}].{name} grad is all zeros"


# ---------------------------------------------------------------------------
# 6. MultiTokenLoss — total_loss is a scalar
# ---------------------------------------------------------------------------
def test_mtl_total_loss_scalar():
    model = make_model()
    ids = make_ids()
    loss_fn = MultiTokenLoss(K)
    logits_list = model(ids)
    total_loss, _ = loss_fn(logits_list, ids)
    assert total_loss.shape == torch.Size([])


# ---------------------------------------------------------------------------
# 7. MultiTokenLoss — per_head_losses length == k
# ---------------------------------------------------------------------------
def test_mtl_per_head_losses_length():
    model = make_model()
    ids = make_ids()
    loss_fn = MultiTokenLoss(K)
    logits_list = model(ids)
    _, per_head = loss_fn(logits_list, ids)
    assert len(per_head) == K


# ---------------------------------------------------------------------------
# 8. MultiTokenLoss — each per_head_loss finite and > 0
# ---------------------------------------------------------------------------
def test_mtl_per_head_losses_finite_positive():
    model = make_model()
    ids = make_ids()
    loss_fn = MultiTokenLoss(K)
    logits_list = model(ids)
    _, per_head = loss_fn(logits_list, ids)
    for i, loss_val in enumerate(per_head):
        assert math.isfinite(loss_val), f"head {i} loss is not finite: {loss_val}"
        assert loss_val > 0, f"head {i} loss is not positive: {loss_val}"


# ---------------------------------------------------------------------------
# 9. MultiTokenLoss k=1 — total_loss == per_head_losses[0]
# ---------------------------------------------------------------------------
def test_mtl_k1_total_equals_head0():
    model = make_model(k=1)
    ids = make_ids()
    loss_fn = MultiTokenLoss(k_predictions=1)
    logits_list = model(ids)
    total_loss, per_head = loss_fn(logits_list, ids)
    assert math.isclose(total_loss.item(), per_head[0], rel_tol=1e-5)


# ---------------------------------------------------------------------------
# 10. MTPTrainer.train_step — all expected keys present, loss finite, grad_norm >= 0
# ---------------------------------------------------------------------------
def test_trainer_step_keys_and_finite():
    model = make_model()
    loss_fn = MultiTokenLoss(K)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainer = MTPTrainer(model, loss_fn, optimizer)
    ids = make_ids()
    result = trainer.train_step(ids)
    assert "total_loss" in result
    assert "per_head_losses" in result
    assert "grad_norm" in result
    assert math.isfinite(result["total_loss"])
    assert result["grad_norm"] >= 0.0


# ---------------------------------------------------------------------------
# 11. MTPTrainer — params actually updated after step
# ---------------------------------------------------------------------------
def test_trainer_params_updated():
    model = make_model()
    # Snapshot params before
    params_before = {name: p.clone().detach() for name, p in model.named_parameters()}
    loss_fn = MultiTokenLoss(K)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainer = MTPTrainer(model, loss_fn, optimizer)
    ids = make_ids()
    trainer.train_step(ids)
    # At least some params must have changed
    changed = any(
        not torch.allclose(p, params_before[name]) for name, p in model.named_parameters()
    )
    assert changed, "No parameters were updated after train_step"


# ---------------------------------------------------------------------------
# 12. SpeculativeDecodingFromMTP.draft — shape (B, k), all ids in [0, vocab_size)
# ---------------------------------------------------------------------------
def test_speculative_draft_shape_and_range():
    model = make_model()
    spec = SpeculativeDecodingFromMTP(model, k=K)
    ids = make_ids()
    draft = spec.draft(ids)
    assert draft.shape == (BATCH, K)
    assert draft.min().item() >= 0
    assert draft.max().item() < VOCAB_SIZE


# ---------------------------------------------------------------------------
# 13. SpeculativeDecodingFromMTP.verify_and_accept — accepted shape (B,k) bool,
#     n_accepted in [0, k]
# ---------------------------------------------------------------------------
def test_speculative_verify_and_accept():
    model = make_model()
    spec = SpeculativeDecodingFromMTP(model, k=K)
    ids = make_ids()
    draft = spec.draft(ids)
    # Create fake target logits
    target_logits = torch.randn(BATCH, K, VOCAB_SIZE)
    accepted, n_accepted = spec.verify_and_accept(draft, target_logits)
    assert accepted.shape == (BATCH, K)
    assert accepted.dtype == torch.bool
    assert 0 <= n_accepted <= K


# ---------------------------------------------------------------------------
# 14. MultiTokenLoss custom weights — weighted sum matches manual calculation
# ---------------------------------------------------------------------------
def test_mtl_custom_weights():
    model = make_model(k=K)
    ids = make_ids()
    weights = [0.5, 0.3, 0.2]
    loss_fn_weighted = MultiTokenLoss(k_predictions=K, weights=weights)
    MultiTokenLoss(k_predictions=K, weights=[1.0] * K)
    logits_list = model(ids)
    total_weighted, per_head = loss_fn_weighted(logits_list, ids)
    # Manually compute expected weighted total
    expected = sum(w * lv for w, lv in zip(weights, per_head))
    assert math.isclose(total_weighted.item(), expected, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# 15. Multiple k values: k=1,2,4 all produce correct output shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 2, 4])
def test_multiple_k_values(k):
    model = make_model(k=k)
    ids = make_ids()
    logits_list = model(ids)
    assert len(logits_list) == k
    for logits in logits_list:
        assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE)
    # Also confirm loss computation works
    loss_fn = MultiTokenLoss(k_predictions=k)
    total_loss, per_head = loss_fn(logits_list, ids)
    assert total_loss.shape == torch.Size([])
    assert len(per_head) == k
