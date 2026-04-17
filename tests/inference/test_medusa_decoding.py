"""
Tests for src/inference/medusa_decoding.py

Tiny config: d_model=16, vocab=16, n_heads=2, seq_len=8, top_k=2, batch=2
All tests run forward and/or backward passes.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import pytest

from src.inference.medusa_decoding import (
    MedusaHead,
    MedusaLoss,
    MedusaModel,
    MedusaTrainer,
    MedusaTreeDecoder,
)

# ---------------------------------------------------------------------------
# Tiny config
# ---------------------------------------------------------------------------
D_MODEL = 16
VOCAB = 16
N_HEADS = 2
SEQ_LEN = 8
TOP_K = 2
BATCH = 2


# ---------------------------------------------------------------------------
# Tiny base model: Embedding -> Linear, returns (logits, hidden_states)
# ---------------------------------------------------------------------------

class TinyBaseModel(nn.Module):
    """Tiny base model for testing.  Returns (logits, hidden_states)."""

    def __init__(self, vocab_size: int = VOCAB, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embed(input_ids)          # (B, T, D)
        logits = self.proj(hidden)              # (B, T, V)
        return logits, hidden


def make_base() -> TinyBaseModel:
    return TinyBaseModel(VOCAB, D_MODEL)


def make_model(n_heads: int = N_HEADS) -> MedusaModel:
    return MedusaModel(make_base(), D_MODEL, VOCAB, n_heads=n_heads)


def make_input(batch: int = BATCH, seq_len: int = SEQ_LEN) -> torch.Tensor:
    return torch.randint(0, VOCAB, (batch, seq_len))


# ---------------------------------------------------------------------------
# Test 1: MedusaHead output shape
# ---------------------------------------------------------------------------

def test_medusa_head_output_shape():
    head = MedusaHead(D_MODEL, VOCAB, head_idx=1)
    hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = head(hidden)
    assert out.shape == (BATCH, SEQ_LEN, VOCAB), f"Expected {(BATCH, SEQ_LEN, VOCAB)}, got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: MedusaHead stores head_idx
# ---------------------------------------------------------------------------

def test_medusa_head_stores_head_idx():
    for idx in [1, 2, 5]:
        head = MedusaHead(D_MODEL, VOCAB, head_idx=idx)
        assert head.head_idx == idx, f"Expected head_idx={idx}, got {head.head_idx}"


# ---------------------------------------------------------------------------
# Test 3: MedusaModel returns (base_logits, list of K medusa_logits)
# ---------------------------------------------------------------------------

def test_medusa_model_returns_correct_types():
    model = make_model()
    ids = make_input()
    base_logits, medusa_logits = model(ids)
    assert isinstance(base_logits, torch.Tensor)
    assert isinstance(medusa_logits, list)
    assert len(medusa_logits) == N_HEADS


# ---------------------------------------------------------------------------
# Test 4: MedusaModel -- all logit tensors have shape (B, T, V)
# ---------------------------------------------------------------------------

def test_medusa_model_all_logit_shapes():
    model = make_model()
    ids = make_input()
    base_logits, medusa_logits = model(ids)
    assert base_logits.shape == (BATCH, SEQ_LEN, VOCAB)
    for k, ml in enumerate(medusa_logits):
        assert ml.shape == (BATCH, SEQ_LEN, VOCAB), f"Head {k}: {ml.shape}"


# ---------------------------------------------------------------------------
# Test 5: Medusa heads have trainable params, base can be frozen
# ---------------------------------------------------------------------------

def test_medusa_heads_trainable_base_frozen():
    model = make_model()
    # Freeze base
    for p in model.base_model.parameters():
        p.requires_grad = False
    # Medusa heads still trainable
    for p in model.medusa_heads.parameters():
        assert p.requires_grad, "Medusa head param should be trainable"
    # Base is frozen
    for p in model.base_model.parameters():
        assert not p.requires_grad, "Base param should be frozen"


# ---------------------------------------------------------------------------
# Test 6: MedusaLoss total_loss is scalar and finite
# ---------------------------------------------------------------------------

def test_medusa_loss_scalar_finite():
    model = make_model()
    loss_fn = MedusaLoss(n_heads=N_HEADS)
    ids = make_input()
    base_logits, medusa_logits = model(ids)
    total_loss, head_losses = loss_fn(base_logits, medusa_logits, ids)
    assert total_loss.shape == torch.Size([]), "total_loss should be scalar"
    assert math.isfinite(total_loss.item()), "total_loss should be finite"


# ---------------------------------------------------------------------------
# Test 7: MedusaLoss head_losses length == n_heads
# ---------------------------------------------------------------------------

def test_medusa_loss_head_losses_length():
    model = make_model()
    loss_fn = MedusaLoss(n_heads=N_HEADS)
    ids = make_input()
    base_logits, medusa_logits = model(ids)
    _, head_losses = loss_fn(base_logits, medusa_logits, ids)
    assert len(head_losses) == N_HEADS, f"Expected {N_HEADS}, got {len(head_losses)}"


# ---------------------------------------------------------------------------
# Test 8: MedusaLoss geometric decay weights decrease across heads
# ---------------------------------------------------------------------------

def test_medusa_loss_geometric_decay_weights():
    n = 4
    loss_fn = MedusaLoss(n_heads=n)
    weights = loss_fn.head_weights
    assert len(weights) == n
    for i in range(n - 1):
        assert weights[i] > weights[i + 1], (
            f"Weight at idx {i} ({weights[i]}) should be > weight at idx {i+1} ({weights[i+1]})"
        )


# ---------------------------------------------------------------------------
# Test 9: MedusaLoss backward through all heads succeeds
# ---------------------------------------------------------------------------

def test_medusa_loss_backward():
    model = make_model()
    loss_fn = MedusaLoss(n_heads=N_HEADS)
    ids = make_input()
    base_logits, medusa_logits = model(ids)
    total_loss, _ = loss_fn(base_logits, medusa_logits, ids)
    total_loss.backward()
    # Check at least one Medusa head param has a gradient
    has_grad = any(
        p.grad is not None for p in model.medusa_heads.parameters()
    )
    assert has_grad, "Medusa head params should have gradients after backward"


# ---------------------------------------------------------------------------
# Test 10: MedusaTreeDecoder.draft_candidates shape (top_k^K, K)
# ---------------------------------------------------------------------------

def test_draft_candidates_shape():
    model = make_model()
    decoder = MedusaTreeDecoder(model, top_k=TOP_K)
    ids = make_input(batch=1)
    candidates = decoder.draft_candidates(ids)
    expected_rows = TOP_K ** N_HEADS
    assert candidates.shape == (expected_rows, N_HEADS), (
        f"Expected {(expected_rows, N_HEADS)}, got {candidates.shape}"
    )


# ---------------------------------------------------------------------------
# Test 11: MedusaTreeDecoder.draft_candidates -- all ids are valid vocab ids
# ---------------------------------------------------------------------------

def test_draft_candidates_valid_vocab_ids():
    model = make_model()
    decoder = MedusaTreeDecoder(model, top_k=TOP_K)
    ids = make_input(batch=1)
    candidates = decoder.draft_candidates(ids)
    assert (candidates >= 0).all(), "All candidate ids should be >= 0"
    assert (candidates < VOCAB).all(), f"All candidate ids should be < {VOCAB}"


# ---------------------------------------------------------------------------
# Test 12: MedusaTreeDecoder.verify -- accepted shape valid, n_accepted in [0, K]
# ---------------------------------------------------------------------------

def test_verify_shape_and_bounds():
    model = make_model()
    decoder = MedusaTreeDecoder(model, top_k=TOP_K)
    ids = make_input(batch=1)
    candidates = decoder.draft_candidates(ids)
    accepted, n_accepted = decoder.verify(ids, candidates)
    assert 0 <= n_accepted <= N_HEADS, f"n_accepted={n_accepted} out of [0, {N_HEADS}]"
    assert accepted.shape == (n_accepted,), (
        f"Expected accepted shape ({n_accepted},), got {accepted.shape}"
    )


# ---------------------------------------------------------------------------
# Test 13: MedusaTreeDecoder.generate -- output shape (B, T + max_new_tokens)
# ---------------------------------------------------------------------------

def test_generate_output_shape():
    model = make_model()
    decoder = MedusaTreeDecoder(model, top_k=TOP_K)
    ids = make_input(batch=BATCH, seq_len=SEQ_LEN)
    max_new = 4
    output = decoder.generate(ids, max_new_tokens=max_new)
    assert output.shape == (BATCH, SEQ_LEN + max_new), (
        f"Expected {(BATCH, SEQ_LEN + max_new)}, got {output.shape}"
    )


# ---------------------------------------------------------------------------
# Test 14: MedusaTrainer.train_step -- all keys present, loss finite
# ---------------------------------------------------------------------------

def test_trainer_train_step_keys_and_finite():
    model = make_model()
    opt = torch.optim.Adam(model.medusa_heads.parameters(), lr=1e-3)
    trainer = MedusaTrainer(model, opt, freeze_base=True)
    ids = make_input()
    result = trainer.train_step(ids)
    assert "total_loss" in result
    assert "head_losses" in result
    assert "grad_norm" in result
    assert math.isfinite(result["total_loss"]), "total_loss should be finite"
    assert math.isfinite(result["grad_norm"]), "grad_norm should be finite"


# ---------------------------------------------------------------------------
# Test 15: MedusaTrainer -- after freeze_base(), base params are non-trainable
# ---------------------------------------------------------------------------

def test_trainer_freeze_base():
    model = make_model()
    opt = torch.optim.Adam(model.medusa_heads.parameters(), lr=1e-3)
    trainer = MedusaTrainer(model, opt, freeze_base=False)
    trainer.freeze_base()
    for p in model.base_model.parameters():
        assert not p.requires_grad, "Base param should be frozen after freeze_base()"


# ---------------------------------------------------------------------------
# Test 16: MedusaTrainer.acceptance_rate_estimate returns float in [0, 1]
# ---------------------------------------------------------------------------

def test_acceptance_rate_estimate_range():
    model = make_model()
    opt = torch.optim.Adam(model.medusa_heads.parameters(), lr=1e-3)
    trainer = MedusaTrainer(model, opt, freeze_base=True)
    ids = make_input()
    rate = trainer.acceptance_rate_estimate(ids)
    assert isinstance(rate, float), f"Expected float, got {type(rate)}"
    assert 0.0 <= rate <= 1.0, f"Acceptance rate {rate} out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 17: MedusaHead n_layers=2 still produces (B, T, V)
# ---------------------------------------------------------------------------

def test_medusa_head_deep_output_shape():
    head = MedusaHead(D_MODEL, VOCAB, head_idx=1, n_layers=2)
    hidden = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = head(hidden)
    assert out.shape == (BATCH, SEQ_LEN, VOCAB), (
        f"Deep head: expected {(BATCH, SEQ_LEN, VOCAB)}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 18: 3-step training -- loss stays finite (and should generally decrease)
# ---------------------------------------------------------------------------

def test_three_step_training_finite():
    model = make_model()
    opt = torch.optim.Adam(model.medusa_heads.parameters(), lr=1e-2)
    trainer = MedusaTrainer(model, opt, freeze_base=True)
    ids = make_input()
    losses = []
    for _ in range(3):
        result = trainer.train_step(ids)
        losses.append(result["total_loss"])
    for loss in losses:
        assert math.isfinite(loss), f"Loss became non-finite: {loss}"


# ---------------------------------------------------------------------------
# Test 19: top_k=1 with n_heads=2 -> draft_candidates shape (1, 2)
# ---------------------------------------------------------------------------

def test_draft_candidates_top_k_1():
    model = make_model(n_heads=2)
    decoder = MedusaTreeDecoder(model, top_k=1)
    ids = make_input(batch=1)
    candidates = decoder.draft_candidates(ids)
    assert candidates.shape == (1, 2), f"Expected (1, 2), got {candidates.shape}"
