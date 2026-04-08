"""Tests for SimCSE contrastive embedding training."""
from __future__ import annotations

import copy
import torch
import pytest

from src.alignment.simcse import SimCSEConfig, SimCSETrainer, extract_embeddings, simcse_loss
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_model():
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
        dropout=0.1,  # critical: enables different dropout masks per forward pass
    )
    return AureliusTransformer(cfg)


def _make_input(batch_size: int = 4, seq_len: int = 8, vocab_size: int = 256) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randint(0, vocab_size, (batch_size, seq_len))


# ---------------------------------------------------------------------------
# simcse_loss tests
# ---------------------------------------------------------------------------

def test_simcse_loss_shape():
    """simcse_loss returns a 0-dim scalar tensor."""
    torch.manual_seed(0)
    N, D = 4, 64
    a = torch.randn(N, D)
    b = torch.randn(N, D)
    loss = simcse_loss(a, b)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"


def test_simcse_loss_nonnegative():
    """simcse_loss is always >= 0."""
    torch.manual_seed(1)
    N, D = 4, 64
    a = torch.randn(N, D)
    b = torch.randn(N, D)
    loss = simcse_loss(a, b)
    assert loss.item() >= 0.0, f"Loss should be non-negative, got {loss.item()}"


def test_simcse_loss_zero_with_identical_embeddings():
    """When emb_a == emb_b are perfectly aligned unit vectors, loss should be 0."""
    N, D = 4, 64
    # Use one-hot basis vectors: orthogonal and unit-normed, so diagonal sim=1, off-diag=0
    base = torch.zeros(N, D)
    for i in range(N):
        base[i, i] = 1.0
    a = base
    b = base.clone()

    # With a[i]==b[i] and orthogonal rows, sim[i,i]=1/temp, sim[i,j]=0 for i!=j
    # cross_entropy of one-hot logits -> loss = 0
    loss = simcse_loss(a, b, temperature=0.05)
    assert loss.item() < 1e-5, f"Expected ~0 loss with identical embeddings, got {loss.item()}"


def test_simcse_loss_batch_size_1():
    """N=1 should not crash (single positive pair, no negatives)."""
    torch.manual_seed(3)
    a = torch.randn(1, 64)
    b = torch.randn(1, 64)
    loss = simcse_loss(a, b)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# extract_embeddings tests
# ---------------------------------------------------------------------------

def test_extract_embeddings_shape(small_model):
    """extract_embeddings returns (B, D=64) tensor."""
    input_ids = _make_input(batch_size=4, seq_len=8)
    cfg = SimCSEConfig(pooling="mean", normalize=False)
    small_model.eval()
    emb = extract_embeddings(small_model, input_ids, cfg)
    assert emb.shape == (4, 64), f"Expected (4, 64), got {emb.shape}"


def test_extract_embeddings_normalized(small_model):
    """With normalize=True, all embedding norms should be ~1.0."""
    input_ids = _make_input(batch_size=4, seq_len=8)
    cfg = SimCSEConfig(pooling="mean", normalize=True)
    small_model.eval()
    emb = extract_embeddings(small_model, input_ids, cfg)
    norms = emb.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5), \
        f"Norms should be ~1.0, got: {norms}"


def test_extract_embeddings_mean_pooling(small_model):
    """SimCSEConfig(pooling='mean') should work without error and give finite output."""
    input_ids = _make_input(batch_size=2, seq_len=8)
    cfg = SimCSEConfig(pooling="mean")
    small_model.eval()
    emb = extract_embeddings(small_model, input_ids, cfg)
    assert emb.shape == (2, 64)
    assert torch.isfinite(emb).all()


# ---------------------------------------------------------------------------
# SimCSETrainer tests
# ---------------------------------------------------------------------------

def test_train_step_returns_finite_loss(small_model):
    """train_step should return a finite float loss."""
    input_ids = _make_input(batch_size=4, seq_len=8)
    trainer = SimCSETrainer(small_model, cfg=SimCSEConfig(), lr=3e-5)
    loss_val = trainer.train_step(input_ids)
    assert isinstance(loss_val, float), f"Expected float, got {type(loss_val)}"
    assert torch.isfinite(torch.tensor(loss_val)), f"Loss is not finite: {loss_val}"


def test_train_step_updates_weights(small_model):
    """At least one weight should change after a train_step."""
    input_ids = _make_input(batch_size=4, seq_len=8)
    trainer = SimCSETrainer(small_model, cfg=SimCSEConfig(), lr=3e-5)

    # Capture a snapshot of all parameters before the step
    params_before = {
        name: p.data.clone()
        for name, p in small_model.named_parameters()
    }

    trainer.train_step(input_ids)

    # At least one parameter must have changed
    changed = any(
        not torch.equal(params_before[name], p.data)
        for name, p in small_model.named_parameters()
    )
    assert changed, "No parameters were updated after train_step"


def test_encode_returns_embeddings(small_model):
    """encode() should return (B, D) tensor with no grad and finite values."""
    input_ids = _make_input(batch_size=3, seq_len=8)
    trainer = SimCSETrainer(small_model, cfg=SimCSEConfig(normalize=True))
    emb = trainer.encode(input_ids)
    assert emb.shape == (3, 64), f"Expected (3, 64), got {emb.shape}"
    assert not emb.requires_grad, "encode() output should not require grad"
    assert torch.isfinite(emb).all(), "Embeddings contain non-finite values"
