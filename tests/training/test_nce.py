"""Tests for src/training/nce.py — NCE embedding training module."""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.nce import (
    EmbeddingProjector,
    NCEConfig,
    NCEEmbeddingTrainer,
    hard_negative_mining,
    in_batch_nce_loss,
    info_nce_loss,
    sample_random_negatives,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

B = 4  # batch size
T = 8  # sequence length
D = 64  # model d_model
E = 32  # embed_dim for projector

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=D,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model() -> AureliusTransformer:
    torch.manual_seed(0)
    return AureliusTransformer(TINY_CFG)


@pytest.fixture(scope="module")
def projector() -> EmbeddingProjector:
    torch.manual_seed(0)
    return EmbeddingProjector(d_model=D, embed_dim=E)


@pytest.fixture(scope="module")
def nce_cfg() -> NCEConfig:
    return NCEConfig(temperature=0.07, n_negatives=7, embedding_dim=E)


@pytest.fixture(scope="module")
def trainer(tiny_model, projector, nce_cfg) -> NCEEmbeddingTrainer:
    optimizer = torch.optim.Adam(
        list(tiny_model.parameters()) + list(projector.parameters()), lr=1e-4
    )
    return NCEEmbeddingTrainer(
        backbone=tiny_model,
        projector=projector,
        cfg=nce_cfg,
        optimizer=optimizer,
    )


@pytest.fixture
def anchor_ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, TINY_CFG.vocab_size, (B, T))


@pytest.fixture
def positive_ids() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randint(0, TINY_CFG.vocab_size, (B, T))


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = NCEConfig()
    assert cfg.temperature == 0.07
    assert cfg.n_negatives == 7


# ---------------------------------------------------------------------------
# 2. test_info_nce_loss_scalar
# ---------------------------------------------------------------------------


def test_info_nce_loss_scalar():
    torch.manual_seed(0)
    anchors = torch.randn(B, D)
    positives = torch.randn(B, D)
    negatives = torch.randn(B, 5, D)
    loss = info_nce_loss(anchors, positives, negatives)
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert torch.isfinite(loss).item()


# ---------------------------------------------------------------------------
# 3. test_info_nce_loss_positive_match — when pos == anchor, loss is low
# ---------------------------------------------------------------------------


def test_info_nce_loss_positive_match():
    """When positives == anchors, the loss should be lower than with random positives."""
    torch.manual_seed(0)
    anchors = torch.randn(B, D)
    negatives = torch.randn(B, 5, D)

    loss_match = info_nce_loss(anchors, anchors.clone(), negatives)
    loss_random = info_nce_loss(anchors, torch.randn(B, D), negatives)

    assert loss_match.item() < loss_random.item(), (
        f"Matched loss ({loss_match.item():.4f}) should be less than random "
        f"({loss_random.item():.4f})"
    )


# ---------------------------------------------------------------------------
# 4. test_in_batch_nce_loss_scalar
# ---------------------------------------------------------------------------


def test_in_batch_nce_loss_scalar():
    torch.manual_seed(0)
    a = torch.randn(B, D)
    b = torch.randn(B, D)
    loss = in_batch_nce_loss(a, b)
    assert loss.shape == (), f"Expected scalar, got {loss.shape}"
    assert torch.isfinite(loss).item()


# ---------------------------------------------------------------------------
# 5. test_in_batch_nce_loss_identity — when a == b, loss is near 0
# ---------------------------------------------------------------------------


def test_in_batch_nce_loss_identity():
    """When embeddings_a == embeddings_b, diagonal dominates -> very low loss."""
    torch.manual_seed(0)
    # Use orthogonal vectors so diagonal clearly dominates
    a = torch.eye(B, D)  # each row is a unit vector
    loss = in_batch_nce_loss(a, a.clone(), temperature=0.07)
    assert loss.item() < 0.5, f"Identity loss should be near 0, got {loss.item():.4f}"


# ---------------------------------------------------------------------------
# 6. test_hard_negative_mining_shape
# ---------------------------------------------------------------------------


def test_hard_negative_mining_shape():
    torch.manual_seed(0)
    anchor = torch.randn(D)
    candidates = torch.randn(20, D)
    indices = hard_negative_mining(anchor, candidates, top_k=5)
    assert indices.shape == (5,), f"Expected (5,), got {indices.shape}"


# ---------------------------------------------------------------------------
# 7. test_hard_negative_mining_excludes
# ---------------------------------------------------------------------------


def test_hard_negative_mining_excludes():
    torch.manual_seed(0)
    anchor = torch.randn(D)
    candidates = torch.randn(20, D)
    exclude = 3
    indices = hard_negative_mining(anchor, candidates, top_k=5, exclude_idx=exclude)
    assert exclude not in indices.tolist(), (
        f"Excluded index {exclude} should not appear in result {indices.tolist()}"
    )


# ---------------------------------------------------------------------------
# 8. test_sample_random_negatives_shape
# ---------------------------------------------------------------------------


def test_sample_random_negatives_shape():
    K = 6
    seqs = sample_random_negatives(batch_size=B, n_negatives=K, vocab_size=256, seq_len=T)
    assert seqs.shape == (B, K, T), f"Expected ({B}, {K}, {T}), got {seqs.shape}"
    assert seqs.dtype == torch.long


# ---------------------------------------------------------------------------
# 9. test_sample_random_negatives_vocab_range
# ---------------------------------------------------------------------------


def test_sample_random_negatives_vocab_range():
    vocab_size = 256
    seqs = sample_random_negatives(batch_size=B, n_negatives=5, vocab_size=vocab_size, seq_len=T)
    assert seqs.min().item() >= 0, "Token IDs must be >= 0"
    assert seqs.max().item() < vocab_size, f"Token IDs must be < {vocab_size}"


# ---------------------------------------------------------------------------
# 10. test_embedding_projector_shape
# ---------------------------------------------------------------------------


def test_embedding_projector_shape():
    torch.manual_seed(0)
    proj = EmbeddingProjector(d_model=D, embed_dim=E)
    x = torch.randn(B, D)
    out = proj(x)
    assert out.shape == (B, E), f"Expected ({B}, {E}), got {out.shape}"


# ---------------------------------------------------------------------------
# 11. test_embedding_projector_normalized
# ---------------------------------------------------------------------------


def test_embedding_projector_normalized():
    torch.manual_seed(0)
    proj = EmbeddingProjector(d_model=D, embed_dim=E)
    x = torch.randn(B, D)
    out = proj(x)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(B), atol=1e-5), f"Expected unit norms, got {norms}"


# ---------------------------------------------------------------------------
# 12. test_nce_trainer_get_embeddings_shape
# ---------------------------------------------------------------------------


def test_nce_trainer_get_embeddings_shape(trainer, anchor_ids):
    emb = trainer.get_embeddings(anchor_ids)
    assert emb.shape == (B, E), f"Expected ({B}, {E}), got {emb.shape}"


# ---------------------------------------------------------------------------
# 13. test_nce_trainer_train_step_keys
# ---------------------------------------------------------------------------


def test_nce_trainer_train_step_keys(trainer, anchor_ids, positive_ids):
    result = trainer.train_step_in_batch(anchor_ids, positive_ids)
    assert "loss" in result, "Missing 'loss' key"
    assert "mean_similarity" in result, "Missing 'mean_similarity' key"
    assert "alignment" in result, "Missing 'alignment' key"


# ---------------------------------------------------------------------------
# 14. test_nce_trainer_loss_positive
# ---------------------------------------------------------------------------


def test_nce_trainer_loss_positive(trainer, anchor_ids, positive_ids):
    result = trainer.train_step_in_batch(anchor_ids, positive_ids)
    assert result["loss"] > 0, f"Loss should be positive, got {result['loss']}"


# ---------------------------------------------------------------------------
# 15. test_nce_trainer_evaluate_retrieval_keys
# ---------------------------------------------------------------------------


def test_nce_trainer_evaluate_retrieval_keys(trainer, anchor_ids, positive_ids):
    result = trainer.evaluate_retrieval(anchor_ids, positive_ids, top_k=5)
    assert "recall@1" in result, "Missing 'recall@1'"
    assert "recall@5" in result, "Missing 'recall@5'"
    assert "mrr" in result, "Missing 'mrr'"
