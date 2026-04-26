"""Tests for src/training/knowledge_graph.py."""

from __future__ import annotations

import pytest
import torch

from src.training.knowledge_graph import (
    EntityEmbeddings,
    KGConfig,
    KGTrainer,
    RelationEmbeddings,
    distmult_score,
    margin_ranking_loss,
    rotate_score,
    sample_negatives,
    transe_score,
)

# ── Shared small config ────────────────────────────────────────────────────────

N_ENTITIES = 20
N_RELATIONS = 5
DIM = 16
B = 4


def small_config(scoring_fn: str = "transe") -> KGConfig:
    return KGConfig(
        n_entities=N_ENTITIES,
        n_relations=N_RELATIONS,
        embedding_dim=DIM,
        scoring_fn=scoring_fn,
        margin=1.0,
        neg_samples=3,
        learning_rate=1e-3,
        regularization=1e-4,
    )


def random_triples(B: int = B) -> torch.Tensor:
    heads = torch.randint(0, N_ENTITIES, (B,))
    rels = torch.randint(0, N_RELATIONS, (B,))
    tails = torch.randint(0, N_ENTITIES, (B,))
    return torch.stack([heads, rels, tails], dim=1)


# ── Test 1: KGConfig defaults ──────────────────────────────────────────────────


def test_kgconfig_defaults():
    cfg = KGConfig()
    assert cfg.n_entities == 100
    assert cfg.n_relations == 10
    assert cfg.embedding_dim == 64
    assert cfg.scoring_fn == "transe"
    assert cfg.margin == 1.0
    assert cfg.neg_samples == 10
    assert cfg.learning_rate == 1e-3
    assert cfg.regularization == 1e-4


# ── Test 2: EntityEmbeddings output is L2-normalized ──────────────────────────


def test_entity_embeddings_normalized():
    emb = EntityEmbeddings(N_ENTITIES, DIM)
    ids = torch.arange(N_ENTITIES)
    out = emb(ids)
    norms = out.norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(N_ENTITIES), atol=1e-5), (
        f"Entity embeddings not unit-norm; got norms: {norms}"
    )


# ── Test 3: RelationEmbeddings output shape ────────────────────────────────────


def test_relation_embeddings_shape():
    emb = RelationEmbeddings(N_RELATIONS, DIM)
    ids = torch.arange(N_RELATIONS)
    out = emb(ids)
    assert out.shape == (N_RELATIONS, DIM)


# ── Test 4: transe_score output shape (B,) ────────────────────────────────────


def test_transe_score_shape():
    h = torch.randn(B, DIM)
    r = torch.randn(B, DIM)
    t = torch.randn(B, DIM)
    scores = transe_score(h, r, t)
    assert scores.shape == (B,), f"Expected ({B},), got {scores.shape}"


# ── Test 5: transe_score correct triple > incorrect ───────────────────────────


def test_transe_score_correct_higher():
    # Construct a case where h + r ≈ t (correct), so score is near 0 (highest possible)
    # vs random t (incorrect), score is much lower
    torch.manual_seed(0)
    h = torch.randn(1, DIM)
    r = torch.randn(1, DIM)
    t_correct = h + r  # h + r - t = 0, score = 0
    t_wrong = torch.randn(1, DIM) * 5.0  # far away

    score_correct = transe_score(h, r, t_correct)
    score_wrong = transe_score(h, r, t_wrong)

    assert score_correct.item() > score_wrong.item(), (
        f"Correct score {score_correct.item():.4f} should be > wrong score {score_wrong.item():.4f}"
    )


# ── Test 6: rotate_score output shape (B,) ────────────────────────────────────


def test_rotate_score_shape():
    h = torch.randn(B, DIM)
    r = torch.randn(B, DIM)
    t = torch.randn(B, DIM)
    scores = rotate_score(h, r, t)
    assert scores.shape == (B,), f"Expected ({B},), got {scores.shape}"


# ── Test 7: distmult_score output shape (B,) ──────────────────────────────────


def test_distmult_score_shape():
    h = torch.randn(B, DIM)
    r = torch.randn(B, DIM)
    t = torch.randn(B, DIM)
    scores = distmult_score(h, r, t)
    assert scores.shape == (B,), f"Expected ({B},), got {scores.shape}"


# ── Test 8: margin_ranking_loss is zero when pos >> neg ───────────────────────


def test_margin_ranking_loss_zero():
    # When pos_scores >> neg_scores + margin, loss = 0
    pos = torch.tensor([10.0, 10.0, 10.0])
    neg = torch.tensor([-10.0, -10.0, -10.0])
    loss = margin_ranking_loss(pos, neg, margin=1.0)
    assert loss.item() == pytest.approx(0.0, abs=1e-6), f"Expected 0 loss, got {loss.item()}"


# ── Test 9: margin_ranking_loss > 0 when neg > pos ───────────────────────────


def test_margin_ranking_loss_positive():
    pos = torch.tensor([-1.0, -1.0])
    neg = torch.tensor([1.0, 1.0])
    loss = margin_ranking_loss(pos, neg, margin=1.0)
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"


# ── Test 10: sample_negatives returns correct count ───────────────────────────


def test_sample_negatives_count():
    n_neg = 5
    batch = random_triples(B)
    negs = sample_negatives(batch, N_ENTITIES, n_neg)
    assert negs.shape == (B * n_neg, 3), f"Expected ({B * n_neg}, 3), got {negs.shape}"


# ── Test 11: sample_negatives produces different triples from positives ────────


def test_sample_negatives_different():
    torch.manual_seed(42)
    # Use triples with specific known values so we can check corruption
    batch = torch.tensor([[0, 0, 1], [2, 1, 3], [4, 2, 5], [6, 3, 7]])
    n_neg = 10
    negs = sample_negatives(batch, N_ENTITIES, n_neg)

    # Check that relation ids are preserved (only heads/tails get corrupted)
    batch_rel_expanded = batch[:, 1].repeat_interleave(n_neg)
    assert torch.all(negs[:, 1] == batch_rel_expanded), (
        "Relation ids should not be corrupted in negative sampling"
    )

    # Check that at least some head or tail differs from the original
    pos_repeated = batch.repeat_interleave(n_neg, dim=0)
    head_same = negs[:, 0] == pos_repeated[:, 0]
    tail_same = negs[:, 2] == pos_repeated[:, 2]
    # At least some negatives must differ in head or tail
    # (with 10 negatives and 20 entities this is essentially guaranteed)
    assert not torch.all(head_same & tail_same), (
        "All negatives are identical to positives — corruption failed"
    )


# ── Test 12: KGTrainer.train_step returns correct keys ───────────────────────


def test_trainer_train_step_keys():
    cfg = small_config("transe")
    trainer = KGTrainer(cfg)
    triples = random_triples(B)
    result = trainer.train_step(triples)
    assert set(result.keys()) == {"loss", "mean_pos_score", "mean_neg_score"}, (
        f"Unexpected keys: {result.keys()}"
    )
    assert isinstance(result["loss"], float)
    assert isinstance(result["mean_pos_score"], float)
    assert isinstance(result["mean_neg_score"], float)


# ── Test 13: KGTrainer.predict_tail returns top_k entity ids ─────────────────


def test_trainer_predict_tail():
    cfg = small_config("transe")
    trainer = KGTrainer(cfg)
    top_k = 5
    result = trainer.predict_tail(head_id=0, rel_id=0, top_k=top_k)
    assert result.shape == (top_k,), f"Expected ({top_k},), got {result.shape}"
    # All ids should be valid entity indices
    assert (result >= 0).all() and (result < N_ENTITIES).all(), (
        f"Entity ids out of range [0, {N_ENTITIES}): {result}"
    )
    # Should not contain duplicates
    assert result.unique().shape[0] == top_k, "predict_tail returned duplicate entity ids"


# ── Test 14: KGTrainer.score_triples returns (B,) tensor ─────────────────────


def test_trainer_score_triples_shape():
    for fn in ["transe", "rotate", "distmult"]:
        cfg = small_config(fn)
        trainer = KGTrainer(cfg)
        triples = random_triples(B)
        scores = trainer.score_triples(triples)
        assert scores.shape == (B,), f"[{fn}] Expected ({B},), got {scores.shape}"
