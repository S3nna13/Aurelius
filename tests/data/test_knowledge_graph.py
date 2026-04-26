"""Tests for knowledge_graph module."""

from __future__ import annotations

import random

import pytest
import torch

from src.data.knowledge_graph import (
    KGConfig,
    KGRetriever,
    KnowledgeGraph,
    TransEEmbedder,
    Triple,
    build_entity_index,
    build_relation_index,
    format_triples_as_text,
    retrieve_by_entity,
    transe_margin_loss,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_simple_kg() -> KnowledgeGraph:
    """Return a small KG with 3 triples for reuse in tests."""
    kg = KnowledgeGraph()
    kg.add_triple("Alice", "knows", "Bob")
    kg.add_triple("Bob", "likes", "Python")
    kg.add_triple("Alice", "uses", "Python")
    return kg


# ---------------------------------------------------------------------------
# 1. test_kg_config_defaults
# ---------------------------------------------------------------------------


def test_kg_config_defaults():
    cfg = KGConfig()
    assert cfg.embed_dim == 64
    assert cfg.n_relations == 8
    assert cfg.margin == 1.0
    assert cfg.norm == 2
    assert cfg.negative_k == 4


# ---------------------------------------------------------------------------
# 2. test_kg_add_and_len
# ---------------------------------------------------------------------------


def test_kg_add_and_len():
    kg = make_simple_kg()
    assert len(kg) == 3


# ---------------------------------------------------------------------------
# 3. test_kg_get_neighbors
# ---------------------------------------------------------------------------


def test_kg_get_neighbors():
    kg = make_simple_kg()
    neighbors = kg.get_neighbors("Alice")
    # Alice has two outgoing edges
    assert ("knows", "Bob") in neighbors
    assert ("uses", "Python") in neighbors
    assert len(neighbors) == 2

    bob_neighbors = kg.get_neighbors("Bob")
    assert ("likes", "Python") in bob_neighbors


# ---------------------------------------------------------------------------
# 4. test_kg_get_entities_unique
# ---------------------------------------------------------------------------


def test_kg_get_entities_unique():
    kg = make_simple_kg()
    entities = kg.get_entities()
    assert len(entities) == len(set(entities)), "Entities list contains duplicates"
    assert set(entities) == {"Alice", "Bob", "Python"}


# ---------------------------------------------------------------------------
# 5. test_kg_get_relations_unique
# ---------------------------------------------------------------------------


def test_kg_get_relations_unique():
    kg = make_simple_kg()
    relations = kg.get_relations()
    assert len(relations) == len(set(relations)), "Relations list contains duplicates"
    assert set(relations) == {"knows", "likes", "uses"}


# ---------------------------------------------------------------------------
# 6. test_kg_sample_negative_different
# ---------------------------------------------------------------------------


def test_kg_sample_negative_different():
    kg = make_simple_kg()
    rng = random.Random(42)
    original = Triple(head="Alice", relation="knows", tail="Bob")

    # Run many times to get variation
    for _ in range(50):
        neg = kg.sample_negative(original, rng)
        # relation should be preserved
        assert neg.relation == original.relation
        # head or tail should differ in at least some samples
        if neg.head != original.head or neg.tail != original.tail:
            return  # Found a differing negative

    pytest.fail("sample_negative never returned a triple different from original")


# ---------------------------------------------------------------------------
# 7. test_build_entity_index_contiguous
# ---------------------------------------------------------------------------


def test_build_entity_index_contiguous():
    kg = make_simple_kg()
    index = build_entity_index(kg)
    n = len(index)
    assert set(index.values()) == set(range(n)), "Entity indices are not contiguous 0..n-1"


# ---------------------------------------------------------------------------
# 8. test_build_relation_index_deterministic
# ---------------------------------------------------------------------------


def test_build_relation_index_deterministic():
    kg = make_simple_kg()
    idx1 = build_relation_index(kg)
    idx2 = build_relation_index(kg)
    assert idx1 == idx2, "build_relation_index is not deterministic"


# ---------------------------------------------------------------------------
# 9. test_transe_embedder_score_shape
# ---------------------------------------------------------------------------


def test_transe_embedder_score_shape():
    n_entities = 10
    n_relations = 5
    embed_dim = 16
    B = 4

    model = TransEEmbedder(n_entities, n_relations, embed_dim)
    h = torch.randint(0, n_entities, (B,))
    r = torch.randint(0, n_relations, (B,))
    t = torch.randint(0, n_entities, (B,))

    scores = model.score(h, r, t)
    assert scores.shape == (B,), f"Expected shape ({B},), got {scores.shape}"


# ---------------------------------------------------------------------------
# 10. test_transe_margin_loss_positive_case
# ---------------------------------------------------------------------------


def test_transe_margin_loss_positive_case():
    # pos_score >> neg_score → loss ~ 0
    pos_scores = torch.tensor([5.0, 5.0, 5.0])
    neg_scores = torch.tensor([-5.0, -5.0, -5.0])
    margin = 1.0
    loss = transe_margin_loss(pos_scores, neg_scores, margin)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 11. test_transe_margin_loss_negative_case
# ---------------------------------------------------------------------------


def test_transe_margin_loss_negative_case():
    # neg_score >> pos_score → loss > 0
    pos_scores = torch.tensor([-5.0, -5.0, -5.0])
    neg_scores = torch.tensor([5.0, 5.0, 5.0])
    margin = 1.0
    loss = transe_margin_loss(pos_scores, neg_scores, margin)
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# 12. test_retrieve_by_entity_depth1
# ---------------------------------------------------------------------------


def test_retrieve_by_entity_depth1():
    kg = make_simple_kg()
    triples = retrieve_by_entity(kg, "Alice", max_hops=1)
    # At depth 1, only direct neighbors of Alice
    heads = {t.head for t in triples}
    assert heads == {"Alice"}, f"Expected only Alice's direct triples, got heads: {heads}"
    assert len(triples) == 2  # Alice->Bob, Alice->Python


# ---------------------------------------------------------------------------
# 13. test_retrieve_by_entity_depth2
# ---------------------------------------------------------------------------


def test_retrieve_by_entity_depth2():
    kg = make_simple_kg()
    triples = retrieve_by_entity(kg, "Alice", max_hops=2)
    # At depth 2, should also include Bob's neighbors
    heads = {t.head for t in triples}
    assert "Bob" in heads, "2-hop retrieval should include Bob's outgoing triples"
    # Should have Alice's 2 triples + Bob's 1 triple
    assert len(triples) == 3


# ---------------------------------------------------------------------------
# 14. test_format_triples_as_text
# ---------------------------------------------------------------------------


def test_format_triples_as_text():
    triples = [Triple(head="Alice", relation="knows", tail="Bob")]
    text = format_triples_as_text(triples)
    assert "Alice knows Bob" in text


# ---------------------------------------------------------------------------
# 15. test_kg_retriever_retrieve_count
# ---------------------------------------------------------------------------


def test_kg_retriever_retrieve_count():
    kg = KnowledgeGraph()
    for i in range(20):
        kg.add_triple(f"entity{i}", "rel", f"entity{i + 1}")

    retriever = KGRetriever(kg, max_hops=2)
    # Query mentioning entity0 — should return at most top_k
    results = retriever.retrieve("entity0", top_k=3)
    assert len(results) <= 3


# ---------------------------------------------------------------------------
# 16. test_kg_retriever_format_context
# ---------------------------------------------------------------------------


def test_kg_retriever_format_context():
    kg = make_simple_kg()
    retriever = KGRetriever(kg, max_hops=1)
    context = retriever.format_context("Tell me about Alice", top_k=5)
    assert len(context) > 0, "format_context should return non-empty string when entity is in query"
    assert "Alice" in context
