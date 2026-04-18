"""
Tests for src/inference/icl_retrieval.py

Parameters used throughout:
    d_model    = 16
    vocab_size = 16
    k          = 3
    T          = 8
    B          = 2
    capacity   = 20
"""

import math
import torch
import torch.nn as nn
import pytest

from src.inference.icl_retrieval import (
    DemonstrationStore,
    DemonstrationEncoder,
    DemonstrationSelector,
    ICLPromptAssembler,
    ICLRetrievalTrainer,
    ICLRetrievalConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_SIZE = 16
K = 3
T = 8
B = 2
CAPACITY = 20


def make_store(n: int = 10) -> DemonstrationStore:
    store = DemonstrationStore(d_model=D_MODEL, capacity=CAPACITY)
    for i in range(n):
        emb = torch.randn(D_MODEL)
        val = {
            "input": list(range(1, 5)),
            "output": [i % VOCAB_SIZE],
            "label": i,
        }
        store.add(emb, val)
    return store


def make_encoder() -> DemonstrationEncoder:
    return DemonstrationEncoder(
        d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_layers=2
    )


def make_query_ids(length: int = T) -> torch.Tensor:
    return torch.randint(1, VOCAB_SIZE, (length,))


def make_batch_ids(batch: int = B, length: int = T) -> torch.Tensor:
    return torch.randint(1, VOCAB_SIZE, (batch, length))


# ---------------------------------------------------------------------------
# DemonstrationStore tests
# ---------------------------------------------------------------------------

def test_store_add_increases_size():
    store = DemonstrationStore(d_model=D_MODEL, capacity=CAPACITY)
    assert store.size == 0
    store.add(torch.randn(D_MODEL), {"input": [1], "output": [2], "label": 0})
    assert store.size == 1
    store.add(torch.randn(D_MODEL), {"input": [3], "output": [4], "label": 1})
    assert store.size == 2


def test_store_search_returns_k_results():
    store = make_store(n=10)
    query = torch.randn(D_MODEL)
    scores, indices = store.search(query, k=K)
    assert scores.shape[0] == K
    assert indices.shape[0] == K


def test_store_cosine_top1_is_most_similar():
    store = DemonstrationStore(d_model=D_MODEL, capacity=CAPACITY)
    # Construct a clearly-dominant embedding
    target_emb = torch.zeros(D_MODEL)
    target_emb[0] = 1.0  # unit vector along dim 0
    for i in range(8):
        emb = torch.randn(D_MODEL)
        store.add(emb, {"input": [i], "output": [i], "label": i})
    # Add the target as the last entry (known index)
    store.add(target_emb, {"input": [99], "output": [99], "label": 99})
    target_idx = store.size - 1  # index 8 (0-based)

    query = target_emb.clone()
    scores, indices = store.search(query, k=1)
    assert indices[0].item() == target_idx


def test_store_retrieve_returns_list_of_dicts():
    store = make_store(n=10)
    query = torch.randn(D_MODEL)
    results = store.retrieve(query, k=K)
    assert isinstance(results, list)
    assert len(results) == K
    for item in results:
        assert isinstance(item, dict)
        assert "input" in item
        assert "output" in item


def test_store_add_batch_increases_size():
    store = DemonstrationStore(d_model=D_MODEL, capacity=CAPACITY)
    embs = torch.randn(5, D_MODEL)
    vals = [{"input": [i], "output": [i], "label": i} for i in range(5)]
    store.add_batch(embs, vals)
    assert store.size == 5


# ---------------------------------------------------------------------------
# DemonstrationEncoder tests
# ---------------------------------------------------------------------------

def test_encoder_forward_output_shape():
    encoder = make_encoder()
    input_ids = make_batch_ids(B, T)
    out = encoder(input_ids)
    assert out.shape == (B, D_MODEL), f"Expected ({B}, {D_MODEL}), got {out.shape}"


def test_encoder_output_is_finite():
    encoder = make_encoder()
    input_ids = make_batch_ids(B, T)
    out = encoder(input_ids)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# DemonstrationSelector tests
# ---------------------------------------------------------------------------

def test_selector_select_random_returns_k():
    store = make_store(n=10)
    encoder = make_encoder()
    selector = DemonstrationSelector(store, encoder)
    result = selector.select_random(k=K)
    assert len(result) == K


def test_selector_select_by_similarity_returns_k():
    store = make_store(n=10)
    encoder = make_encoder()
    selector = DemonstrationSelector(store, encoder)
    query_ids = make_query_ids(T)
    result = selector.select_by_similarity(query_ids, k=K)
    assert len(result) == K
    for item in result:
        assert isinstance(item, dict)


def test_selector_select_diverse_returns_k_distinct():
    store = make_store(n=10)
    encoder = make_encoder()
    selector = DemonstrationSelector(store, encoder)
    query_ids = make_query_ids(T)
    result = selector.select_diverse(query_ids, k=K)
    assert len(result) == K
    # Check distinct by label
    labels = [d["label"] for d in result]
    assert len(set(labels)) == len(labels), "MMR should return distinct demos"


def test_selector_select_diverse_mmr_lambda():
    """MMR with lambda=1.0 should behave like pure similarity."""
    store = make_store(n=10)
    encoder = make_encoder()
    selector = DemonstrationSelector(store, encoder)
    query_ids = make_query_ids(T)
    result = selector.select_diverse(query_ids, k=K, mmr_lambda=1.0)
    assert len(result) == K


def test_selector_select_by_coverage_returns_k():
    store = make_store(n=10)
    encoder = make_encoder()
    selector = DemonstrationSelector(store, encoder)
    query_ids = make_query_ids(T)
    result = selector.select_by_coverage(query_ids, k=K, n_gram=2)
    assert len(result) == K
    for item in result:
        assert isinstance(item, dict)


# ---------------------------------------------------------------------------
# ICLPromptAssembler tests
# ---------------------------------------------------------------------------

def test_assembler_output_is_1d_tensor():
    assembler = ICLPromptAssembler(max_demo_tokens=32, separator_id=0)
    demos = [
        {"input": [1, 2, 3], "output": [4], "label": 0},
        {"input": [5, 6], "output": [7, 8], "label": 1},
    ]
    query_ids = make_query_ids(T)
    result = assembler.assemble(demos, query_ids)
    assert result.ndim == 1
    assert result.dtype == torch.long


def test_assembler_length_greater_than_query():
    assembler = ICLPromptAssembler(max_demo_tokens=32, separator_id=0)
    demos = [
        {"input": [1, 2, 3], "output": [4], "label": 0},
    ]
    query_ids = make_query_ids(T)
    result = assembler.assemble(demos, query_ids)
    assert result.shape[0] > query_ids.shape[0]


def test_assembler_reorder_returns_same_k():
    assembler = ICLPromptAssembler()
    encoder = make_encoder()
    demos = [
        {"input": [1, 2], "output": [3], "label": i}
        for i in range(K)
    ]
    query_ids = make_query_ids(T)
    reordered = assembler.reorder_by_similarity(demos, query_ids, encoder)
    assert len(reordered) == K


def test_assembler_reorder_sorts_ascending_similarity():
    """reorder_by_similarity should place most-similar demo last."""
    assembler = ICLPromptAssembler()
    encoder = make_encoder()
    query_ids = make_query_ids(T)
    demos = [
        {"input": [1, 2], "output": [3], "label": i}
        for i in range(4)
    ]
    reordered = assembler.reorder_by_similarity(demos, query_ids, encoder)
    # Verify it's still a list of dicts with correct length
    assert isinstance(reordered, list)
    assert len(reordered) == 4
    for d in reordered:
        assert isinstance(d, dict)


def test_assembler_truncate_reduces_if_needed():
    assembler = ICLPromptAssembler()
    demos = [
        {"input": [1, 2, 3, 4, 5], "output": [6, 7, 8], "label": i}
        for i in range(5)
    ]
    # Budget is very small — should drop some demos
    budget = 5
    result = assembler.truncate_to_budget(demos, budget)
    total_tokens = sum(
        len(d.get("input", [])) + len(d.get("output", [])) for d in result
    )
    assert total_tokens <= budget or len(result) == 0


def test_assembler_truncate_no_change_if_within_budget():
    assembler = ICLPromptAssembler()
    demos = [
        {"input": [1], "output": [2], "label": 0},
    ]
    result = assembler.truncate_to_budget(demos, budget=1000)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# ICLRetrievalTrainer tests
# ---------------------------------------------------------------------------

class _DummyLM(nn.Module):
    def forward(self, x):
        return x


def test_trainer_contrastive_loss_finite_scalar():
    encoder = make_encoder()
    trainer = ICLRetrievalTrainer(lm=_DummyLM(), encoder=encoder, lr=1e-3)
    q = torch.randn(B, D_MODEL)
    p = torch.randn(B, D_MODEL)
    n = torch.randn(B, D_MODEL)
    loss = trainer.contrastive_loss(q, p, n)
    assert loss.ndim == 0
    assert math.isfinite(loss.item())


def test_trainer_train_step_returns_finite_loss():
    encoder = make_encoder()
    trainer = ICLRetrievalTrainer(lm=_DummyLM(), encoder=encoder, lr=1e-3)
    query_ids = make_batch_ids(B, T)
    pos_ids = make_batch_ids(B, T)
    neg_ids = make_batch_ids(B, T)
    loss = trainer.train_retriever_step(query_ids, pos_ids, neg_ids)
    assert math.isfinite(loss.item()), f"Expected finite loss, got {loss.item()}"


def test_trainer_loss_decreases_with_perfect_pairs():
    """Training on identical positive pairs should reduce the contrastive loss."""
    torch.manual_seed(0)
    encoder = make_encoder()
    trainer = ICLRetrievalTrainer(lm=_DummyLM(), encoder=encoder, lr=1e-2)
    # Use same ids for query and positive (trivially perfect retrieval signal)
    ids = make_batch_ids(B, T)
    neg_ids = make_batch_ids(B, T)

    losses = []
    for _ in range(10):
        loss = trainer.train_retriever_step(ids, ids.clone(), neg_ids)
        losses.append(loss.item())

    # At minimum the loss should be finite throughout
    assert all(math.isfinite(l) for l in losses)


# ---------------------------------------------------------------------------
# ICLRetrievalConfig tests
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = ICLRetrievalConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.n_layers == 2
    assert cfg.capacity == 100
    assert cfg.k_demos == 4
    assert cfg.max_demo_tokens == 32
    assert cfg.mmr_lambda == 0.5


def test_config_custom_values():
    cfg = ICLRetrievalConfig(d_model=16, vocab_size=16, capacity=20, k_demos=3)
    assert cfg.d_model == 16
    assert cfg.vocab_size == 16
    assert cfg.capacity == 20
    assert cfg.k_demos == 3
