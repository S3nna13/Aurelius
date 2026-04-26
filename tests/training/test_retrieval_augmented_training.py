"""Tests for src/training/retrieval_augmented_training.py — RAT module."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.training.retrieval_augmented_training import (
    CrossAttentionRetriever,
    RATConfig,
    RATLayer,
    RATModel,
    RATTrainer,
    VectorMemoryBank,
)

# ---------------------------------------------------------------------------
# Shared tiny-config constants
# ---------------------------------------------------------------------------

D_MODEL = 16
D_MEM = 16
VOCAB = 16
N_LAYERS = 2
N_HEADS = 2
SEQ_LEN = 4
BATCH = 2
K = 2
CAPACITY = 16


def tiny_config() -> RATConfig:
    return RATConfig(
        d_model=D_MODEL,
        vocab_size=VOCAB,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_mem=D_MEM,
        k_retrieved=K,
        capacity=CAPACITY,
    )


def tiny_model() -> RATModel:
    return RATModel(tiny_config())


def fresh_bank() -> VectorMemoryBank:
    return VectorMemoryBank(capacity=CAPACITY, d_key=D_MODEL, d_val=D_MEM)


def input_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN))


# ===========================================================================
# VectorMemoryBank — add / query / retrieve shapes
# ===========================================================================


def test_memory_bank_add_increases_size():
    bank = fresh_bank()
    keys = torch.randn(4, D_MODEL)
    vals = torch.randn(4, D_MEM)
    bank.add(keys, vals)
    assert bank.size == 4


def test_memory_bank_add_multiple_batches():
    bank = fresh_bank()
    for _ in range(3):
        bank.add(torch.randn(3, D_MODEL), torch.randn(3, D_MEM))
    assert bank.size == 9


def test_memory_bank_query_indices_shape():
    bank = fresh_bank()
    bank.add(torch.randn(8, D_MODEL), torch.randn(8, D_MEM))
    q = torch.randn(BATCH, D_MODEL)
    indices, scores = bank.query(q, K)
    assert indices.shape == (BATCH, K)
    assert scores.shape == (BATCH, K)


def test_memory_bank_retrieve_shape():
    bank = fresh_bank()
    bank.add(torch.randn(8, D_MODEL), torch.randn(8, D_MEM))
    q = torch.randn(BATCH, D_MODEL)
    out = bank.retrieve(q, K)
    assert out.shape == (BATCH, K, D_MEM)


# ===========================================================================
# VectorMemoryBank — circular buffer wrap
# ===========================================================================


def test_memory_bank_circular_buffer_wraps():
    """Filling beyond capacity should not raise and size should cap at capacity."""
    bank = fresh_bank()
    n_extra = CAPACITY + 4
    bank.add(torch.randn(n_extra, D_MODEL), torch.randn(n_extra, D_MEM))
    assert bank.size == CAPACITY


def test_memory_bank_circular_buffer_overwrites():
    """After a full wrap, the oldest entries should be overwritten."""
    bank = VectorMemoryBank(capacity=4, d_key=4, d_val=4)
    # Fill exactly capacity
    keys_first = torch.ones(4, 4)
    vals_first = torch.ones(4, 4)
    bank.add(keys_first, vals_first)
    # Overwrite with zeros
    keys_new = torch.zeros(4, 4)
    vals_new = torch.zeros(4, 4)
    bank.add(keys_new, vals_new)
    # All stored values should now be zeros
    assert bank.values.sum().item() == 0.0


# ===========================================================================
# VectorMemoryBank — cosine similarity top-k ordering
# ===========================================================================


def test_memory_bank_topk_ordering():
    """Top-1 result should be the most similar entry."""
    bank = VectorMemoryBank(capacity=8, d_key=D_MODEL, d_val=D_MEM)
    # Create 8 random key vectors
    keys = torch.randn(8, D_MODEL)
    bank.add(keys, torch.randn(8, D_MEM))

    # Build a query that is identical to key #3
    q = keys[3:4]  # (1, D_MODEL)
    indices, scores = bank.query(q, k=1)
    assert indices[0, 0].item() == 3, f"Top-1 should be index 3, got {indices[0, 0].item()}"


def test_memory_bank_query_scores_descending():
    """Scores returned by query should be in descending order."""
    bank = fresh_bank()
    bank.add(torch.randn(8, D_MODEL), torch.randn(8, D_MEM))
    q = torch.randn(BATCH, D_MODEL)
    _, scores = bank.query(q, K)
    for b in range(BATCH):
        row = scores[b]
        assert (row[:-1] >= row[1:]).all(), f"Scores not descending for batch {b}: {row}"


def test_memory_bank_query_k_equals_size():
    """Querying with k == size should return all entries."""
    bank = VectorMemoryBank(capacity=8, d_key=D_MODEL, d_val=D_MEM)
    bank.add(torch.randn(5, D_MODEL), torch.randn(5, D_MEM))
    q = torch.randn(1, D_MODEL)
    indices, scores = bank.query(q, k=5)
    assert indices.shape == (1, 5)


# ===========================================================================
# CrossAttentionRetriever
# ===========================================================================


def test_cross_attn_retriever_output_shape():
    retriever = CrossAttentionRetriever(d_model=D_MODEL, d_mem=D_MEM, n_heads=N_HEADS)
    q = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    r = torch.randn(BATCH, K, D_MEM)
    out = retriever(q, r)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_cross_attn_retriever_k1():
    """CrossAttentionRetriever should work with a single retrieved document."""
    retriever = CrossAttentionRetriever(d_model=D_MODEL, d_mem=D_MEM, n_heads=N_HEADS)
    q = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    r = torch.randn(BATCH, 1, D_MEM)
    out = retriever(q, r)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


# ===========================================================================
# RATLayer
# ===========================================================================


def test_rat_layer_output_shape():
    bank = fresh_bank()
    bank.add(torch.randn(8, D_MODEL), torch.randn(8, D_MEM))
    layer = RATLayer(d_model=D_MODEL, d_mem=D_MEM, n_heads=N_HEADS, k_retrieved=K)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = layer(x, bank)
    assert out.shape == (BATCH, SEQ_LEN, D_MODEL)


def test_rat_layer_gate_in_0_1():
    layer = RATLayer(d_model=D_MODEL, d_mem=D_MEM, n_heads=N_HEADS, k_retrieved=K)
    g = layer.gate.item()
    assert 0.0 <= g <= 1.0, f"Gate value {g} not in [0, 1]"


# ===========================================================================
# RATModel
# ===========================================================================


def test_rat_model_forward_shape():
    model = tiny_model()
    ids = input_ids()
    logits = model(ids)
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB)


def test_rat_model_encode_to_memory_increases_size():
    model = tiny_model()
    assert model.memory_bank.size == 0
    model.encode_to_memory(input_ids())
    assert model.memory_bank.size == BATCH


def test_rat_model_empty_memory_graceful():
    """Forward pass with an empty memory bank must not raise."""
    model = tiny_model()
    assert model.memory_bank.size == 0
    ids = input_ids()
    logits = model(ids)  # should not raise
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB)


# ===========================================================================
# RATTrainer
# ===========================================================================


def test_rat_trainer_train_step_finite_loss():
    model = tiny_model()
    trainer = RATTrainer(model, lr=1e-3, k_retrieved=K)
    ids = input_ids()
    labels = ids.clone()
    loss = trainer.train_step(ids, labels)
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"


def test_rat_trainer_train_step_scalar():
    model = tiny_model()
    trainer = RATTrainer(model, lr=1e-3, k_retrieved=K)
    ids = input_ids()
    labels = ids.clone()
    loss = trainer.train_step(ids, labels)
    assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"


def test_rat_trainer_train_step_backward_grads():
    """After train_step, model parameters should have gradients."""
    model = tiny_model()
    RATTrainer(model, lr=1e-3, k_retrieved=K)
    ids = input_ids()
    labels = ids.clone()
    # Preserve grad state by running step then checking a param grad
    # (optimizer.step clears grads via zero_grad at start of next step)
    # We inspect right after the backward call by running train_step and then
    # checking that at least one grad was accumulated in the optimizer's param
    # groups. Since train_step calls zero_grad before backward, we run a manual
    # backward check instead.
    model.train()
    model.encode_to_memory(ids)
    logits = model(ids)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1))
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0.0
        for p in model.parameters()
        if p.requires_grad
    )
    assert has_grad, "No parameter received a non-zero gradient"


def test_rat_trainer_populate_memory():
    """populate_memory should fill the bank up to capacity."""
    model = tiny_model()
    trainer = RATTrainer(model, lr=1e-3, k_retrieved=K)
    # Each item adds BATCH=2 entries; CAPACITY=16 → need 8 items
    corpus = [torch.randint(0, VOCAB, (BATCH, SEQ_LEN)) for _ in range(8)]
    trainer.populate_memory(corpus)
    assert model.memory_bank.size == CAPACITY


def test_rat_trainer_populate_memory_1d():
    """populate_memory should accept 1-D tensors (single sequences)."""
    model = tiny_model()
    trainer = RATTrainer(model, lr=1e-3, k_retrieved=K)
    corpus = [torch.randint(0, VOCAB, (SEQ_LEN,)) for _ in range(4)]
    trainer.populate_memory(corpus)
    assert model.memory_bank.size == 4


# ===========================================================================
# RATConfig defaults
# ===========================================================================


def test_rat_config_defaults():
    cfg = RATConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.n_layers == 2
    assert cfg.n_heads == 2
    assert cfg.d_mem == 32
    assert cfg.k_retrieved == 3
    assert cfg.capacity == 128


# ===========================================================================
# retrieve values are non-zero after populate
# ===========================================================================


def test_retrieve_values_non_zero_after_populate():
    """Values retrieved from the bank after populate should not all be zero."""
    model = tiny_model()
    trainer = RATTrainer(model, lr=1e-3, k_retrieved=K)
    corpus = [torch.randint(0, VOCAB, (BATCH, SEQ_LEN)) for _ in range(4)]
    trainer.populate_memory(corpus)

    q = torch.randn(1, D_MODEL)
    retrieved = model.memory_bank.retrieve(q, K)
    assert retrieved.abs().sum().item() > 0.0, "Retrieved values are all zero"
