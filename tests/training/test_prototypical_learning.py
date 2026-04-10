"""Tests for prototypical_learning module — prototype-based few-shot learning."""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.prototypical_learning import (
    ProtoConfig,
    PrototypicalClassifier,
    ProtoNetTrainer,
    compute_prototypes,
    extract_embeddings,
    prototypical_loss,
)

# ---------------------------------------------------------------------------
# Shared constants (fast test dimensions)
# ---------------------------------------------------------------------------

VOCAB = 256
SEQ_LEN = 4
D_MODEL = 64
N_WAY = 2
N_SUPPORT = 2
N_QUERY = 3


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=D_MODEL,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def model(small_cfg: AureliusConfig) -> AureliusTransformer:
    torch.manual_seed(42)
    return AureliusTransformer(small_cfg)


@pytest.fixture(scope="module")
def proto_cfg() -> ProtoConfig:
    return ProtoConfig(
        n_way=N_WAY,
        n_support=N_SUPPORT,
        n_query=N_QUERY,
        embedding_dim=D_MODEL,
        distance="euclidean",
    )


def _rand_ids(batch: int = 1, seed: int = 0) -> Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, VOCAB, (batch, SEQ_LEN))


def _make_support(n_way: int = N_WAY, n_support: int = N_SUPPORT):
    """Return (support_ids list, support_labels list) for n_way * n_support examples."""
    ids_list: list[Tensor] = []
    labels: list[int] = []
    for cls in range(n_way):
        for i in range(n_support):
            ids_list.append(_rand_ids(batch=1, seed=cls * 100 + i))
            labels.append(cls)
    return ids_list, labels


def _make_query(n_way: int = N_WAY, n_query: int = N_QUERY):
    """Return (query_ids list, query_labels list)."""
    ids_list: list[Tensor] = []
    labels: list[int] = []
    for cls in range(n_way):
        for i in range(n_query):
            ids_list.append(_rand_ids(batch=1, seed=cls * 200 + i + 50))
            labels.append(cls)
    return ids_list, labels


# ---------------------------------------------------------------------------
# 1. ProtoConfig defaults
# ---------------------------------------------------------------------------

def test_proto_config_defaults():
    cfg = ProtoConfig()
    assert cfg.n_way == 5
    assert cfg.n_support == 5
    assert cfg.n_query == 10
    assert cfg.embedding_dim == 64
    assert cfg.distance == "euclidean"


# ---------------------------------------------------------------------------
# 2. extract_embeddings returns shape (B, d_model)
# ---------------------------------------------------------------------------

def test_extract_embeddings_shape(model):
    B = 3
    ids = _rand_ids(batch=B, seed=1)
    embs = extract_embeddings(model, ids)
    assert embs.shape == (B, D_MODEL), f"Expected ({B}, {D_MODEL}), got {embs.shape}"


# ---------------------------------------------------------------------------
# 3. extract_embeddings L2-normalised (norm ≈ 1)
# ---------------------------------------------------------------------------

def test_extract_embeddings_l2_normalised(model):
    ids = _rand_ids(batch=4, seed=2)
    embs = extract_embeddings(model, ids)
    norms = embs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
        f"Embeddings not L2-normalised; norms: {norms}"
    )


# ---------------------------------------------------------------------------
# 4. compute_prototypes shape (n_classes, D)
# ---------------------------------------------------------------------------

def test_compute_prototypes_shape():
    N, D, C = 6, 32, 3
    embs = torch.randn(N, D)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    protos = compute_prototypes(embs, labels, C)
    assert protos.shape == (C, D), f"Expected ({C}, {D}), got {protos.shape}"


# ---------------------------------------------------------------------------
# 5. compute_prototypes: 1 sample per class == that sample
# ---------------------------------------------------------------------------

def test_compute_prototypes_single_sample_per_class():
    D, C = 16, 3
    embs = torch.randn(C, D)
    labels = torch.arange(C)
    protos = compute_prototypes(embs, labels, C)
    assert torch.allclose(protos, embs, atol=1e-6), (
        "Prototype of a single sample should equal the sample itself"
    )


# ---------------------------------------------------------------------------
# 6. prototypical_loss returns (scalar Tensor, float)
# ---------------------------------------------------------------------------

def test_prototypical_loss_return_types():
    Q, D, C = 6, 32, 3
    q_embs = F_normalize(torch.randn(Q, D))
    protos = F_normalize(torch.randn(C, D))
    q_labels = torch.tensor([0, 1, 2, 0, 1, 2])
    loss, acc = prototypical_loss(q_embs, protos, q_labels)
    assert isinstance(loss, Tensor), "loss must be a Tensor"
    assert loss.ndim == 0, f"loss must be scalar, got shape {loss.shape}"
    assert isinstance(acc, float), f"accuracy must be float, got {type(acc)}"


# ---------------------------------------------------------------------------
# 7. prototypical_loss accuracy in [0, 1]
# ---------------------------------------------------------------------------

def test_prototypical_loss_accuracy_range():
    Q, D, C = 6, 32, 3
    q_embs = F_normalize(torch.randn(Q, D))
    protos = F_normalize(torch.randn(C, D))
    q_labels = torch.tensor([0, 1, 2, 0, 1, 2])
    _, acc = prototypical_loss(q_embs, protos, q_labels)
    assert 0.0 <= acc <= 1.0, f"Accuracy out of range: {acc}"


# ---------------------------------------------------------------------------
# 8. prototypical_loss perfect alignment → high accuracy
# ---------------------------------------------------------------------------

def test_prototypical_loss_perfect_alignment():
    """When query embeddings equal their class prototype, accuracy should be 1."""
    C, D = 3, 32
    protos = F_normalize(torch.randn(C, D))
    # Queries = exact copies of prototypes (already L2-normalised)
    q_embs = protos.clone()
    q_labels = torch.arange(C)
    _, acc = prototypical_loss(q_embs, protos, q_labels)
    assert acc == pytest.approx(1.0), f"Perfect alignment should give acc=1.0, got {acc}"


# ---------------------------------------------------------------------------
# 9. PrototypicalClassifier.fit stores prototypes shape (n_way, D)
# ---------------------------------------------------------------------------

def test_classifier_fit_stores_prototypes(model, proto_cfg):
    clf = PrototypicalClassifier(model, proto_cfg)
    support_ids, support_labels = _make_support()
    clf.fit(support_ids, support_labels)
    assert clf.prototypes is not None
    assert clf.prototypes.shape == (N_WAY, D_MODEL), (
        f"Expected ({N_WAY}, {D_MODEL}), got {clf.prototypes.shape}"
    )


# ---------------------------------------------------------------------------
# 10. PrototypicalClassifier.predict returns (B,) ints
# ---------------------------------------------------------------------------

def test_classifier_predict_returns_int_tensor(model, proto_cfg):
    clf = PrototypicalClassifier(model, proto_cfg)
    support_ids, support_labels = _make_support()
    clf.fit(support_ids, support_labels)

    B = 4
    query = _rand_ids(batch=B, seed=99)
    preds = clf.predict(query)
    assert preds.shape == (B,), f"Expected shape ({B},), got {preds.shape}"
    assert preds.dtype in (torch.int64, torch.long), f"Expected long tensor, got {preds.dtype}"


# ---------------------------------------------------------------------------
# 11. PrototypicalClassifier.predict_proba sums to 1 per query
# ---------------------------------------------------------------------------

def test_classifier_predict_proba_sums_to_one(model, proto_cfg):
    clf = PrototypicalClassifier(model, proto_cfg)
    support_ids, support_labels = _make_support()
    clf.fit(support_ids, support_labels)

    B = 5
    query = _rand_ids(batch=B, seed=77)
    proba = clf.predict_proba(query)
    assert proba.shape == (B, N_WAY), f"Expected ({B}, {N_WAY}), got {proba.shape}"
    row_sums = proba.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-5), (
        f"Row probabilities don't sum to 1: {row_sums}"
    )


# ---------------------------------------------------------------------------
# 12. ProtoNetTrainer.train_episode returns required keys
# ---------------------------------------------------------------------------

def test_trainer_train_episode_keys(model, proto_cfg):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = ProtoNetTrainer(model, proto_cfg, opt)
    support_ids, support_labels = _make_support()
    query_ids, query_labels = _make_query()
    result = trainer.train_episode(support_ids, support_labels, query_ids, query_labels)
    assert "loss" in result, "result must contain 'loss'"
    assert "accuracy" in result, "result must contain 'accuracy'"


# ---------------------------------------------------------------------------
# 13. train_episode accuracy in [0, 1]
# ---------------------------------------------------------------------------

def test_trainer_train_episode_accuracy_range(model, proto_cfg):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = ProtoNetTrainer(model, proto_cfg, opt)
    support_ids, support_labels = _make_support()
    query_ids, query_labels = _make_query()
    result = trainer.train_episode(support_ids, support_labels, query_ids, query_labels)
    acc = result["accuracy"]
    assert 0.0 <= acc <= 1.0, f"Accuracy out of [0, 1]: {acc}"


# ---------------------------------------------------------------------------
# 14. Cosine distance option works without error
# ---------------------------------------------------------------------------

def test_cosine_distance_option(model, small_cfg):
    cosine_cfg = ProtoConfig(
        n_way=N_WAY,
        n_support=N_SUPPORT,
        n_query=N_QUERY,
        embedding_dim=D_MODEL,
        distance="cosine",
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = ProtoNetTrainer(model, cosine_cfg, opt)
    support_ids, support_labels = _make_support()
    query_ids, query_labels = _make_query()
    result = trainer.train_episode(support_ids, support_labels, query_ids, query_labels)
    assert "loss" in result
    assert "accuracy" in result
    assert torch.isfinite(torch.tensor(result["loss"])), (
        f"Cosine distance produced non-finite loss: {result['loss']}"
    )


# ---------------------------------------------------------------------------
# Helper: thin wrapper so tests read clearly
# ---------------------------------------------------------------------------

def F_normalize(t: Tensor) -> Tensor:
    import torch.nn.functional as F
    return F.normalize(t, p=2, dim=-1)
