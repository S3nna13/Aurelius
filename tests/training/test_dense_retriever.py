"""Tests for src/training/dense_retriever.py — DPR-style dense retriever training."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.dense_retriever import (
    DPRTrainer,
    DenseEncoder,
    RetrieverConfig,
    in_batch_negatives_loss,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

EMB_DIM = 32
BATCH = 2
SEQ_LEN = 4


def make_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


def make_backbone() -> AureliusTransformer:
    return AureliusTransformer(make_cfg())


def make_encoder(pooling: str = "cls") -> DenseEncoder:
    cfg = RetrieverConfig(embedding_dim=EMB_DIM, pooling=pooling, n_negatives=1)
    return DenseEncoder(make_backbone(), cfg)


def make_input() -> torch.Tensor:
    return torch.randint(0, 256, (BATCH, SEQ_LEN))


# ---------------------------------------------------------------------------
# RetrieverConfig tests
# ---------------------------------------------------------------------------


def test_retriever_config_defaults():
    cfg = RetrieverConfig()
    assert cfg.embedding_dim == 64
    assert cfg.temperature == 0.07
    assert cfg.n_negatives == 4
    assert cfg.pooling == "cls"


# ---------------------------------------------------------------------------
# DenseEncoder tests
# ---------------------------------------------------------------------------


def test_dense_encoder_encode_shape():
    enc = make_encoder()
    ids = make_input()
    out = enc.encode(ids)
    assert out.shape == (BATCH, EMB_DIM), f"Expected ({BATCH}, {EMB_DIM}), got {out.shape}"


def test_dense_encoder_encode_l2_normalized():
    enc = make_encoder()
    ids = make_input()
    out = enc.encode(ids)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5), f"Norms not ~1: {norms}"


def test_dense_encoder_cls_pooling_first_token():
    """With cls pooling, output should be determined by first token."""
    enc = make_encoder(pooling="cls")
    ids = make_input()

    captured = {}

    def hook(module, inp, out):
        hidden = out[0] if isinstance(out, (tuple, list)) else out
        captured["hidden"] = hidden.detach()

    handle = enc.backbone.layers[-1].register_forward_hook(hook)
    try:
        emb = enc.encode(ids)
    finally:
        handle.remove()

    hidden = captured["hidden"]  # (B, S, d_model)
    cls_tokens = hidden[:, 0, :]
    # Project and normalize manually to verify
    with torch.no_grad():
        expected = torch.nn.functional.normalize(enc.proj(cls_tokens), dim=-1)
    assert torch.allclose(emb, expected, atol=1e-5)


def test_dense_encoder_mean_pooling():
    """With mean pooling, output should be the mean over all positions."""
    enc = make_encoder(pooling="mean")
    ids = make_input()

    captured = {}

    def hook(module, inp, out):
        hidden = out[0] if isinstance(out, (tuple, list)) else out
        captured["hidden"] = hidden.detach()

    handle = enc.backbone.layers[-1].register_forward_hook(hook)
    try:
        emb = enc.encode(ids)
    finally:
        handle.remove()

    hidden = captured["hidden"]  # (B, S, d_model)
    mean_tokens = hidden.mean(dim=1)
    with torch.no_grad():
        expected = torch.nn.functional.normalize(enc.proj(mean_tokens), dim=-1)
    assert torch.allclose(emb, expected, atol=1e-5)


def test_dense_encoder_differentiable():
    """Loss computed from encoder output should be differentiable."""
    enc = make_encoder()
    ids = make_input()
    out = enc.encode(ids)
    loss = out.sum()
    loss.backward()
    # If no exception, gradients flowed
    assert enc.proj.weight.grad is not None


# ---------------------------------------------------------------------------
# in_batch_negatives_loss tests
# ---------------------------------------------------------------------------


def test_in_batch_negatives_loss_returns_scalar():
    q = torch.randn(BATCH, EMB_DIM)
    p = torch.randn(BATCH, EMB_DIM)
    q = torch.nn.functional.normalize(q, dim=-1)
    p = torch.nn.functional.normalize(p, dim=-1)
    loss = in_batch_negatives_loss(q, p, temperature=0.07)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


def test_in_batch_negatives_loss_batch1_near_zero():
    """With batch_size=1, there are no negatives; loss should be near 0."""
    q = torch.randn(1, EMB_DIM)
    p = torch.randn(1, EMB_DIM)
    q = torch.nn.functional.normalize(q, dim=-1)
    p = torch.nn.functional.normalize(p, dim=-1)
    loss = in_batch_negatives_loss(q, p, temperature=0.07)
    # cross_entropy with 1 class always gives 0 (log(softmax([x])) = 0)
    assert loss.item() < 1e-5, f"Expected near-0 loss for batch=1, got {loss.item()}"


def test_in_batch_negatives_loss_identical_embeddings_low_loss():
    """When query and passage embeddings are identical, diagonal scores are high -> low loss."""
    emb = torch.nn.functional.normalize(torch.randn(BATCH, EMB_DIM), dim=-1)
    loss = in_batch_negatives_loss(emb, emb, temperature=0.07)
    # With identical and normalized embeddings, diagonal >> off-diagonal when B is small
    # Loss should be lower than log(B)
    max_loss = torch.log(torch.tensor(float(BATCH)))
    assert loss.item() < max_loss.item(), (
        f"Expected loss < {max_loss.item():.4f}, got {loss.item():.4f}"
    )


# ---------------------------------------------------------------------------
# DPRTrainer tests
# ---------------------------------------------------------------------------


def _make_trainer():
    q_enc = make_encoder()
    p_enc = make_encoder()
    cfg = RetrieverConfig(embedding_dim=EMB_DIM, n_negatives=1)
    params = list(q_enc.parameters()) + list(p_enc.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    return DPRTrainer(q_enc, p_enc, cfg, opt)


def test_dpr_trainer_train_step_keys():
    trainer = _make_trainer()
    result = trainer.train_step(make_input(), make_input())
    assert "loss" in result, "train_step result missing 'loss'"
    assert "sim_pos_mean" in result, "train_step result missing 'sim_pos_mean'"


def test_dpr_trainer_train_step_loss_finite():
    trainer = _make_trainer()
    result = trainer.train_step(make_input(), make_input())
    assert isinstance(result["loss"], float), "loss should be a float"
    assert torch.isfinite(torch.tensor(result["loss"])), f"loss is not finite: {result['loss']}"


def test_dpr_trainer_retrieve_length():
    trainer = _make_trainer()
    query_ids = make_input()[0:1]  # (1, SEQ_LEN)
    passage_list = [make_input()[0:1] for _ in range(5)]
    top_k = 3
    result = trainer.retrieve(query_ids, passage_list, top_k=top_k)
    assert len(result) == top_k, f"Expected {top_k} results, got {len(result)}"


def test_dpr_trainer_retrieve_valid_indices():
    trainer = _make_trainer()
    query_ids = make_input()[0:1]
    n_passages = 5
    passage_list = [make_input()[0:1] for _ in range(n_passages)]
    top_k = 3
    result = trainer.retrieve(query_ids, passage_list, top_k=top_k)
    for idx in result:
        assert 0 <= idx < n_passages, f"Index {idx} out of range [0, {n_passages})"


def test_dpr_trainer_retrieve_same_query_same_top1():
    """The same query should produce the same top-1 result across two calls."""
    trainer = _make_trainer()
    query_ids = make_input()[0:1]
    passage_list = [make_input()[0:1] for _ in range(5)]
    result1 = trainer.retrieve(query_ids, passage_list, top_k=1)
    result2 = trainer.retrieve(query_ids, passage_list, top_k=1)
    assert result1 == result2, f"Different top-1 results: {result1} vs {result2}"
