"""Tests for src/multimodal/cross_modal_attention.py — Q-Former cross-modal attention."""

from __future__ import annotations

import torch

from src.multimodal.cross_modal_attention import (
    CROSS_MODAL_REGISTRY,
    CrossModalAttentionLayer,
    CrossModalConfig,
    QFormer,
)

# ---------------------------------------------------------------------------
# Tiny test config (batch=1, d=64, n_heads=4, n_layers=2, n_query_tokens=8)
# ---------------------------------------------------------------------------

TINY_CFG = CrossModalConfig(
    d_model=64,
    n_heads=4,
    n_layers=2,
    n_query_tokens=8,
    dropout=0.0,
    feedforward_mult=4,
)


# ---------------------------------------------------------------------------
# CrossModalConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults_d_model():
    cfg = CrossModalConfig()
    assert cfg.d_model == 512


def test_config_defaults_n_heads():
    cfg = CrossModalConfig()
    assert cfg.n_heads == 8


def test_config_defaults_n_query_tokens():
    cfg = CrossModalConfig()
    assert cfg.n_query_tokens == 32


def test_config_defaults_n_layers():
    cfg = CrossModalConfig()
    assert cfg.n_layers == 2


def test_config_defaults_dropout():
    cfg = CrossModalConfig()
    assert cfg.dropout == 0.0


def test_config_defaults_feedforward_mult():
    cfg = CrossModalConfig()
    assert cfg.feedforward_mult == 4


# ---------------------------------------------------------------------------
# QFormer.from_config
# ---------------------------------------------------------------------------


def test_qformer_from_config_builds():
    model = QFormer.from_config(TINY_CFG)
    assert isinstance(model, QFormer)


def test_qformer_from_config_correct_n_layers():
    model = QFormer.from_config(TINY_CFG)
    assert len(model.layers) == TINY_CFG.n_layers


def test_qformer_from_config_query_shape():
    model = QFormer.from_config(TINY_CFG)
    assert model.query_tokens.shape == (1, TINY_CFG.n_query_tokens, TINY_CFG.d_model)


def test_qformer_query_tokens_are_parameter():
    model = QFormer.from_config(TINY_CFG)
    assert isinstance(model.query_tokens, torch.nn.Parameter)


# ---------------------------------------------------------------------------
# QFormer forward shape and correctness
# ---------------------------------------------------------------------------


def test_qformer_forward_output_shape():
    torch.manual_seed(42)
    model = QFormer.from_config(TINY_CFG)
    model.train(False)
    vision = torch.randn(1, 16, TINY_CFG.d_model)
    out = model(vision)
    assert out.shape == (1, TINY_CFG.n_query_tokens, TINY_CFG.d_model)


def test_qformer_forward_output_shape_exact():
    torch.manual_seed(42)
    model = QFormer.from_config(TINY_CFG)
    model.train(False)
    vision = torch.randn(1, 16, TINY_CFG.d_model)
    out = model(vision)
    B, Nq, D = out.shape
    assert B == 1
    assert Nq == TINY_CFG.n_query_tokens
    assert D == TINY_CFG.d_model


def test_qformer_forward_no_nan():
    torch.manual_seed(42)
    model = QFormer.from_config(TINY_CFG)
    model.train(False)
    vision = torch.randn(1, 16, TINY_CFG.d_model)
    out = model(vision)
    assert not out.isnan().any()


def test_qformer_forward_batch_size_2():
    torch.manual_seed(42)
    model = QFormer.from_config(TINY_CFG)
    model.train(False)
    vision = torch.randn(2, 16, TINY_CFG.d_model)
    out = model(vision)
    assert out.shape == (2, TINY_CFG.n_query_tokens, TINY_CFG.d_model)


def test_qformer_forward_no_nan_batch_2():
    torch.manual_seed(42)
    model = QFormer.from_config(TINY_CFG)
    model.train(False)
    vision = torch.randn(2, 16, TINY_CFG.d_model)
    out = model(vision)
    assert not out.isnan().any()


# ---------------------------------------------------------------------------
# CrossModalAttentionLayer edge case: seq_len=1
# ---------------------------------------------------------------------------


def test_cross_modal_attention_layer_seq_len_1():
    """Edge case: both query and kv have seq_len=1 — must not crash."""
    torch.manual_seed(42)
    layer = CrossModalAttentionLayer(d_model=64, n_heads=4, dropout=0.0)
    layer.train(False)
    query = torch.randn(1, 1, 64)
    kv = torch.randn(1, 1, 64)
    out = layer(query, kv)
    assert out.shape == (1, 1, 64)
    assert not out.isnan().any()


# ---------------------------------------------------------------------------
# Registry checks
# ---------------------------------------------------------------------------


def test_cross_modal_registry_contains_qformer():
    assert "qformer" in CROSS_MODAL_REGISTRY


def test_cross_modal_registry_qformer_is_class():
    assert CROSS_MODAL_REGISTRY["qformer"] is QFormer


def test_modality_projector_registry_contains_qformer():
    """After importing cross_modal_attention, MODALITY_PROJECTOR_REGISTRY must have 'QFormer'."""
    from src.multimodal.multimodal_registry import MODALITY_PROJECTOR_REGISTRY

    assert "QFormer" in MODALITY_PROJECTOR_REGISTRY


def test_modality_projector_registry_qformer_is_class():
    from src.multimodal.multimodal_registry import MODALITY_PROJECTOR_REGISTRY

    assert MODALITY_PROJECTOR_REGISTRY["QFormer"] is QFormer
