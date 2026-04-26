"""Tests for src/model/embedding_compression_v2.py

Tiny config: VOCAB=32, D=16, D_EMBED=8, N_CB=2, N_CODES=8, B=2, T=4
"""

from __future__ import annotations

import torch

from src.model.embedding_compression_v2 import (
    EmbedConfigV2,
    FactorizedEmbeddingV2,
    MixedPrecisionEmbedding,
    ProductQuantizedEmbedding,
    compute_embedding_norm,
    embed_dropout,
)

VOCAB = 32
D = 16
D_EMBED = 8
N_CB = 2
N_CODES = 8
B = 2
T = 4


def _ids():
    return torch.randint(0, VOCAB, (B, T))


# ---------------------------------------------------------------------------
# EmbedConfigV2 defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = EmbedConfigV2()
    assert cfg.vocab_size == 50257
    assert cfg.d_model == 512
    assert cfg.d_embed == 128
    assert cfg.n_codebooks == 4
    assert cfg.n_codes == 256
    assert cfg.tie_weights is True


# ---------------------------------------------------------------------------
# FactorizedEmbeddingV2
# ---------------------------------------------------------------------------


def test_factorized_embedding_output_shape():
    emb = FactorizedEmbeddingV2(VOCAB, D_EMBED, D)
    out = emb(_ids())
    assert out.shape == (B, T, D)


def test_factorized_embedding_param_count_less_than_standard():
    emb = FactorizedEmbeddingV2(VOCAB, D_EMBED, D)
    assert emb.parameter_count() < VOCAB * D


def test_factorized_embedding_output_finite():
    emb = FactorizedEmbeddingV2(VOCAB, D_EMBED, D)
    out = emb(_ids())
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# ProductQuantizedEmbedding
# ---------------------------------------------------------------------------


def test_pq_embedding_output_shape():
    emb = ProductQuantizedEmbedding(VOCAB, D, N_CB, N_CODES)
    out = emb(_ids())
    assert out.shape == (B, T, D)


def test_pq_embedding_param_count_less_than_standard():
    emb = ProductQuantizedEmbedding(VOCAB, D, N_CB, N_CODES)
    assert emb.parameter_count() < VOCAB * D


def test_pq_embedding_output_finite():
    emb = ProductQuantizedEmbedding(VOCAB, D, N_CB, N_CODES)
    out = emb(_ids())
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# embed_dropout
# ---------------------------------------------------------------------------


def test_embed_dropout_zeros_at_p1():
    emb = torch.randn(B, T, D)
    torch.manual_seed(0)
    out = embed_dropout(emb, p=1.0, training=True)
    assert torch.all(out == 0)


def test_embed_dropout_unchanged_at_p0():
    emb = torch.randn(B, T, D)
    out = embed_dropout(emb, p=0.0, training=True)
    assert torch.allclose(out, emb)


def test_embed_dropout_no_op_when_not_training():
    emb = torch.randn(B, T, D)
    out = embed_dropout(emb, p=0.9, training=False)
    assert torch.allclose(out, emb)


# ---------------------------------------------------------------------------
# compute_embedding_norm
# ---------------------------------------------------------------------------


def test_compute_embedding_norm_shape():
    weight = torch.randn(VOCAB, D)
    norms = compute_embedding_norm(weight)
    assert norms.shape == (VOCAB,)


def test_compute_embedding_norm_non_negative():
    weight = torch.randn(VOCAB, D)
    norms = compute_embedding_norm(weight)
    assert (norms >= 0).all()


# ---------------------------------------------------------------------------
# MixedPrecisionEmbedding
# ---------------------------------------------------------------------------


def test_mixed_precision_output_shape():
    emb = MixedPrecisionEmbedding(VOCAB, D, dtype=torch.float16)
    out = emb(_ids())
    assert out.shape == (B, T, D)


def test_mixed_precision_output_dtype_float32():
    emb = MixedPrecisionEmbedding(VOCAB, D, dtype=torch.float16)
    out = emb(_ids())
    assert out.dtype == torch.float32
