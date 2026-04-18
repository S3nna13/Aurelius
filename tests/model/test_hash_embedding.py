"""
Tests for src/model/hash_embedding.py
======================================
16 tests covering all public classes and their core behaviours.

Test parameters
---------------
    vocab_size = 64
    d_model    = 16
    B, T       = 2, 6
    n_hashes   = 4
    hash_vocab = 32
"""

import math
import torch
import pytest

from src.model.hash_embedding import (
    HashEmbedding,
    FeatureHasher,
    SubwordHashEmbedding,
    CompressedEmbeddingLayer,
    EmbeddingCompressionBenchmark,
    HashEmbeddingConfig,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
VOCAB_SIZE = 64
D_MODEL    = 16
B, T       = 2, 6
N_HASHES   = 4
HASH_VOCAB = 32


# ===========================================================================
# HashEmbedding tests (4)
# ===========================================================================

def test_hash_embedding_output_shape():
    """Forward pass must return [B, T, d_model]."""
    emb = HashEmbedding(num_hashes=N_HASHES, hash_vocab_size=HASH_VOCAB, d_model=D_MODEL)
    token_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    out = emb(token_ids)
    assert out.shape == (B, T, D_MODEL), f"Expected {(B, T, D_MODEL)}, got {out.shape}"


def test_hash_embedding_different_tokens_different_embeddings():
    """Two clearly different token ids should (almost certainly) map to different embeddings."""
    torch.manual_seed(42)
    emb = HashEmbedding(num_hashes=N_HASHES, hash_vocab_size=HASH_VOCAB, d_model=D_MODEL)
    ids_a = torch.zeros(1, 1, dtype=torch.long)          # token 0
    ids_b = torch.full((1, 1), VOCAB_SIZE - 1, dtype=torch.long)  # token 63
    out_a = emb(ids_a)
    out_b = emb(ids_b)
    # With non-trivial hash tables the two outputs should differ
    assert not torch.allclose(out_a, out_b), "Different token ids should produce different embeddings"


def test_hash_embedding_hash_fn_deterministic():
    """hash_fn must return the same result for the same input and seed."""
    emb = HashEmbedding(num_hashes=N_HASHES, hash_vocab_size=HASH_VOCAB, d_model=D_MODEL)
    token_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    result1 = emb.hash_fn(token_ids, seed=7)
    result2 = emb.hash_fn(token_ids, seed=7)
    assert torch.equal(result1, result2), "hash_fn must be deterministic"


def test_hash_embedding_param_count_less_than_standard():
    """Hash embedding should use fewer parameters than a standard embedding for a large vocabulary."""
    # Use a realistically large vocab (50k) so hash compression is meaningful.
    # Hash tables: 4 * 32 * D_MODEL  vs standard: 50000 * D_MODEL
    large_vocab = 50_000
    emb = HashEmbedding(num_hashes=N_HASHES, hash_vocab_size=HASH_VOCAB, d_model=D_MODEL)
    hash_params = sum(p.numel() for p in emb.parameters())
    standard_params = large_vocab * D_MODEL
    assert hash_params < standard_params, (
        f"Hash params ({hash_params}) should be < standard params ({standard_params})"
    )


# ===========================================================================
# FeatureHasher tests (2)
# ===========================================================================

def test_feature_hasher_output_shape():
    """Forward pass must return [B, d_model]."""
    N_FEAT = 512
    N_SPARSE = 10
    hasher = FeatureHasher(n_features=N_FEAT, d_model=D_MODEL)
    indices = torch.randint(0, 10_000, (B, N_SPARSE))
    values  = torch.randn(B, N_SPARSE)
    out = hasher(indices, values)
    assert out.shape == (B, D_MODEL), f"Expected {(B, D_MODEL)}, got {out.shape}"


def test_feature_hasher_dense_from_sparse():
    """Output should be non-zero for non-zero inputs (sparse → dense projection)."""
    N_FEAT = 512
    N_SPARSE = 10
    hasher = FeatureHasher(n_features=N_FEAT, d_model=D_MODEL)
    indices = torch.randint(0, 10_000, (B, N_SPARSE))
    values  = torch.ones(B, N_SPARSE)
    out = hasher(indices, values)
    # Dense output should not be all zeros after non-trivial embedding weights
    assert out.abs().sum().item() > 0.0, "Dense output should be non-zero for non-zero inputs"


# ===========================================================================
# SubwordHashEmbedding tests (2)
# ===========================================================================

def test_subword_hash_embedding_output_shape():
    """Forward pass must return [B, T, d_model]."""
    MAX_WORD_LEN = 8
    model = SubwordHashEmbedding(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    token_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    char_ids  = torch.randint(0, 256, (B, T, MAX_WORD_LEN))
    out = model(token_ids, char_ids)
    assert out.shape == (B, T, D_MODEL), f"Expected {(B, T, D_MODEL)}, got {out.shape}"


def test_subword_hash_embedding_char_ids_influence_output():
    """Different char_ids should produce different outputs even for identical token_ids."""
    torch.manual_seed(0)
    MAX_WORD_LEN = 8
    model = SubwordHashEmbedding(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    token_ids = torch.zeros(1, 1, dtype=torch.long)
    char_ids_a = torch.zeros(1, 1, MAX_WORD_LEN, dtype=torch.long)
    char_ids_b = torch.full((1, 1, MAX_WORD_LEN), 100, dtype=torch.long)
    out_a = model(token_ids, char_ids_a)
    out_b = model(token_ids, char_ids_b)
    assert not torch.allclose(out_a, out_b), "char_ids should influence the output embedding"


# ===========================================================================
# CompressedEmbeddingLayer tests (3)
# ===========================================================================

def test_compressed_embedding_output_shape():
    """Forward pass must return [B, T, d_model]."""
    layer = CompressedEmbeddingLayer(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    token_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    out = layer(token_ids)
    assert out.shape == (B, T, D_MODEL), f"Expected {(B, T, D_MODEL)}, got {out.shape}"


def test_compressed_embedding_compression_factor_less_than_one():
    """compression_factor() must be < 1.0 for a large-vocabulary scenario."""
    # Use a large vocab where hash compression is meaningful
    large_vocab = 50_000
    layer = CompressedEmbeddingLayer(
        vocab_size=large_vocab, d_model=D_MODEL, compression_ratio=0.05
    )
    factor = layer.compression_factor()
    assert factor < 1.0, f"Expected compression_factor < 1.0, got {factor}"


def test_compressed_embedding_gradient_flows():
    """Loss.backward() must populate gradients for all parameters."""
    layer = CompressedEmbeddingLayer(vocab_size=VOCAB_SIZE, d_model=D_MODEL)
    token_ids = torch.randint(0, VOCAB_SIZE, (B, T))
    out = layer(token_ids)
    loss = out.sum()
    loss.backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' has no gradient"
        assert param.grad.abs().sum().item() > 0.0, f"Parameter '{name}' has all-zero gradient"


# ===========================================================================
# EmbeddingCompressionBenchmark tests (4)
# ===========================================================================

def test_benchmark_standard_param_count():
    """standard_param_count should equal vocab_size * d_model."""
    result = EmbeddingCompressionBenchmark.standard_param_count(VOCAB_SIZE, D_MODEL)
    assert result == VOCAB_SIZE * D_MODEL, (
        f"Expected {VOCAB_SIZE * D_MODEL}, got {result}"
    )


def test_benchmark_hash_param_count_less_than_standard():
    """hash_param_count should be less than standard for a realistically large vocabulary."""
    # With a large vocab, hash compression saves substantially.
    large_vocab = 50_000
    hash_count = EmbeddingCompressionBenchmark.hash_param_count(
        num_hashes=N_HASHES, hash_vocab_size=HASH_VOCAB, d_model=D_MODEL
    )
    standard_count = EmbeddingCompressionBenchmark.standard_param_count(large_vocab, D_MODEL)
    assert hash_count < standard_count, (
        f"hash_count ({hash_count}) should be < standard_count ({standard_count})"
    )


def test_benchmark_collision_rate_in_0_1():
    """collision_rate_estimate must be in [0, 1]."""
    rate = EmbeddingCompressionBenchmark.collision_rate_estimate(VOCAB_SIZE, HASH_VOCAB)
    assert 0.0 <= rate <= 1.0, f"Collision rate {rate} is outside [0, 1]"


def test_benchmark_reconstruction_error_nonneg():
    """reconstruction_error must be ≥ 0."""
    torch.manual_seed(1)
    N = 20
    emb1 = torch.randn(N, D_MODEL)
    emb2 = torch.randn(N, D_MODEL)
    err = EmbeddingCompressionBenchmark.reconstruction_error(emb1, emb2)
    assert err >= 0.0, f"Reconstruction error must be non-negative, got {err}"


# ===========================================================================
# HashEmbeddingConfig tests (1)
# ===========================================================================

def test_hash_embedding_config_defaults():
    """HashEmbeddingConfig must have the documented default values."""
    cfg = HashEmbeddingConfig()
    assert cfg.num_hashes == 4
    assert cfg.hash_vocab_size == 1024
    assert cfg.d_model == 32
    assert cfg.n_char_hashes == 4
    assert math.isclose(cfg.compression_ratio, 0.1)
