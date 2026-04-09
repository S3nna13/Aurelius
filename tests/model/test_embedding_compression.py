"""Tests for src/model/embedding_compression.py."""
import torch
import pytest

from src.model.embedding_compression import (
    EmbedConfig,
    FactorizedEmbedding,
    ClusteredEmbedding,
    TiedEmbedding,
    estimate_embedding_memory,
    EmbeddingCompressor,
)

# Shared constants
VOCAB_SIZE = 128
D_MODEL = 64
EMBEDDING_DIM = 16
N_CLUSTERS = 32
TARGET_RANK = 8
BATCH_SIZE = 2
SEQ_LEN = 8


@pytest.fixture
def input_ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))


# 1. EmbedConfig defaults
def test_embed_config_defaults():
    cfg = EmbedConfig()
    assert cfg.vocab_size == 32000
    assert cfg.d_model == 512
    assert cfg.embedding_dim == 128
    assert cfg.n_clusters == 256
    assert cfg.tie_weights is True
    assert cfg.use_factorization is True


# 2. FactorizedEmbedding output shape
def test_factorized_embedding_output_shape(input_ids):
    torch.manual_seed(0)
    model = FactorizedEmbedding(VOCAB_SIZE, EMBEDDING_DIM, D_MODEL)
    out = model(input_ids)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


# 3. FactorizedEmbedding compression_ratio > 1 when embedding_dim < d_model
def test_factorized_embedding_compression_ratio():
    torch.manual_seed(0)
    model = FactorizedEmbedding(VOCAB_SIZE, EMBEDDING_DIM, D_MODEL)
    ratio = model.compression_ratio(D_MODEL)
    assert ratio > 1.0, f"Expected ratio > 1, got {ratio}"


# 4. FactorizedEmbedding num_parameters correct formula
def test_factorized_embedding_num_params():
    torch.manual_seed(0)
    model = FactorizedEmbedding(VOCAB_SIZE, EMBEDDING_DIM, D_MODEL)
    expected = VOCAB_SIZE * EMBEDDING_DIM + EMBEDDING_DIM * D_MODEL
    assert model.num_parameters() == expected


# 5. ClusteredEmbedding output shape
def test_clustered_embedding_output_shape(input_ids):
    torch.manual_seed(0)
    model = ClusteredEmbedding(VOCAB_SIZE, D_MODEL, N_CLUSTERS)
    out = model(input_ids)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


# 6. TiedEmbedding embed shape
def test_tied_embedding_embed_shape(input_ids):
    torch.manual_seed(0)
    model = TiedEmbedding(VOCAB_SIZE, D_MODEL)
    out = model.embed(input_ids)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, D_MODEL)


# 7. TiedEmbedding unembed shape
def test_tied_embedding_unembed_shape():
    torch.manual_seed(0)
    model = TiedEmbedding(VOCAB_SIZE, D_MODEL)
    hidden = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    out = model.unembed(hidden)
    assert out.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)


# 8. TiedEmbedding weight shared between embed and unembed
def test_tied_embedding_weight_shared():
    torch.manual_seed(0)
    model = TiedEmbedding(VOCAB_SIZE, D_MODEL)
    # unembed uses weight.T, so weight == weight.T.T
    assert torch.equal(model.weight, model.weight.T.T)
    # Both embed and unembed reference the same parameter object
    input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    embed_out = model.embed(input_ids)
    # Verify unembed is w @ weight.T (reconstruction check)
    hidden = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
    unembed_out = model.unembed(hidden)
    expected = hidden @ model.weight.T
    assert torch.allclose(unembed_out, expected)


# 9. estimate_memory_factorized_less than standard
def test_estimate_memory_factorized_less():
    result = estimate_embedding_memory(VOCAB_SIZE, D_MODEL, EMBEDDING_DIM)
    assert result["factorized_mb"] < result["standard_mb"]
    assert result["reduction_factor"] > 1.0


# 10. EmbeddingCompressor output shapes
def test_embedding_compressor_shapes():
    torch.manual_seed(0)
    weight = torch.randn(VOCAB_SIZE, D_MODEL)
    compressor = EmbeddingCompressor(target_rank=TARGET_RANK)
    u_r, v_r = compressor.compress(weight)
    assert u_r.shape == (VOCAB_SIZE, TARGET_RANK), f"u_r shape: {u_r.shape}"
    assert v_r.shape == (TARGET_RANK, D_MODEL), f"v_r shape: {v_r.shape}"


# 11. EmbeddingCompressor reconstruction error in [0, 1]
def test_embedding_compressor_reconstruction_error_range():
    torch.manual_seed(0)
    weight = torch.randn(VOCAB_SIZE, D_MODEL)
    compressor = EmbeddingCompressor(target_rank=TARGET_RANK)
    u_r, v_r = compressor.compress(weight)
    err = compressor.reconstruction_error(weight, u_r, v_r)
    assert 0.0 <= err <= 1.0, f"Reconstruction error out of range: {err}"


# 12. FactorizedEmbedding gradient flow
def test_factorized_embedding_gradient_flow(input_ids):
    torch.manual_seed(0)
    model = FactorizedEmbedding(VOCAB_SIZE, EMBEDDING_DIM, D_MODEL)
    out = model(input_ids)
    loss = out.sum()
    loss.backward()
    assert model.embed.weight.grad is not None, "embed.weight has no grad"
    assert model.proj.weight.grad is not None, "proj.weight has no grad"
    assert model.embed.weight.grad.abs().sum() > 0
    assert model.proj.weight.grad.abs().sum() > 0
