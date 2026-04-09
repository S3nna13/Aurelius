"""Tests for sequence parallelism simulation."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.sequence_parallel import (
    SeqParallelConfig,
    partition_sequence,
    gather_sequence,
    compute_chunk_attention,
    RingAttentionSimulator,
    compute_communication_cost,
    SequenceParallelTrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# Shared constants
B = 2
T = 32
D = 16
WORLD_SIZE = 4


@pytest.fixture
def seq_config():
    return SeqParallelConfig(world_size=WORLD_SIZE)


@pytest.fixture
def input_ids():
    torch.manual_seed(42)
    return torch.randint(0, 256, (B, T))


@pytest.fixture
def qkv():
    torch.manual_seed(42)
    Q = torch.randn(B, T, D)
    K = torch.randn(B, T, D)
    V = torch.randn(B, T, D)
    return Q, K, V


# ---------------------------------------------------------------------------
# 1. SeqParallelConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = SeqParallelConfig()
    assert cfg.world_size == 4
    assert cfg.overlap_tokens == 0
    assert cfg.ring_attn is True
    assert cfg.load_balance is True


# ---------------------------------------------------------------------------
# 2. partition_sequence creates correct number of chunks
# ---------------------------------------------------------------------------


def test_partition_creates_correct_num_chunks(input_ids):
    chunks = partition_sequence(input_ids, WORLD_SIZE)
    assert len(chunks) == WORLD_SIZE


# ---------------------------------------------------------------------------
# 3. partition_sequence chunks cover all tokens (no loss)
# ---------------------------------------------------------------------------


def test_partition_covers_all_tokens(input_ids):
    chunks = partition_sequence(input_ids, WORLD_SIZE, overlap=0)
    total_tokens = sum(c.shape[1] for c in chunks)
    assert total_tokens == T


# ---------------------------------------------------------------------------
# 4. partition_sequence with overlap extends chunks
# ---------------------------------------------------------------------------


def test_partition_with_overlap(input_ids):
    overlap = 4
    chunks = partition_sequence(input_ids, WORLD_SIZE, overlap=overlap)
    # First chunk has no overlap prefix; subsequent chunks are extended
    assert chunks[0].shape[1] == T // WORLD_SIZE
    for c in chunks[1:]:
        assert c.shape[1] >= T // WORLD_SIZE + overlap or c.shape[1] > T // WORLD_SIZE


# ---------------------------------------------------------------------------
# 5. gather_sequence inverts partition (round-trip)
# ---------------------------------------------------------------------------


def test_gather_inverts_partition(input_ids):
    chunks = partition_sequence(input_ids, WORLD_SIZE, overlap=0)
    recovered = gather_sequence(chunks, overlap=0)
    assert torch.equal(recovered, input_ids)


# ---------------------------------------------------------------------------
# 6. gather_sequence with overlap removes overlaps correctly
# ---------------------------------------------------------------------------


def test_gather_with_overlap_removes_overlap(input_ids):
    overlap = 4
    chunks = partition_sequence(input_ids, WORLD_SIZE, overlap=overlap)
    recovered = gather_sequence(chunks, overlap=overlap)
    assert torch.equal(recovered, input_ids)


# ---------------------------------------------------------------------------
# 7. compute_chunk_attention output shape correct
# ---------------------------------------------------------------------------


def test_chunk_attention_output_shape(qkv):
    Q, K, V = qkv
    chunk_len = T // WORLD_SIZE
    q_chunk = Q[:, :chunk_len]
    k_chunks = [K[:, i * chunk_len:(i + 1) * chunk_len] for i in range(WORLD_SIZE)]
    v_chunks = [V[:, i * chunk_len:(i + 1) * chunk_len] for i in range(WORLD_SIZE)]
    out = compute_chunk_attention(q_chunk, k_chunks, v_chunks)
    assert out.shape == (B, chunk_len, D)


# ---------------------------------------------------------------------------
# 8. compute_chunk_attention with single chunk matches standard attention
# ---------------------------------------------------------------------------


def test_chunk_attention_single_chunk_matches_standard(qkv):
    Q, K, V = qkv
    # Full standard attention
    import math
    scale = math.sqrt(D)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
    weights = torch.softmax(scores, dim=-1)
    expected = torch.matmul(weights, V)

    # compute_chunk_attention with single chunk = all keys
    out = compute_chunk_attention(Q, [K], [V])
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 9. RingAttentionSimulator output shape (B, T, d)
# ---------------------------------------------------------------------------


def test_ring_attention_output_shape(qkv, seq_config):
    Q, K, V = qkv
    sim = RingAttentionSimulator(seq_config, d_model=D)
    out = sim.forward(Q, K, V)
    assert out.shape == (B, T, D)


# ---------------------------------------------------------------------------
# 10. RingAttentionSimulator ring_attn=True uses neighbors
# ---------------------------------------------------------------------------


def test_ring_attention_uses_neighbors(qkv):
    Q, K, V = qkv
    cfg_ring = SeqParallelConfig(world_size=WORLD_SIZE, ring_attn=True)
    cfg_causal = SeqParallelConfig(world_size=WORLD_SIZE, ring_attn=False)
    sim_ring = RingAttentionSimulator(cfg_ring, d_model=D)
    sim_causal = RingAttentionSimulator(cfg_causal, d_model=D)
    out_ring = sim_ring.forward(Q, K, V)
    out_causal = sim_causal.forward(Q, K, V)
    # Ring and causal should produce different results (different key sets)
    assert not torch.allclose(out_ring, out_causal, atol=1e-5)


# ---------------------------------------------------------------------------
# 11. compute_communication_cost returns correct keys
# ---------------------------------------------------------------------------


def test_communication_cost_keys(seq_config):
    result = compute_communication_cost(T, seq_config, d_model=D)
    assert set(result.keys()) == {"tokens_per_rank", "overlap_tokens_total", "all_reduce_volume"}


# ---------------------------------------------------------------------------
# 12. compute_communication_cost tokens_per_rank = T / world_size
# ---------------------------------------------------------------------------


def test_communication_cost_tokens_per_rank(seq_config):
    result = compute_communication_cost(T, seq_config, d_model=D)
    assert result["tokens_per_rank"] == T // WORLD_SIZE


# ---------------------------------------------------------------------------
# 13. SequenceParallelTrainer train_step returns correct keys
# ---------------------------------------------------------------------------


def test_trainer_train_step_returns_correct_keys():
    model_cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    model = AureliusTransformer(model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    sp_cfg = SeqParallelConfig(world_size=4, overlap_tokens=0)
    trainer = SequenceParallelTrainer(model, optimizer, sp_cfg)

    input_ids = torch.randint(0, 256, (B, T))
    result = trainer.train_step(input_ids)

    assert set(result.keys()) == {"loss", "n_chunks", "tokens_per_chunk"}
    assert isinstance(result["loss"], float)
    assert result["n_chunks"] == 4
    assert result["tokens_per_chunk"] > 0
