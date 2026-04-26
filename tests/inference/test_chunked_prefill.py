"""Tests for chunked prefill inference module."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from src.inference.chunked_prefill import (
    ChunkedPrefillConfig,
    ChunkedPrefillEngine,
    chunk_sequence,
    merge_chunked_logits,
)

# ── Shared constants ──────────────────────────────────────────────────────────
VOCAB_SIZE = 64
D_MODEL = 16
B = 1
PROMPT_LEN = 20
CHUNK_SIZE = 8


# ── Mock model ────────────────────────────────────────────────────────────────
class MockModel(nn.Module):
    def __init__(self, vocab_size: int = VOCAB_SIZE, d_model: int = D_MODEL):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        return (None, self.proj(self.embed(input_ids)), None)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def model():
    torch.manual_seed(0)
    return MockModel()


@pytest.fixture
def config():
    return ChunkedPrefillConfig(chunk_size=CHUNK_SIZE)


@pytest.fixture
def engine(model, config):
    return ChunkedPrefillEngine(model=model, config=config)


@pytest.fixture
def prompt_ids():
    torch.manual_seed(42)
    return torch.randint(0, VOCAB_SIZE, (B, PROMPT_LEN))


# ── Test 1: ChunkedPrefillConfig instantiates with defaults ──────────────────
def test_config_defaults():
    cfg = ChunkedPrefillConfig()
    assert cfg.chunk_size == 512
    assert cfg.overlap == 0
    assert cfg.use_kv_cache is True


# ── Test 2: ChunkedPrefillEngine instantiates ─────────────────────────────────
def test_engine_instantiates(model):
    engine = ChunkedPrefillEngine(model=model)
    assert engine.config.chunk_size == 512
    assert engine.model is model


# ── Test 3: get_chunk_schedule seq_len=10, chunk=4, overlap=0 ─────────────────
def test_chunk_schedule_basic():
    cfg = ChunkedPrefillConfig(chunk_size=4, overlap=0)
    engine = ChunkedPrefillEngine(model=MockModel(), config=cfg)
    schedule = engine.get_chunk_schedule(10)
    assert schedule == [(0, 4), (4, 8), (8, 10)]


# ── Test 4: get_chunk_schedule seq_len=8, chunk=8 → 1 chunk ──────────────────
def test_chunk_schedule_exact_fit():
    cfg = ChunkedPrefillConfig(chunk_size=8, overlap=0)
    engine = ChunkedPrefillEngine(model=MockModel(), config=cfg)
    schedule = engine.get_chunk_schedule(8)
    assert schedule == [(0, 8)]


# ── Test 5: get_chunk_schedule with overlap=1 ─────────────────────────────────
def test_chunk_schedule_with_overlap():
    cfg = ChunkedPrefillConfig(chunk_size=4, overlap=1)
    engine = ChunkedPrefillEngine(model=MockModel(), config=cfg)
    schedule = engine.get_chunk_schedule(10)
    # With chunk_size=4, overlap=1: advance=3 each step
    # (0,4), (3,7), (6,10)
    assert len(schedule) == 3
    # Each consecutive chunk should overlap by 1 token
    for i in range(len(schedule) - 1):
        prev_end = schedule[i][1]
        next_start = schedule[i + 1][0]
        assert next_start < prev_end, "Consecutive chunks should overlap"
    # All positions covered
    assert schedule[0][0] == 0
    assert schedule[-1][1] == 10


# ── Test 6: chunk_sequence T=10, chunk=4 → 3 tensors, last shape (B, 2) ──────
def test_chunk_sequence_basic():
    ids = torch.arange(10).unsqueeze(0).expand(B, -1)  # (B, 10)
    chunks = chunk_sequence(ids, chunk_size=4)
    assert len(chunks) == 3
    assert chunks[0].shape == (B, 4)
    assert chunks[1].shape == (B, 4)
    assert chunks[2].shape == (B, 2)


# ── Test 7: chunk_sequence T=8, chunk=8 → 1 tensor shape (B, 8) ──────────────
def test_chunk_sequence_exact_fit():
    ids = torch.arange(8).unsqueeze(0).expand(B, -1)
    chunks = chunk_sequence(ids, chunk_size=8)
    assert len(chunks) == 1
    assert chunks[0].shape == (B, 8)


# ── Test 8: merge_chunked_logits 3 chunks of (B,4,V) → (B, 12, V) ────────────
def test_merge_chunked_logits_no_overlap():
    V = VOCAB_SIZE
    chunks = [torch.zeros(B, 4, V) for _ in range(3)]
    merged = merge_chunked_logits(chunks, overlap=0)
    assert merged.shape == (B, 12, V)


# ── Test 9: merge_chunked_logits with overlap=1 removes 1 token per boundary ─
def test_merge_chunked_logits_with_overlap():
    V = VOCAB_SIZE
    # 3 chunks of size 4 with overlap=1
    # chunk0: 4 tokens kept as-is
    # chunk1: drop first 1 → 3 tokens kept
    # chunk2: drop first 1 → 3 tokens kept
    # total = 4 + 3 + 3 = 10
    chunks = [torch.zeros(B, 4, V) for _ in range(3)]
    merged = merge_chunked_logits(chunks, overlap=1)
    assert merged.shape == (B, 10, V)


# ── Test 10: prefill returns logits of shape (B, T_prompt, V) ─────────────────
def test_prefill_logits_shape(engine, prompt_ids):
    logits, stats = engine.prefill(prompt_ids)
    assert logits.shape == (B, PROMPT_LEN, VOCAB_SIZE)


# ── Test 11: prefill stats.n_chunks == ceil(T_prompt / chunk_size) ────────────
def test_prefill_stats_n_chunks(engine, prompt_ids):
    _, stats = engine.prefill(prompt_ids)
    expected_n_chunks = math.ceil(PROMPT_LEN / CHUNK_SIZE)
    assert stats.n_chunks == expected_n_chunks


# ── Test 12: prefill stats.peak_tokens_in_flight <= chunk_size + overlap ──────
def test_prefill_peak_tokens(engine, prompt_ids):
    _, stats = engine.prefill(prompt_ids)
    max_allowed = engine.config.chunk_size + engine.config.overlap
    assert stats.peak_tokens_in_flight <= max_allowed


# ── Test 13: prefill_and_decode returns output_ids shape (B, T_prompt + max_new) ─
def test_prefill_and_decode_shape(engine, prompt_ids):
    max_new = 5
    output_ids, stats = engine.prefill_and_decode(prompt_ids, max_new_tokens=max_new)
    assert output_ids.shape == (B, PROMPT_LEN + max_new)


# ── Test 14: prefill_and_decode greedy is deterministic ──────────────────────
def test_prefill_and_decode_deterministic(model, config, prompt_ids):
    engine1 = ChunkedPrefillEngine(model=model, config=config)
    engine2 = ChunkedPrefillEngine(model=model, config=config)
    out1, _ = engine1.prefill_and_decode(prompt_ids, max_new_tokens=5, temperature=0.0)
    out2, _ = engine2.prefill_and_decode(prompt_ids, max_new_tokens=5, temperature=0.0)
    assert torch.equal(out1, out2)


# ── Test 15: Large prompt T=100, chunk=16 prefills without error ──────────────
def test_large_prompt_prefill():
    torch.manual_seed(7)
    m = MockModel()
    cfg = ChunkedPrefillConfig(chunk_size=16)
    engine = ChunkedPrefillEngine(model=m, config=cfg)
    ids = torch.randint(0, VOCAB_SIZE, (1, 100))
    logits, stats = engine.prefill(ids)
    assert logits.shape == (1, 100, VOCAB_SIZE)
    assert stats.n_chunks == math.ceil(100 / 16)


# ── Test 16: chunk_size >= prompt_len → 1 chunk (no chunking needed) ─────────
def test_single_chunk_when_chunk_size_ge_prompt_len():
    m = MockModel()
    cfg = ChunkedPrefillConfig(chunk_size=PROMPT_LEN + 10)  # larger than prompt
    engine = ChunkedPrefillEngine(model=m, config=cfg)
    ids = torch.randint(0, VOCAB_SIZE, (B, PROMPT_LEN))
    logits, stats = engine.prefill(ids)
    assert stats.n_chunks == 1
    assert logits.shape == (B, PROMPT_LEN, VOCAB_SIZE)
