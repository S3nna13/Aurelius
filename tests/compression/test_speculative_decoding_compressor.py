"""Tests for src/compression/speculative_decoding_compressor.py — 8+ tests."""
import math
import pytest
import torch
from src.compression.speculative_decoding_compressor import (
    SDCConfig,
    DraftBuffer,
    SpeculativeCompressionMetrics,
    SpeculativeDecodingCompressor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB = 16  # small vocab for tests


@pytest.fixture
def compressor():
    cfg = SDCConfig(draft_steps=4, acceptance_threshold=0.8)
    return SpeculativeDecodingCompressor(cfg)


def _logits(token_id: int, vocab: int = VOCAB) -> torch.Tensor:
    """Return 1-D logits with a spike at token_id."""
    t = torch.full((vocab,), -10.0)
    t[token_id] = 10.0
    return t


def _target_lp(token_id: int, vocab: int = VOCAB, good: bool = True) -> torch.Tensor:
    """Return per-position log-prob for a single draft position."""
    lp = math.log(0.9) if good else math.log(0.5)
    return torch.tensor([lp])


# ---------------------------------------------------------------------------
# 1. DraftBuffer
# ---------------------------------------------------------------------------

def test_draft_buffer_push_and_len():
    buf = DraftBuffer(capacity=8)
    buf.push(1, -0.1)
    buf.push(2, -0.2)
    assert len(buf) == 2


def test_draft_buffer_accept_prefix():
    buf = DraftBuffer(capacity=8)
    for i in range(5):
        buf.push(i, -float(i))
    prefix = buf.accept_prefix(3)
    assert prefix == [0, 1, 2]


def test_draft_buffer_wraps_at_capacity():
    buf = DraftBuffer(capacity=3)
    for i in range(5):
        buf.push(i, 0.0)
    assert len(buf) == 3


def test_draft_buffer_clear():
    buf = DraftBuffer(capacity=8)
    buf.push(1, 0.0)
    buf.clear()
    assert len(buf) == 0


# ---------------------------------------------------------------------------
# 2. compress_step — acceptance
# ---------------------------------------------------------------------------

def test_accepts_token_above_threshold(compressor):
    # log(0.9) > log(0.8) → accepted
    draft = _logits(5).unsqueeze(0)       # shape (1, 16)
    target = torch.tensor([[math.log(0.9)] * VOCAB])  # shape (1, 16)
    accepted = compressor.compress_step(draft, target)
    assert 5 in accepted


def test_rejects_token_below_threshold(compressor):
    # log(0.5) < log(0.8) → rejected
    draft = _logits(3).unsqueeze(0)
    target = torch.full((1, VOCAB), math.log(0.5))
    accepted = compressor.compress_step(draft, target)
    assert accepted == []


def test_prefix_acceptance_stops_at_first_rejection(compressor):
    """With 3 draft positions, second position rejected → only first accepted."""
    # 3 draft positions, each with a distinct greedy token
    draft = torch.zeros(3, VOCAB)
    for i in range(3):
        draft[i, i] = 10.0  # greedy tokens: 0, 1, 2
    # target lp: pos 0 good, pos 1 bad, pos 2 good
    target_lp = torch.tensor([math.log(0.9), math.log(0.5), math.log(0.9)]).unsqueeze(1).expand(3, VOCAB)
    accepted = compressor.compress_step(draft, target_lp)
    assert len(accepted) == 1
    assert accepted[0] == 0


# ---------------------------------------------------------------------------
# 3. Metrics
# ---------------------------------------------------------------------------

def test_metrics_zero_before_any_step():
    c = SpeculativeDecodingCompressor()
    m = c.get_metrics()
    assert m.acceptance_rate == pytest.approx(0.0)
    assert m.tokens_per_step == pytest.approx(0.0)


def test_metrics_acceptance_rate_tracked(compressor):
    draft = _logits(7).unsqueeze(0)
    # All accepted
    target = torch.full((1, VOCAB), math.log(0.95))
    compressor.compress_step(draft, target)
    m = compressor.get_metrics()
    assert m.acceptance_rate > 0.0


def test_metrics_compression_ratio_between_0_and_1(compressor):
    draft = _logits(2).unsqueeze(0)
    target = torch.full((1, VOCAB), math.log(0.9))
    compressor.compress_step(draft, target)
    m = compressor.get_metrics()
    assert 0.0 <= m.compression_ratio <= 1.0


# ---------------------------------------------------------------------------
# 4. SDCConfig defaults
# ---------------------------------------------------------------------------

def test_sdc_config_defaults():
    cfg = SDCConfig()
    assert cfg.draft_steps == 4
    assert cfg.acceptance_threshold == pytest.approx(0.8)
