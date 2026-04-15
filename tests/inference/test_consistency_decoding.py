"""Tests for consistency decoding (Wang et al. 2023 'Self-Consistency').

Uses a lightweight mock model:
  nn.Embedding(vocab, d_model) + nn.Linear(d_model, vocab)
  forward(ids) -> (None, logits, None)   logits: (B, T, V)

Hyperparameters: vocab=32, d_model=16, seq_len=4, batch=2,
                 max_new_tokens=3, n_samples=4.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.inference.consistency_decoding import (
    ConsistencyConfig,
    ConsistencyDecoder,
    compute_sequence_agreement,
    greedy_decode,
    majority_vote,
    sample_with_temperature,
    temperature_decode,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB = 32
D_MODEL = 16
SEQ_LEN = 4
BATCH = 2
MAX_NEW = 3
N_SAMPLES = 4


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------


class MockModel(nn.Module):
    """Minimal transformer mock: Embedding -> Linear -> (None, logits, None)."""

    def __init__(self, vocab: int = VOCAB, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, ids: torch.Tensor, **kwargs):
        # ids: (B, T)
        x = self.embed(ids)       # (B, T, D)
        logits = self.head(x)     # (B, T, V)
        return None, logits, None


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = MockModel()
    m.eval()
    return m


@pytest.fixture(scope="module")
def input_ids():
    torch.manual_seed(1)
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN))


@pytest.fixture(scope="module")
def config():
    return ConsistencyConfig(
        n_samples=N_SAMPLES,
        temperature=0.7,
        top_k=10,
        aggregation="majority_vote",
    )


@pytest.fixture(scope="module")
def decoder(model, config):
    return ConsistencyDecoder(model, config)


# ---------------------------------------------------------------------------
# sample_with_temperature tests (4 tests)
# ---------------------------------------------------------------------------


def test_sample_with_temperature_output_shape():
    """Output should be (B,)."""
    torch.manual_seed(42)
    logits = torch.randn(BATCH, VOCAB)
    result = sample_with_temperature(logits, temperature=1.0)
    assert result.shape == (BATCH,)


def test_sample_with_temperature_values_in_vocab():
    """All sampled ids should be in [0, VOCAB)."""
    torch.manual_seed(42)
    logits = torch.randn(BATCH, VOCAB)
    result = sample_with_temperature(logits, temperature=1.0)
    assert (result >= 0).all()
    assert (result < VOCAB).all()


def test_sample_with_temperature_high_temp_diverse():
    """Very high temperature should produce diverse outputs across many calls."""
    torch.manual_seed(7)
    logits = torch.randn(1, VOCAB)
    tokens = set()
    for _ in range(50):
        tokens.add(sample_with_temperature(logits, temperature=100.0).item())
    # With 50 draws at very high temp over 32 vocab tokens, expect > 1 unique
    assert len(tokens) > 1


def test_sample_with_temperature_top_k_1_deterministic():
    """top_k=1 should always return the argmax token (deterministic)."""
    torch.manual_seed(0)
    logits = torch.randn(BATCH, VOCAB)
    expected = logits.argmax(dim=-1)
    results = [sample_with_temperature(logits, temperature=1.0, top_k=1) for _ in range(10)]
    for res in results:
        assert torch.equal(res, expected)


# ---------------------------------------------------------------------------
# greedy_decode tests (2 tests)
# ---------------------------------------------------------------------------


def test_greedy_decode_output_shape(model, input_ids):
    """greedy_decode should return (B, max_new_tokens)."""
    out = greedy_decode(model, input_ids, MAX_NEW)
    assert out.shape == (BATCH, MAX_NEW)


def test_greedy_decode_values_in_vocab(model, input_ids):
    """All greedy decoded tokens should be valid vocab indices."""
    out = greedy_decode(model, input_ids, MAX_NEW)
    assert (out >= 0).all()
    assert (out < VOCAB).all()


# ---------------------------------------------------------------------------
# temperature_decode tests (2 tests)
# ---------------------------------------------------------------------------


def test_temperature_decode_output_shape_matches_greedy(model, input_ids):
    """temperature_decode should return the same shape as greedy_decode."""
    greedy_out = greedy_decode(model, input_ids, MAX_NEW)
    temp_out = temperature_decode(model, input_ids, MAX_NEW, temperature=1.0)
    assert temp_out.shape == greedy_out.shape


def test_temperature_decode_values_in_vocab(model, input_ids):
    """All temperature decoded tokens should be valid vocab indices."""
    out = temperature_decode(model, input_ids, MAX_NEW, temperature=0.7, top_k=10)
    assert (out >= 0).all()
    assert (out < VOCAB).all()


# ---------------------------------------------------------------------------
# majority_vote tests (4 tests)
# ---------------------------------------------------------------------------


def test_majority_vote_all_same_returns_that_token():
    """When all samples are identical, majority_vote should return that value."""
    token = torch.tensor([5, 7])  # (B=2,)
    sequences = [token.clone() for _ in range(4)]
    result = majority_vote(sequences)
    assert torch.equal(result, token)


def test_majority_vote_shape_1d():
    """majority_vote on (B,) inputs returns (B,)."""
    sequences = [torch.randint(0, VOCAB, (BATCH,)) for _ in range(N_SAMPLES)]
    result = majority_vote(sequences)
    assert result.shape == (BATCH,)


def test_majority_vote_shape_2d():
    """majority_vote on (B, T) inputs returns (B, T)."""
    sequences = [torch.randint(0, VOCAB, (BATCH, MAX_NEW)) for _ in range(N_SAMPLES)]
    result = majority_vote(sequences)
    assert result.shape == (BATCH, MAX_NEW)


def test_majority_vote_3_samples_2_agree():
    """When 2 out of 3 samples agree on a token, that token wins."""
    # Position 0: tokens 3,3,9 -> winner is 3
    # Position 1: tokens 1,5,5 -> winner is 5
    s0 = torch.tensor([3, 1])
    s1 = torch.tensor([3, 5])
    s2 = torch.tensor([9, 5])
    result = majority_vote([s0, s1, s2])
    assert result[0].item() == 3
    assert result[1].item() == 5


# ---------------------------------------------------------------------------
# compute_sequence_agreement tests (3 tests)
# ---------------------------------------------------------------------------


def test_compute_sequence_agreement_identical():
    """Agreement is 1.0 when all sequences are identical."""
    seq = torch.randint(0, VOCAB, (BATCH, MAX_NEW))
    sequences = [seq.clone() for _ in range(N_SAMPLES)]
    score = compute_sequence_agreement(sequences)
    assert score == pytest.approx(1.0)


def test_compute_sequence_agreement_differs():
    """Agreement is < 1.0 when sequences differ."""
    torch.manual_seed(99)
    sequences = [torch.randint(0, VOCAB, (BATCH, MAX_NEW)) for _ in range(N_SAMPLES)]
    score = compute_sequence_agreement(sequences)
    assert score < 1.0


def test_compute_sequence_agreement_in_range():
    """Agreement score is always in [0, 1]."""
    torch.manual_seed(123)
    sequences = [torch.randint(0, VOCAB, (BATCH, MAX_NEW)) for _ in range(N_SAMPLES)]
    score = compute_sequence_agreement(sequences)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# ConsistencyConfig tests (1 test)
# ---------------------------------------------------------------------------


def test_consistency_config_defaults():
    """ConsistencyConfig should have the specified default values."""
    cfg = ConsistencyConfig()
    assert cfg.n_samples == 8
    assert cfg.temperature == pytest.approx(0.7)
    assert cfg.top_k == 50
    assert cfg.aggregation == "majority_vote"


# ---------------------------------------------------------------------------
# ConsistencyDecoder tests (6 tests)
# ---------------------------------------------------------------------------


def test_decoder_decode_output_shape(decoder, input_ids):
    """decode() should return (B, max_new_tokens)."""
    out = decoder.decode(input_ids, MAX_NEW)
    assert out.shape == (BATCH, MAX_NEW)


def test_decoder_decode_with_score_returns_tuple(decoder, input_ids):
    """decode_with_score() should return a tuple of (Tensor, float)."""
    result = decoder.decode_with_score(input_ids, MAX_NEW)
    assert isinstance(result, tuple)
    assert len(result) == 2
    voted, score = result
    assert isinstance(voted, torch.Tensor)
    assert isinstance(score, float)


def test_decoder_agreement_score_in_range(decoder, input_ids):
    """Agreement score from decode_with_score should be in [0, 1]."""
    _, score = decoder.decode_with_score(input_ids, MAX_NEW)
    assert 0.0 <= score <= 1.0


def test_decoder_agreement_score_type_is_float(decoder, input_ids):
    """The score returned by decode_with_score must be a Python float."""
    _, score = decoder.decode_with_score(input_ids, MAX_NEW)
    assert type(score) is float


def test_decoder_agreement_score_1_when_all_identical(model):
    """Agreement is 1.0 when all samples are identical (top_k=1 forces argmax)."""
    cfg = ConsistencyConfig(n_samples=N_SAMPLES, temperature=1.0, top_k=1)
    det_decoder = ConsistencyDecoder(model, cfg)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    _, score = det_decoder.decode_with_score(ids, MAX_NEW)
    assert score == pytest.approx(1.0)


def test_decoder_decode_with_score_shape(decoder, input_ids):
    """Voted output from decode_with_score should be (B, max_new_tokens)."""
    voted, _ = decoder.decode_with_score(input_ids, MAX_NEW)
    assert voted.shape == (BATCH, MAX_NEW)
