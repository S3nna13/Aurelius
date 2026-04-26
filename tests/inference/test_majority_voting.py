"""Tests for majority voting / self-consistency decoding (Wang et al., 2022).

Tests cover SelfConsistencyConfig, MajorityVoter, AnswerExtractor, and
SelfConsistencyDecoder with a lightweight MockModel.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.inference.majority_voting import (
    AnswerExtractor,
    MajorityVoter,
    SelfConsistencyConfig,
    SelfConsistencyDecoder,
)

# ---------------------------------------------------------------------------
# MockModel
# ---------------------------------------------------------------------------


class MockModel(nn.Module):
    """Minimal model that returns deterministic logits for testing.

    Forward signature matches AureliusTransformer: returns (hidden, logits, None).
    Logits shape: (batch, seq_len, vocab_size).
    """

    def __init__(self, vocab_size: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        # A simple linear layer to produce logits
        self.proj = nn.Linear(1, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, **kwargs):
        batch, seq_len = input_ids.shape
        # Produce logits: uniform + small random variation seeded by input
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        # Add a slight bias toward token id = 5 for determinism
        logits[:, :, 5] += 1.0
        return (None, logits, None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vocab_size() -> int:
    return 64


@pytest.fixture
def mock_model(vocab_size):
    torch.manual_seed(42)
    return MockModel(vocab_size=vocab_size)


@pytest.fixture
def extractor():
    return AnswerExtractor()


@pytest.fixture
def voter():
    return MajorityVoter(n_samples=8, temperature=0.7, aggregation="plurality")


@pytest.fixture
def decoder(mock_model, voter, extractor):
    return SelfConsistencyDecoder(model=mock_model, voter=voter, extractor=extractor)


@pytest.fixture
def input_ids(vocab_size):
    return torch.randint(1, vocab_size, (1, 4))


# ---------------------------------------------------------------------------
# Test 1: SelfConsistencyConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    """SelfConsistencyConfig should have correct default values."""
    cfg = SelfConsistencyConfig()
    assert cfg.n_samples == 8
    assert cfg.temperature == 0.7
    assert cfg.max_new_tokens == 32
    assert cfg.aggregation == "plurality"


# ---------------------------------------------------------------------------
# Test 2: MajorityVoter.vote returns (winner, stats_dict)
# ---------------------------------------------------------------------------


def test_vote_returns_tuple_with_stats(voter):
    """vote() should return a 2-tuple: (winner, stats_dict)."""
    answers = [1, 2, 1, 3, 1]
    result = voter.vote(answers)
    assert isinstance(result, tuple)
    assert len(result) == 2
    winner, stats = result
    assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# Test 3: Plurality vote picks most common answer
# ---------------------------------------------------------------------------


def test_plurality_vote_picks_most_common(voter):
    """Most frequent answer should win."""
    answers = [7, 7, 3, 7, 3, 1]
    winner, stats = voter.vote(answers)
    assert winner == 7
    assert stats["vote_counts"][7] == 3


# ---------------------------------------------------------------------------
# Test 4: vote with all same answers -> confidence=1.0
# ---------------------------------------------------------------------------


def test_vote_all_same_answers_confidence_one(voter):
    """When all answers are identical confidence must be 1.0."""
    answers = [42, 42, 42, 42]
    winner, stats = voter.vote(answers)
    assert winner == 42
    assert stats["confidence"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Test 5: vote with all different answers -> confidence=1/n
# ---------------------------------------------------------------------------


def test_vote_all_different_confidence_one_over_n(voter):
    """When all answers differ confidence equals 1/n_valid."""
    n = 5
    answers = list(range(n))  # [0, 1, 2, 3, 4] - all distinct
    winner, stats = voter.vote(answers)
    expected_confidence = 1.0 / n
    assert stats["confidence"] == pytest.approx(expected_confidence)


# ---------------------------------------------------------------------------
# Test 6: weighted_vote returns winner with unequal weights
# ---------------------------------------------------------------------------


def test_weighted_vote_unequal_weights(voter):
    """Heavy weight on answer=99 should make it win despite fewer votes."""
    answers = [1, 1, 1, 99]
    weights = [0.1, 0.1, 0.1, 10.0]
    winner = voter.weighted_vote(answers, weights)
    assert winner == 99


# ---------------------------------------------------------------------------
# Test 7: AnswerExtractor.extract returns int or None
# ---------------------------------------------------------------------------


def test_extractor_extract_returns_int_or_none(extractor):
    """extract() should return an int for valid tensors, None for all-padding."""
    # Valid response
    ids = torch.tensor([10, 20, 30], dtype=torch.long)
    result = extractor.extract(ids)
    assert isinstance(result, int) or result is None

    # All-padding -> None
    pad_ids = torch.zeros(5, dtype=torch.long)
    result_pad = extractor.extract(pad_ids)
    assert result_pad is None


# ---------------------------------------------------------------------------
# Test 8: extract_from_candidates returns list of length n_samples
# ---------------------------------------------------------------------------


def test_extract_from_candidates_length(extractor):
    """extract_from_candidates should return a list matching n_samples."""
    n_samples = 6
    seq_len = 8
    candidate_ids = torch.randint(1, 50, (n_samples, seq_len))
    results = extractor.extract_from_candidates(candidate_ids)
    assert isinstance(results, list)
    assert len(results) == n_samples


# ---------------------------------------------------------------------------
# Test 9: SelfConsistencyDecoder.sample_responses returns correct shape
# ---------------------------------------------------------------------------


def test_sample_responses_correct_shape(decoder, input_ids):
    """sample_responses should return tensor of shape (n_samples, max_new_tokens)."""
    n_samples = 4
    max_new_tokens = 6
    samples = decoder.sample_responses(
        input_ids, n_samples=n_samples, max_new_tokens=max_new_tokens, temperature=1.0
    )
    assert samples.shape == (n_samples, max_new_tokens)


# ---------------------------------------------------------------------------
# Test 10: get_confidence returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_get_confidence_in_range(decoder):
    """get_confidence should return a float in [0.0, 1.0]."""
    answers = [5, 5, 3, 5, None, 2]
    confidence = decoder.get_confidence(answers)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# Test 11: decode returns tuple with (answer, dict)
# ---------------------------------------------------------------------------


def test_decode_returns_answer_and_dict(decoder, input_ids):
    """decode() should return a 2-tuple of (Optional[int], dict)."""
    result = decoder.decode(input_ids, max_new_tokens=4)
    assert isinstance(result, tuple)
    assert len(result) == 2
    answer, stats = result
    assert answer is None or isinstance(answer, int)
    assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# Test 12: vote stats have 'n_valid' key
# ---------------------------------------------------------------------------


def test_vote_stats_have_n_valid_key(voter):
    """Stats dict returned by vote() must include 'n_valid'."""
    answers = [1, 2, None, 1, None]
    _, stats = voter.vote(answers)
    assert "n_valid" in stats
    # Two Nones should not be counted
    assert stats["n_valid"] == 3
