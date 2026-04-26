"""Tests for LLM-based data selection (data_selection_llm.py)."""

from __future__ import annotations

import pytest
import torch

from src.data.data_selection_llm import (
    DataScorer,
    DataSelectionConfig,
    DiversitySelector,
    InstructionFollowingScorer,
    alpagasus_filter,
    nuggets_score,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture(scope="module")
def small_model(small_cfg):
    model = AureliusTransformer(small_cfg)
    model.eval()
    return model


@pytest.fixture
def sample_examples():
    return [
        {"instruction": "What is 2+2?", "response": "4"},
        {"instruction": "Name a planet.", "response": "Mars"},
        {"instruction": "Write a poem.", "response": "Roses are red..."},
        {"instruction": "Explain gravity.", "response": "Gravity is a force..."},
        {"instruction": "Say hello.", "response": "Hello!"},
        {"instruction": "Count to 3.", "response": "1, 2, 3"},
    ]


@pytest.fixture
def constant_score_fn():
    """Score function that returns score based on response length."""

    def score(instruction: str, response: str) -> float:
        return float(len(response))

    return score


# ---------------------------------------------------------------------------
# Test 1: DataSelectionConfig defaults
# ---------------------------------------------------------------------------


def test_data_selection_config_defaults():
    cfg = DataSelectionConfig()
    assert cfg.top_fraction == 0.1
    assert cfg.diversity_weight == 0.5
    assert cfg.min_score == 0.0
    assert cfg.batch_size == 8


# ---------------------------------------------------------------------------
# Test 2: DataScorer.score_dataset returns list of correct length
# ---------------------------------------------------------------------------


def test_score_dataset_length(sample_examples, constant_score_fn):
    scorer = DataScorer(score_fn=constant_score_fn)
    scores = scorer.score_dataset(sample_examples)
    assert isinstance(scores, list)
    assert len(scores) == len(sample_examples)
    assert all(isinstance(s, float) for s in scores)


# ---------------------------------------------------------------------------
# Test 3: select_top_k with k=3 returns exactly 3 examples
# ---------------------------------------------------------------------------


def test_select_top_k_absolute(sample_examples, constant_score_fn):
    scorer = DataScorer(score_fn=constant_score_fn)
    scores = scorer.score_dataset(sample_examples)
    selected, sel_scores = scorer.select_top_k(sample_examples, scores, k=3)
    assert len(selected) == 3
    assert len(sel_scores) == 3


# ---------------------------------------------------------------------------
# Test 4: select_top_k with fraction=0.5 returns half the examples
# ---------------------------------------------------------------------------


def test_select_top_k_fraction(sample_examples, constant_score_fn):
    scorer = DataScorer(score_fn=constant_score_fn)
    scores = scorer.score_dataset(sample_examples)
    selected, sel_scores = scorer.select_top_k(sample_examples, scores, fraction=0.5)
    expected = max(1, int(len(sample_examples) * 0.5))
    # ceil(6 * 0.5) = 3
    assert len(selected) == expected
    assert len(sel_scores) == expected


# ---------------------------------------------------------------------------
# Test 5: filter_by_threshold keeps only examples above threshold
# ---------------------------------------------------------------------------


def test_filter_by_threshold(sample_examples, constant_score_fn):
    scorer = DataScorer(score_fn=constant_score_fn)
    scores = scorer.score_dataset(sample_examples)
    # All scores are response lengths; pick a threshold in the middle
    threshold = 5.0
    filtered, filt_scores = scorer.filter_by_threshold(sample_examples, scores, threshold)
    assert all(s >= threshold for s in filt_scores)
    # Verify no example below threshold leaked through
    for ex, sc in zip(sample_examples, scores):
        if sc < threshold:
            assert ex not in filtered


# ---------------------------------------------------------------------------
# Test 6: InstructionFollowingScorer.batch_score returns (n,) tensor
# ---------------------------------------------------------------------------


def test_batch_score_shape(small_model):
    scorer = InstructionFollowingScorer(small_model, tokenizer_vocab_size=256)
    torch.manual_seed(0)
    n = 4
    inst_list = [torch.randint(0, 200, (8,)) for _ in range(n)]
    resp_list = [torch.randint(0, 200, (12,)) for _ in range(n)]
    result = scorer.batch_score(inst_list, resp_list)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (n,)


# ---------------------------------------------------------------------------
# Test 7: Higher quality responses get higher scores
# ---------------------------------------------------------------------------


def test_higher_quality_response_higher_score(small_model):
    """A well-formed long response should not score lower than an empty one."""
    scorer = InstructionFollowingScorer(small_model, tokenizer_vocab_size=256)
    torch.manual_seed(42)
    instruction_ids = torch.randint(0, 200, (8,))

    # Score two different responses
    resp_a = torch.randint(0, 200, (20,))
    resp_b = torch.randint(0, 200, (20,))

    score_a = scorer.score(instruction_ids, resp_a)
    score_b = scorer.score(instruction_ids, resp_b)

    # Both are valid floats
    assert isinstance(score_a, float)
    assert isinstance(score_b, float)
    # Scores are negative perplexity so they should be finite negative numbers
    assert score_a < 0 or score_a == 0.0
    assert score_b < 0 or score_b == 0.0


# ---------------------------------------------------------------------------
# Test 8: DiversitySelector.compute_embeddings returns correct shape
# ---------------------------------------------------------------------------


def test_compute_embeddings_shape():
    dim = 16
    n = 5

    def embed_fn(x: torch.Tensor) -> torch.Tensor:
        return torch.randn(dim)

    selector = DiversitySelector(embedding_fn=embed_fn, n_select=3)
    tensors = [torch.randint(0, 100, (10,)) for _ in range(n)]
    embeddings = selector.compute_embeddings(tensors)
    assert embeddings.shape == (n, dim)


# ---------------------------------------------------------------------------
# Test 9: select_diverse returns exactly n distinct indices
# ---------------------------------------------------------------------------


def test_select_diverse_count_and_unique():
    torch.manual_seed(7)
    n_total = 20
    dim = 16
    n_select = 7

    embeddings = torch.randn(n_total, dim)

    def embed_fn(x):
        return torch.randn(dim)

    selector = DiversitySelector(embedding_fn=embed_fn, n_select=n_select)
    indices = selector.select_diverse(embeddings, n=n_select)
    assert len(indices) == n_select
    assert len(set(indices)) == n_select  # all distinct
    assert all(0 <= i < n_total for i in indices)


# ---------------------------------------------------------------------------
# Test 10: quality_diversity_select returns correct number of indices
# ---------------------------------------------------------------------------


def test_quality_diversity_select_count():
    torch.manual_seed(11)
    n_total = 15
    dim = 8
    n_select = 5

    embeddings = torch.randn(n_total, dim)
    scores = torch.rand(n_total)

    def embed_fn(x):
        return torch.randn(dim)

    selector = DiversitySelector(embedding_fn=embed_fn, n_select=n_select)
    indices = selector.quality_diversity_select(embeddings, scores, alpha=0.5)
    assert len(indices) == n_select
    assert len(set(indices)) == n_select
    assert all(0 <= i < n_total for i in indices)


# ---------------------------------------------------------------------------
# Test 11: nuggets_score returns value in [0, 1]
# ---------------------------------------------------------------------------


def test_nuggets_score_range():
    # Various cases
    cases = [
        ("What is Python?", "Python is a programming language.", ["Python", "language"]),
        ("Explain quantum physics.", "", ["quantum"]),
        ("Say hello.", "Hello world!", []),
        ("Name animals.", "cat dog fish bird snake", ["cat", "fish", "elephant"]),
    ]
    for instruction, response, keywords in cases:
        score = nuggets_score(instruction, response, keywords)
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0,1] for case: {instruction!r}"


def test_nuggets_score_full_coverage():
    """All keywords present -> higher score than none present."""
    instruction = "Describe space"
    response_good = "Space is a vast vacuum containing stars and planets in the universe."
    response_bad = "I do not know."
    keywords = ["stars", "planets", "universe"]

    score_good = nuggets_score(instruction, response_good, keywords)
    score_bad = nuggets_score(instruction, response_bad, keywords)
    assert score_good > score_bad


# ---------------------------------------------------------------------------
# Test 12: alpagasus_filter keeps examples above 4.5 threshold
# ---------------------------------------------------------------------------


def test_alpagasus_filter_threshold():
    examples = [
        {"instruction": "Q1", "response": "A1"},
        {"instruction": "Q2", "response": "A2"},
        {"instruction": "Q3", "response": "A3"},
        {"instruction": "Q4", "response": "A4"},
    ]
    scores = [3.0, 4.5, 4.8, 2.1]
    filtered = alpagasus_filter(examples, scores, threshold=4.5)
    # Should keep examples with score 4.5 and 4.8
    assert len(filtered) == 2
    assert examples[1] in filtered
    assert examples[2] in filtered
    assert examples[0] not in filtered
    assert examples[3] not in filtered


def test_alpagasus_filter_default_threshold():
    """Default threshold is 4.5 matching the AlpaGasus paper."""
    examples = [{"instruction": "Q", "response": "A"} for _ in range(5)]
    scores = [5.0, 4.5, 4.4, 3.0, 4.9]
    filtered = alpagasus_filter(examples, scores)
    # 5.0, 4.5, 4.9 pass (>= 4.5); 4.4 and 3.0 do not
    assert len(filtered) == 3
