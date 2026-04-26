"""Tests for src/data/quality_scorer.py — 16 tests covering all quality scoring components."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.quality_scorer import (
    DatasetQualityScorer,
    PerplexityScorer,
    QualitySignals,
    compute_dedup_score,
    compute_instruction_relevance,
    compute_length_score,
    compute_ngram_novelty,
    hamming_distance,
    simhash,
)

# ---------------------------------------------------------------------------
# MockModel for perplexity tests
# ---------------------------------------------------------------------------


class MockModel(nn.Module):
    """nn.Embedding(256, 16) + nn.Linear(16, 256) -> (loss, logits, None).
    loss = F.cross_entropy of logits[:, :-1] vs input[:, 1:]
    """

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(256, 16)
        self.proj = nn.Linear(16, 256)

    def forward(self, input_ids: torch.Tensor):
        emb = self.embed(input_ids)  # (B, T, 16)
        logits = self.proj(emb)  # (B, T, 256)
        # Shift for language-model loss
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, 256)
        shift_labels = input_ids[:, 1:].contiguous()  # (B, T-1)
        loss = F.cross_entropy(
            shift_logits.view(-1, 256),
            shift_labels.view(-1),
        )
        return loss, logits, None


def make_scorer() -> PerplexityScorer:
    model = MockModel()
    encode_fn = lambda text: [ord(c) % 256 for c in text]  # noqa: E731
    return PerplexityScorer(model=model, encode_fn=encode_fn, device="cpu")


# ---------------------------------------------------------------------------
# 1. compute_length_score — ideal range returns 1.0
# ---------------------------------------------------------------------------


def test_length_score_ideal_range():
    text = "x" * 400  # within [100, 800]
    score = compute_length_score(text)
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. compute_length_score — too short returns < 1.0
# ---------------------------------------------------------------------------


def test_length_score_too_short():
    text = "x" * 10  # below min_len=50
    score = compute_length_score(text)
    assert score < 1.0


# ---------------------------------------------------------------------------
# 3. compute_length_score — too long returns < 1.0
# ---------------------------------------------------------------------------


def test_length_score_too_long():
    text = "x" * 3000  # above max_len=2000
    score = compute_length_score(text)
    assert score < 1.0


# ---------------------------------------------------------------------------
# 4. compute_ngram_novelty — unique text vs empty reference = 1.0
# ---------------------------------------------------------------------------


def test_ngram_novelty_empty_reference():
    text = "the quick brown fox jumps over the lazy dog in the park"
    score = compute_ngram_novelty(text, reference_texts=[], n=4)
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. compute_ngram_novelty — identical text vs reference ~= 0.0
# ---------------------------------------------------------------------------


def test_ngram_novelty_identical_reference():
    text = "the quick brown fox jumps over the lazy dog in the park today"
    score = compute_ngram_novelty(text, reference_texts=[text], n=4)
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. simhash — same text gives same hash (deterministic)
# ---------------------------------------------------------------------------


def test_simhash_deterministic():
    text = "hello world this is a deterministic test"
    h1 = simhash(text)
    h2 = simhash(text)
    assert h1 == h2


# ---------------------------------------------------------------------------
# 7. simhash — different texts give different hashes
# ---------------------------------------------------------------------------


def test_simhash_different_texts():
    h1 = simhash("aaaaaaaaaaaaaaaa bbbbbbb ccccccc ddddddd eeeeeee fffff")
    h2 = simhash("zzzzzzzzzzzzzzzzz qqqqqqq rrrrrrr sssssss ttttttt uuuuuu")
    assert h1 != h2


# ---------------------------------------------------------------------------
# 8. hamming_distance(x, x) = 0
# ---------------------------------------------------------------------------


def test_hamming_distance_same():
    x = simhash("some text here")
    assert hamming_distance(x, x) == 0


# ---------------------------------------------------------------------------
# 9. hamming_distance(0, 2**64-1) = 64
# ---------------------------------------------------------------------------


def test_hamming_distance_all_bits():
    assert hamming_distance(0, 2**64 - 1, n_bits=64) == 64


# ---------------------------------------------------------------------------
# 10. compute_dedup_score — first occurrence returns 1.0
# ---------------------------------------------------------------------------


def test_dedup_score_first_occurrence():
    seen: set = set()
    text = "a completely unique piece of text never seen before in any corpus"
    score, _ = compute_dedup_score(text, seen)
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 11. compute_dedup_score — near-duplicate returns 0.0
# ---------------------------------------------------------------------------


def test_dedup_score_near_duplicate():
    seen: set = set()
    original = "the quick brown fox jumps over the lazy sleeping dog by the river"
    # First insertion
    compute_dedup_score(original, seen)
    # Exact copy should be a duplicate (hamming distance = 0 <= threshold=3)
    score, _ = compute_dedup_score(original, seen)
    assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 12. compute_instruction_relevance — response contains keywords -> high score
# ---------------------------------------------------------------------------


def test_instruction_relevance_high():
    instruction = "Explain how neural networks learn through backpropagation"
    response = "Neural networks learn through backpropagation by computing gradients."
    score = compute_instruction_relevance(instruction, response)
    assert score > 0.5


# ---------------------------------------------------------------------------
# 13. compute_instruction_relevance — unrelated response -> low score
# ---------------------------------------------------------------------------


def test_instruction_relevance_low():
    instruction = "Explain quantum entanglement physics experiments"
    response = "I enjoy cooking pasta carbonara with eggs and cheese."
    score = compute_instruction_relevance(instruction, response)
    assert score < 0.5


# ---------------------------------------------------------------------------
# 14. DatasetQualityScorer.score_example — composite in [0, 1]
# ---------------------------------------------------------------------------


def test_score_example_composite_in_range():
    scorer = DatasetQualityScorer()
    instruction = "Describe photosynthesis in plants"
    response = (
        "Photosynthesis is the process by which plants convert sunlight into energy. "
        "Chlorophyll in leaves absorbs light and uses it to transform carbon dioxide "
        "and water into glucose and oxygen. This is a fundamental biological process "
        "that sustains nearly all life on Earth by producing oxygen and organic matter."
    )
    signals = scorer.score_example(instruction, response)
    assert isinstance(signals, QualitySignals)
    assert 0.0 <= signals.composite <= 1.0


# ---------------------------------------------------------------------------
# 15. DatasetQualityScorer.score_dataset — returns sorted list (lower rank = higher composite)
# ---------------------------------------------------------------------------


def test_score_dataset_sorted_by_rank():
    scorer = DatasetQualityScorer()

    examples = [
        {
            "instruction": "What is machine learning?",
            "response": (
                "Machine learning is a subset of artificial intelligence that enables "
                "systems to learn and improve from experience without being explicitly "
                "programmed. It focuses on developing computer programs that can access "
                "data and use it to learn and improve decision making over time."
            ),
        },
        {
            "instruction": "Describe recursion",
            "response": (
                "Recursion is a programming technique where a function calls itself "
                "to solve smaller instances of the same problem. It requires a base "
                "case to stop the recursion and a recursive case that moves toward "
                "the base case with each call. Classic examples include factorial and Fibonacci."
            ),
        },
        {
            "instruction": "x",
            "response": "y",
        },
    ]

    scored = scorer.score_dataset(examples)

    assert len(scored) == 3
    # Check ranks are assigned sequentially starting from 0
    ranks = [s.rank for s in scored]
    assert sorted(ranks) == list(range(len(scored)))
    # Check sorted by composite descending (rank 0 has highest composite)
    for i in range(len(scored) - 1):
        assert scored[i].signals.composite >= scored[i + 1].signals.composite


# ---------------------------------------------------------------------------
# 16. filter_top_k — returns exactly k examples
# ---------------------------------------------------------------------------


def test_filter_top_k_returns_k():
    scorer = DatasetQualityScorer()

    examples = [
        {
            "instruction": f"Instruction number {i}",
            "response": (
                f"This is a detailed response for instruction number {i}. "
                "It contains enough content to be scored properly by the pipeline. "
                "The response addresses the instruction and provides useful information."
            ),
        }
        for i in range(10)
    ]

    scored = scorer.score_dataset(examples)
    k = 5
    top_k = scorer.filter_top_k(scored, k=k)
    assert len(top_k) == k
