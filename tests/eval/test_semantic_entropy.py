"""
test_semantic_entropy.py -- Tests for semantic_entropy module.

16+ tests covering SemanticClusterer, SemanticEntropy, GenerationSampler,
SetBasedUncertainty, UncertaintyCalibrator, and SemanticEntropyConfig.
"""

import math
import random

import torch
import torch.nn as nn
import pytest

from src.eval.semantic_entropy import (
    SemanticClusterer,
    SemanticEntropy,
    GenerationSampler,
    SetBasedUncertainty,
    UncertaintyCalibrator,
    SemanticEntropyConfig,
)


# ---------------------------------------------------------------------------
# Tiny LM for GenerationSampler tests
# vocab_size=16, d_model=16, N=8 tokens max context
# ---------------------------------------------------------------------------

class TinyLM(nn.Module):
    """Minimal language model: embedding -> linear -> logits."""

    def __init__(self, vocab_size: int = 16, d_model: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T]  -> returns [B, T, V]
        x = self.embed(input_ids)   # [B, T, d]
        return self.proj(x)          # [B, T, V]


def make_tiny_lm() -> TinyLM:
    torch.manual_seed(42)
    return TinyLM(vocab_size=16, d_model=16)


# ---------------------------------------------------------------------------
# SemanticClusterer tests
# ---------------------------------------------------------------------------

def test_jaccard_identical():
    """Identical sequences should have Jaccard similarity 1.0."""
    clusterer = SemanticClusterer()
    seq = [1, 2, 3, 4, 5]
    assert clusterer.jaccard_similarity(seq, seq) == 1.0


def test_jaccard_disjoint():
    """Disjoint sequences should have Jaccard similarity 0.0."""
    clusterer = SemanticClusterer()
    a = [1, 2, 3]
    b = [4, 5, 6]
    assert clusterer.jaccard_similarity(a, b) == 0.0


def test_jaccard_partial_overlap():
    """Partial overlap returns value strictly between 0 and 1."""
    clusterer = SemanticClusterer()
    a = [1, 2, 3]
    b = [2, 3, 4]
    sim = clusterer.jaccard_similarity(a, b)
    assert 0.0 < sim < 1.0


def test_cluster_by_token_overlap_length():
    """cluster_by_token_overlap returns a list of the same length as input."""
    clusterer = SemanticClusterer()
    sequences = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]
    result = clusterer.cluster_by_token_overlap(sequences)
    assert len(result) == len(sequences)


def test_cluster_by_token_overlap_identical_same_cluster():
    """Identical sequences should be assigned to the same cluster."""
    clusterer = SemanticClusterer(similarity_threshold=0.5)
    seq = [10, 20, 30]
    sequences = [seq, list(seq), list(seq)]
    cluster_ids = clusterer.cluster_by_token_overlap(sequences)
    assert cluster_ids[0] == cluster_ids[1] == cluster_ids[2]


def test_cluster_by_token_overlap_disjoint_different_clusters():
    """Fully disjoint sequences should each get a unique cluster."""
    clusterer = SemanticClusterer(similarity_threshold=0.5)
    sequences = [[1, 2], [3, 4], [5, 6]]
    cluster_ids = clusterer.cluster_by_token_overlap(sequences)
    assert len(set(cluster_ids)) == 3


def test_cluster_by_embedding_length():
    """cluster_by_embedding returns a list of the same length as input."""
    clusterer = SemanticClusterer()
    torch.manual_seed(0)
    embeddings = torch.randn(5, 8)
    result = clusterer.cluster_by_embedding(embeddings)
    assert len(result) == 5


def test_cluster_by_embedding_identical_same_cluster():
    """Identical embedding vectors should be in the same cluster."""
    clusterer = SemanticClusterer(similarity_threshold=0.5)
    vec = torch.randn(1, 8)
    embeddings = vec.expand(4, -1).clone()
    cluster_ids = clusterer.cluster_by_embedding(embeddings)
    assert len(set(cluster_ids)) == 1


# ---------------------------------------------------------------------------
# SemanticEntropy tests
# ---------------------------------------------------------------------------

def test_semantic_entropy_compute_positive():
    """compute() returns a positive float for diverse sequences."""
    se = SemanticEntropy()
    sequences = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    log_probs = [-1.0, -1.5, -2.0]
    h = se.compute(sequences, log_probs)
    assert isinstance(h, float)
    assert h >= 0.0


def test_predictive_ge_semantic_entropy():
    """Predictive entropy >= semantic entropy for any input."""
    se = SemanticEntropy()
    sequences = [[1, 2], [3, 4], [1, 2], [5, 6]]
    log_probs = [-1.0, -1.2, -1.1, -1.3]
    pred = se.predictive_entropy(log_probs)
    sem = se.compute(sequences, log_probs)
    assert pred >= sem - 1e-9


def test_excess_entropy_nonnegative():
    """excess_entropy() should always be >= 0."""
    se = SemanticEntropy()
    sequences = [[1, 2], [3, 4], [1, 2]]
    log_probs = [-0.5, -1.0, -0.6]
    excess = se.excess_entropy(sequences, log_probs)
    assert excess >= 0.0


def test_semantic_entropy_identical_sequences_near_zero():
    """Identical sequences cluster into one group -> semantic entropy ~ 0."""
    se = SemanticEntropy()
    seq = [1, 2, 3, 4]
    sequences = [list(seq) for _ in range(5)]
    log_probs = [-1.0] * 5
    h = se.compute(sequences, log_probs)
    assert h < 1e-9


def test_predictive_entropy_uniform():
    """Uniform distribution over n sequences -> entropy = log(n)."""
    se = SemanticEntropy()
    n = 4
    log_probs = [-math.log(n)] * n  # all equal probability 1/n
    h = se.predictive_entropy(log_probs)
    assert abs(h - math.log(n)) < 1e-6


# ---------------------------------------------------------------------------
# GenerationSampler tests
# ---------------------------------------------------------------------------

def test_nucleus_filter_sum_le_one():
    """nucleus_filter output should sum to <= 1 (up to float precision)."""
    torch.manual_seed(1)
    probs = torch.softmax(torch.randn(16), dim=0)
    filtered = GenerationSampler.nucleus_filter(probs, top_p=0.9)
    assert float(filtered.sum().item()) <= 1.0 + 1e-6


def test_nucleus_filter_removes_low_prob_tokens():
    """nucleus_filter with top_p=0.0 (near) should suppress most tokens."""
    # Heavily skewed distribution: one token has nearly all probability
    probs = torch.zeros(16)
    probs[0] = 0.999
    probs[1:] = 0.001 / 15
    # With top_p=0.5, only token 0 should survive
    filtered = GenerationSampler.nucleus_filter(probs, top_p=0.5)
    n_nonzero = int((filtered > 0).sum().item())
    assert n_nonzero == 1


def test_generation_sampler_sample_sequence_returns_tokens_and_logprob():
    """sample_sequence returns (list[int], float) with correct lengths."""
    model = make_tiny_lm()
    sampler = GenerationSampler(model, temperature=1.0, top_p=0.9)
    tokens, lp = sampler.sample_sequence([0, 1, 2], max_new=4)
    assert isinstance(tokens, list)
    assert len(tokens) == 4
    assert isinstance(lp, float)


def test_generation_sampler_sample_n_length():
    """sample_n returns a list of length n."""
    model = make_tiny_lm()
    sampler = GenerationSampler(model, temperature=1.0, top_p=0.9)
    results = sampler.sample_n([0], n=5, max_new=3)
    assert len(results) == 5
    for tokens, lp in results:
        assert isinstance(tokens, list)
        assert isinstance(lp, float)


# ---------------------------------------------------------------------------
# SetBasedUncertainty tests
# ---------------------------------------------------------------------------

def test_answer_set_size_at_least_one():
    """answer_set_size >= 1 for any non-empty sequence list."""
    sbu = SetBasedUncertainty()
    clusterer = SemanticClusterer()
    sequences = [[1, 2], [3, 4]]
    size = sbu.answer_set_size(sequences, clusterer)
    assert size >= 1


def test_confidence_score_in_range():
    """confidence_score should be in (0, 1] for valid log probs."""
    sbu = SetBasedUncertainty()
    log_probs = [-0.3, -0.5, -0.8]
    score = sbu.confidence_score(log_probs)
    assert 0.0 < score <= 1.0


def test_p_true_estimate_weighted():
    """p_true_estimate returns probability-weighted average of verifier scores."""
    sbu = SetBasedUncertainty()
    # Equal log probs -> simple average of verifier scores
    sequences = [[1, 2], [3, 4]]
    log_probs = [math.log(0.5), math.log(0.5)]
    verifier_scores = [1.0, 0.0]
    result = sbu.p_true_estimate(sequences, log_probs, verifier_scores)
    assert abs(result - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# UncertaintyCalibrator tests
# ---------------------------------------------------------------------------

def test_ece_in_range():
    """ECE should be in [0, 1]."""
    cal = UncertaintyCalibrator()
    random.seed(0)
    uncertainties = [random.random() for _ in range(50)]
    labels = [random.randint(0, 1) for _ in range(50)]
    ece = cal.ece(uncertainties, labels, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_calibrate_returns_float_in_range():
    """calibrate() should return a float in [0, 1]."""
    cal = UncertaintyCalibrator()
    preds = [i / 20 for i in range(20)]
    labels = [1 if p < 0.5 else 0 for p in preds]
    cal.fit(preds, labels, n_bins=10)
    for unc in [0.0, 0.25, 0.5, 0.75, 1.0]:
        result = cal.calibrate(unc)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0


def test_calibrator_fit_then_calibrate_unfitted():
    """calibrate() without fit falls back gracefully (identity clamp)."""
    cal = UncertaintyCalibrator()
    result = cal.calibrate(0.7)
    assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# SemanticEntropyConfig tests
# ---------------------------------------------------------------------------

def test_semantic_entropy_config_defaults():
    """SemanticEntropyConfig should have correct default values."""
    cfg = SemanticEntropyConfig()
    assert cfg.n_samples == 10
    assert cfg.temperature == 1.0
    assert cfg.top_p == 0.9
    assert cfg.similarity_threshold == 0.5
    assert cfg.n_bins == 10


def test_semantic_entropy_config_custom():
    """SemanticEntropyConfig should accept custom values."""
    cfg = SemanticEntropyConfig(n_samples=20, temperature=0.7, top_p=0.95)
    assert cfg.n_samples == 20
    assert cfg.temperature == 0.7
    assert cfg.top_p == 0.95
