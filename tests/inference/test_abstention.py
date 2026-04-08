"""Tests for src/inference/abstention.py — confidence estimation and selective prediction."""

import math
import pytest
import torch
import torch.nn as nn

from src.inference.abstention import (
    ConfidenceEstimator,
    MCDropoutEstimator,
    SelectivePredictor,
    SemanticUncertainty,
)


# ---------------------------------------------------------------------------
# Mock model for MCDropoutEstimator tests
# ---------------------------------------------------------------------------

class MockModel(nn.Module):
    def __init__(self, vocab=32):
        super().__init__()
        self.linear = nn.Linear(8, vocab)
        self.drop = nn.Dropout(0.5)

    def __call__(self, input_ids):
        x = torch.randn(input_ids.shape[0], input_ids.shape[1], 8)
        return (torch.tensor(0.0), self.drop(self.linear(x)), None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB = 64
B, S = 2, 16


@pytest.fixture
def estimator():
    return ConfidenceEstimator(model=None, threshold=0.5)


@pytest.fixture
def uniform_logits():
    """Perfectly uniform distribution over VOCAB tokens."""
    return torch.zeros(B, S, VOCAB)


@pytest.fixture
def peaked_logits():
    """Near-one-hot: one token gets logit 100, rest 0."""
    logits = torch.zeros(B, S, VOCAB)
    logits[:, :, 0] = 100.0
    return logits


@pytest.fixture
def mock_model():
    torch.manual_seed(42)
    return MockModel(vocab=32)


@pytest.fixture
def mc_estimator(mock_model):
    return MCDropoutEstimator(mock_model, n_samples=10)


# ---------------------------------------------------------------------------
# ConfidenceEstimator tests
# ---------------------------------------------------------------------------

def test_token_entropy_shape(estimator, uniform_logits):
    """token_entropy should return shape (B, S)."""
    entropy = estimator.token_entropy(uniform_logits)
    assert entropy.shape == (B, S), f"Expected ({B}, {S}), got {entropy.shape}"


def test_token_entropy_range(estimator, uniform_logits):
    """Entropy values must be non-negative."""
    entropy = estimator.token_entropy(uniform_logits)
    assert (entropy >= 0).all(), "Entropy must be >= 0 everywhere"


def test_uniform_distribution_max_entropy(estimator, uniform_logits):
    """Uniform distribution should have entropy equal to log(V)."""
    entropy = estimator.token_entropy(uniform_logits)
    expected = math.log(VOCAB)
    assert torch.allclose(
        entropy, torch.full_like(entropy, expected), atol=1e-4
    ), f"Expected uniform entropy ~{expected:.4f}, got {entropy[0, 0].item():.4f}"


def test_peaked_distribution_low_entropy(estimator, peaked_logits):
    """Near-one-hot logits should produce near-zero entropy."""
    entropy = estimator.token_entropy(peaked_logits)
    assert (entropy < 0.01).all(), f"Expected near-zero entropy, got max {entropy.max().item()}"


def test_sequence_confidence_range(estimator, uniform_logits):
    """sequence_confidence must return values in [0, 1]."""
    conf = estimator.sequence_confidence(uniform_logits)
    assert conf.shape == (B,)
    assert (conf >= 0.0).all() and (conf <= 1.0).all(), f"Confidence out of range: {conf}"


def test_should_abstain_high_entropy(estimator, uniform_logits):
    """Uniform distribution -> very low max-prob -> should_abstain=True."""
    # With uniform logits over 64 tokens, max_prob = 1/64 ≈ 0.016 << threshold=0.5
    result = estimator.should_abstain(uniform_logits)
    assert result.shape == (B,)
    assert result.all(), "Model should abstain on fully uniform (maximally uncertain) logits"


def test_top_k_confidence_range(estimator, uniform_logits):
    """top_k_confidence must return values in [0, 1]."""
    topk = estimator.top_k_confidence(uniform_logits, k=5)
    assert topk.shape == (B, S)
    assert (topk >= 0.0).all() and (topk <= 1.0 + 1e-6).all(), f"top-k confidence out of range"


# ---------------------------------------------------------------------------
# MCDropoutEstimator tests
# ---------------------------------------------------------------------------

def test_mc_dropout_returns_dict(mc_estimator):
    """estimate() must return a dict with the required keys."""
    input_ids = torch.randint(0, 10, (1, 8))
    result = mc_estimator.estimate(input_ids)
    required_keys = {"mean_logits", "variance", "mean_entropy", "predictive_entropy", "mutual_information"}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )


def test_mc_dropout_mutual_info_nonneg(mc_estimator):
    """mutual_information = predictive_entropy - mean_entropy must be >= 0."""
    input_ids = torch.randint(0, 10, (2, 6))
    result = mc_estimator.estimate(input_ids)
    mi = result["mutual_information"]
    assert (mi >= 0).all(), f"mutual_information has negative values: {mi.min().item()}"


# ---------------------------------------------------------------------------
# SelectivePredictor tests
# ---------------------------------------------------------------------------

def _make_selective_predictor(threshold):
    estimator = ConfidenceEstimator(model=None, threshold=threshold)
    return SelectivePredictor(estimator)


def test_selective_predictor_abstains():
    """Low confidence (uniform logits) below threshold -> abstained=True."""
    predictor = _make_selective_predictor(threshold=0.5)
    # Uniform logits -> max_prob = 1/VOCAB << 0.5
    logits = torch.zeros(1, 8, VOCAB)
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    generated_ids = torch.zeros(1, 8, dtype=torch.long)

    result = predictor.predict(input_ids, generated_ids, logits)
    assert result["abstained"] is True
    assert result["response_ids"] is None
    assert result["message"] is not None
    assert 0.0 <= result["confidence"] <= 1.0


def test_selective_predictor_answers():
    """High confidence (peaked logits) above threshold -> abstained=False."""
    predictor = _make_selective_predictor(threshold=0.5)
    # Near-one-hot -> max_prob ≈ 1.0 >> 0.5
    logits = torch.zeros(1, 8, VOCAB)
    logits[:, :, 0] = 100.0
    input_ids = torch.zeros(1, 4, dtype=torch.long)
    generated_ids = torch.zeros(1, 8, dtype=torch.long)

    result = predictor.predict(input_ids, generated_ids, logits)
    assert result["abstained"] is False
    assert result["response_ids"] is not None
    assert result["message"] is None


def test_abstention_rate():
    """abstention_rate should equal n_abstained / (n_abstained + n_answered)."""
    predictor = _make_selective_predictor(threshold=0.5)

    uniform_logits = torch.zeros(1, 8, VOCAB)          # will abstain
    peaked_logits = torch.zeros(1, 8, VOCAB)            # will answer
    peaked_logits[:, :, 0] = 100.0

    input_ids = torch.zeros(1, 4, dtype=torch.long)
    gen_ids = torch.zeros(1, 8, dtype=torch.long)

    # 2 abstentions
    predictor.predict(input_ids, gen_ids, uniform_logits)
    predictor.predict(input_ids, gen_ids, uniform_logits)
    # 1 answer
    predictor.predict(input_ids, gen_ids, peaked_logits)

    rate = predictor.abstention_rate()
    assert abs(rate - 2 / 3) < 1e-6, f"Expected rate 2/3, got {rate}"

    predictor.reset_stats()
    assert predictor.abstention_rate() == 0.0


# ---------------------------------------------------------------------------
# SemanticUncertainty tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sem():
    return SemanticUncertainty(n_samples=5, similarity_threshold=0.5)


def test_jaccard_identical(sem):
    """Jaccard similarity of a sequence with itself is 1.0."""
    assert sem.jaccard_similarity([1, 2, 3], [1, 2, 3]) == 1.0


def test_jaccard_disjoint(sem):
    """Jaccard similarity of completely disjoint sequences is 0.0."""
    assert sem.jaccard_similarity([1, 2], [3, 4]) == 0.0


def test_semantic_entropy_uniform(sem):
    """All responses in different clusters => maximum entropy = log(n_clusters)."""
    # Each response is unique and far from others -> each in its own cluster
    clusters = [[0], [1], [2], [3], [4]]  # 5 singleton clusters
    n_total = 5
    entropy = sem.semantic_entropy(clusters, n_total)
    expected = math.log(5)
    assert abs(entropy - expected) < 1e-6, f"Expected {expected:.4f}, got {entropy:.4f}"


def test_semantic_entropy_single_cluster(sem):
    """All responses in one cluster => entropy == 0."""
    clusters = [[0, 1, 2, 3, 4]]  # all 5 in one cluster
    n_total = 5
    entropy = sem.semantic_entropy(clusters, n_total)
    assert abs(entropy) < 1e-9, f"Expected 0.0, got {entropy}"
