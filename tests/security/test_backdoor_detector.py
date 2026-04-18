"""Tests for the STRIP backdoor/trojan detection module."""

from __future__ import annotations

import math

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.security.backdoor_detector import STRIPDetector

# ---------------------------------------------------------------------------
# Shared tiny config and fixtures
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

SEQ_LEN = 8
N_REFS = 10
SEED = 42


@pytest.fixture(scope="module")
def model() -> AureliusTransformer:
    torch.manual_seed(SEED)
    m = AureliusTransformer(TINY_CFG)
    m.eval()
    return m


@pytest.fixture(scope="module")
def detector(model) -> STRIPDetector:
    return STRIPDetector(model, n_perturbations=8, entropy_threshold=1.0)


@pytest.fixture(scope="module")
def input_ids() -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randint(0, TINY_CFG.vocab_size, (1, SEQ_LEN))


@pytest.fixture(scope="module")
def reference_pool() -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    return torch.randint(0, TINY_CFG.vocab_size, (N_REFS, SEQ_LEN))


# ---------------------------------------------------------------------------
# _prediction_entropy tests
# ---------------------------------------------------------------------------


def test_entropy_is_positive_scalar(detector):
    """_prediction_entropy returns a positive scalar for random logits."""
    torch.manual_seed(0)
    logits = torch.randn(1, SEQ_LEN, TINY_CFG.vocab_size)
    h = detector._prediction_entropy(logits)
    assert isinstance(h, float)
    assert h > 0.0


def test_entropy_uniform_approx_log_vocab(detector):
    """Entropy of a uniform distribution over vocab_size is close to log(vocab_size)."""
    uniform_logits = torch.zeros(1, 1, TINY_CFG.vocab_size)
    h = detector._prediction_entropy(uniform_logits)
    expected = math.log(TINY_CFG.vocab_size)
    assert abs(h - expected) < 0.01, f"Expected ~{expected:.4f}, got {h:.4f}"


def test_entropy_peaked_approx_zero(detector):
    """Entropy of a near-deterministic distribution is close to 0."""
    logits = torch.full((1, 1, TINY_CFG.vocab_size), -1e6)
    logits[0, 0, 0] = 1e6  # all probability mass on token 0
    h = detector._prediction_entropy(logits)
    assert h < 0.01, f"Expected near-zero entropy, got {h:.6f}"


# ---------------------------------------------------------------------------
# _superimpose tests
# ---------------------------------------------------------------------------


def test_superimpose_returns_same_shape(detector):
    """_superimpose output has the same shape as both inputs."""
    torch.manual_seed(1)
    a = torch.randint(0, TINY_CFG.vocab_size, (1, SEQ_LEN))
    b = torch.randint(0, TINY_CFG.vocab_size, (1, SEQ_LEN))
    out = detector._superimpose(a, b)
    assert out.shape == a.shape


def test_superimpose_tokens_from_union(detector):
    """Every token in the superimposed output comes from input_a or input_b."""
    torch.manual_seed(2)
    # Use disjoint value ranges to make membership unambiguous
    a = torch.randint(0, 50, (1, 32))
    b = torch.randint(100, 150, (1, 32))
    out = detector._superimpose(a, b)
    # Each position must match either a or b at that position
    from_a = (out == a)
    from_b = (out == b)
    assert (from_a | from_b).all(), "Some tokens came from neither input"


# ---------------------------------------------------------------------------
# score tests
# ---------------------------------------------------------------------------


def test_score_returns_float(detector, input_ids, reference_pool):
    """score() returns a Python float."""
    s = detector.score(input_ids, reference_pool)
    assert isinstance(s, float)


def test_score_is_non_negative(detector, input_ids, reference_pool):
    """score() is non-negative (entropy >= 0)."""
    s = detector.score(input_ids, reference_pool)
    assert s >= 0.0


# ---------------------------------------------------------------------------
# is_trojan tests
# ---------------------------------------------------------------------------


def test_is_trojan_returns_bool(detector, input_ids, reference_pool):
    """is_trojan() returns a Python bool."""
    result = detector.is_trojan(input_ids, reference_pool)
    assert isinstance(result, bool)


def test_is_trojan_high_threshold_returns_true(model, input_ids, reference_pool):
    """With a very high threshold, any input is flagged as trojaned."""
    high_thresh_detector = STRIPDetector(
        model, n_perturbations=4, entropy_threshold=1e9
    )
    assert high_thresh_detector.is_trojan(input_ids, reference_pool) is True


def test_is_trojan_low_threshold_returns_false(model, input_ids, reference_pool):
    """With a threshold of 0, no input is flagged as trojaned."""
    low_thresh_detector = STRIPDetector(
        model, n_perturbations=4, entropy_threshold=0.0
    )
    assert low_thresh_detector.is_trojan(input_ids, reference_pool) is False


# ---------------------------------------------------------------------------
# batch_score tests
# ---------------------------------------------------------------------------


def test_batch_score_returns_correct_length(detector, reference_pool):
    """batch_score returns a list whose length matches the batch size."""
    torch.manual_seed(3)
    batch_size = 4
    batch_ids = torch.randint(0, TINY_CFG.vocab_size, (batch_size, SEQ_LEN))
    scores = detector.batch_score(batch_ids, reference_pool)
    assert len(scores) == batch_size


def test_batch_score_elements_are_floats(detector, reference_pool):
    """Every element returned by batch_score is a Python float."""
    torch.manual_seed(4)
    batch_ids = torch.randint(0, TINY_CFG.vocab_size, (3, SEQ_LEN))
    scores = detector.batch_score(batch_ids, reference_pool)
    for i, s in enumerate(scores):
        assert isinstance(s, float), f"Element {i} has type {type(s)}"


# ---------------------------------------------------------------------------
# Integration: model runs without error on superimposed inputs
# ---------------------------------------------------------------------------


def test_model_runs_on_superimposed_inputs(model, input_ids, reference_pool):
    """Model forward pass completes without error on superimposed token sequences."""
    detector_local = STRIPDetector(model, n_perturbations=2, entropy_threshold=1.0)
    torch.manual_seed(5)
    ref = reference_pool[0:1]
    mixed = detector_local._superimpose(input_ids, ref)
    assert mixed.shape == input_ids.shape
    with torch.no_grad():
        loss, logits, pkv = model(mixed)
    assert logits.shape == (1, SEQ_LEN, TINY_CFG.vocab_size)
