"""Tests for randomized_smoothing.py."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.security.randomized_smoothing import SmoothedClassifier

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

D_MODEL = 16
NUM_CLASSES = 10
SIGMA = 0.1
N_SAMPLES = 50
SEQ_LEN = 3


@pytest.fixture()
def base_model() -> nn.Module:
    """Simple linear classifier: (batch, S, 16) -> (batch, S, 10)."""
    torch.manual_seed(0)
    return nn.Linear(D_MODEL, NUM_CLASSES)


@pytest.fixture()
def smoothed(base_model: nn.Module) -> SmoothedClassifier:
    return SmoothedClassifier(
        model=base_model,
        sigma=SIGMA,
        n_samples=N_SAMPLES,
        device=torch.device("cpu"),
    )


@pytest.fixture()
def x() -> torch.Tensor:
    """Input embedding tensor of shape (1, SEQ_LEN, D_MODEL)."""
    torch.manual_seed(1)
    return torch.randn(1, SEQ_LEN, D_MODEL)


# ---------------------------------------------------------------------------
# Deterministic base model for tests that need a peaked distribution
# ---------------------------------------------------------------------------

class _ConstantLogitModel(nn.Module):
    """Always outputs identical logits that peak on class 0."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        logits = torch.zeros(batch, seq, self.num_classes, device=x.device)
        logits[:, :, 0] = 100.0  # overwhelmingly peak at class 0
        return logits


@pytest.fixture()
def peaked_smoothed() -> SmoothedClassifier:
    return SmoothedClassifier(
        model=_ConstantLogitModel(),
        sigma=SIGMA,
        n_samples=N_SAMPLES,
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# 1. Instantiation
def test_instantiation(smoothed: SmoothedClassifier) -> None:
    assert isinstance(smoothed, SmoothedClassifier)


# 2. _add_noise output shape
def test_add_noise_shape(smoothed: SmoothedClassifier, x: torch.Tensor) -> None:
    noisy = smoothed._add_noise(x, N_SAMPLES)
    assert noisy.shape == (N_SAMPLES, 1, SEQ_LEN, D_MODEL)


# 3. Noise is actually different across samples
def test_add_noise_has_variance(smoothed: SmoothedClassifier, x: torch.Tensor) -> None:
    noisy = smoothed._add_noise(x, N_SAMPLES)
    std_across_samples = noisy.std(dim=0).mean().item()
    assert std_across_samples > 0.0


# 4. _majority_vote returns a 3-tuple
def test_majority_vote_returns_tuple(smoothed: SmoothedClassifier, x: torch.Tensor) -> None:
    noisy = smoothed._add_noise(x, N_SAMPLES)
    batch = noisy.view(N_SAMPLES, SEQ_LEN, D_MODEL)
    with torch.no_grad():
        logits = smoothed.model(batch)
    result = smoothed._majority_vote(logits)
    assert isinstance(result, tuple)
    assert len(result) == 3


# 5. top_count <= total
def test_majority_vote_count_leq_total(smoothed: SmoothedClassifier, x: torch.Tensor) -> None:
    noisy = smoothed._add_noise(x, N_SAMPLES)
    batch = noisy.view(N_SAMPLES, SEQ_LEN, D_MODEL)
    with torch.no_grad():
        logits = smoothed.model(batch)
    top_class, top_count, total = smoothed._majority_vote(logits)
    assert top_count <= total


# 6. top_class is in valid range [0, num_classes)
def test_majority_vote_class_range(smoothed: SmoothedClassifier, x: torch.Tensor) -> None:
    noisy = smoothed._add_noise(x, N_SAMPLES)
    batch = noisy.view(N_SAMPLES, SEQ_LEN, D_MODEL)
    with torch.no_grad():
        logits = smoothed.model(batch)
    top_class, _, _ = smoothed._majority_vote(logits)
    assert 0 <= top_class < NUM_CLASSES


# 7. predict returns int
def test_predict_returns_int(smoothed: SmoothedClassifier, x: torch.Tensor) -> None:
    result = smoothed.predict(x)
    assert isinstance(result, int)


# 8. predict output is in valid class range
def test_predict_class_range(smoothed: SmoothedClassifier, x: torch.Tensor) -> None:
    result = smoothed.predict(x)
    assert 0 <= result < NUM_CLASSES


# 9. certify returns 3-tuple (int, float, bool)
def test_certify_returns_tuple(smoothed: SmoothedClassifier, x: torch.Tensor) -> None:
    result = smoothed.certify(x)
    assert isinstance(result, tuple)
    assert len(result) == 3
    predicted_class, certified_radius, abstain = result
    assert isinstance(predicted_class, int)
    assert isinstance(certified_radius, float)
    assert isinstance(abstain, bool)


# 10. certified_radius >= 0
def test_certify_radius_nonnegative(smoothed: SmoothedClassifier, x: torch.Tensor) -> None:
    _, certified_radius, _ = smoothed.certify(x)
    assert certified_radius >= 0.0


# 11. abstain is bool
def test_certify_abstain_is_bool(smoothed: SmoothedClassifier, x: torch.Tensor) -> None:
    _, _, abstain = smoothed.certify(x)
    assert isinstance(abstain, bool)


# 12. All samples agree -> abstain=False
def test_certify_no_abstain_on_peaked_model(peaked_smoothed: SmoothedClassifier, x: torch.Tensor) -> None:
    predicted_class, certified_radius, abstain = peaked_smoothed.certify(x)
    assert abstain is False
    assert predicted_class == 0


# 13. certified_radius > 0 when not abstaining on peaked distribution
def test_certify_positive_radius_on_peaked_model(peaked_smoothed: SmoothedClassifier, x: torch.Tensor) -> None:
    _, certified_radius, abstain = peaked_smoothed.certify(x)
    if not abstain:
        assert certified_radius > 0.0


# 14. Works with seq_len=4
def test_certify_seq_len_4(smoothed: SmoothedClassifier) -> None:
    torch.manual_seed(42)
    x4 = torch.randn(1, 4, D_MODEL)
    predicted_class, certified_radius, abstain = smoothed.certify(x4)
    assert 0 <= predicted_class < NUM_CLASSES
    assert certified_radius >= 0.0
    assert isinstance(abstain, bool)
