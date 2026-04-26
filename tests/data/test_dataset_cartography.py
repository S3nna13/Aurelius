"""Tests for src/data/dataset_cartography.py."""

import math

import torch

from src.data.dataset_cartography import (
    CartographyConfig,
    CartographyTracker,
    SampleDynamics,
    compute_forgetting_events,
    select_training_subset,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_CLASSES = 10


def _make_tracker(n_samples: int = 20, n_epochs: int = 3) -> CartographyTracker:
    """Create a CartographyTracker pre-populated with n_epochs of data."""
    torch.manual_seed(42)
    cfg = CartographyConfig(n_epochs=n_epochs)
    tracker = CartographyTracker(n_samples=n_samples, config=cfg)

    indices = list(range(n_samples))
    for _ in range(n_epochs):
        logits = torch.randn(n_samples, N_CLASSES)
        labels = torch.randint(0, N_CLASSES, (n_samples,))
        tracker.record_epoch(indices, logits, labels)
    return tracker


# ---------------------------------------------------------------------------
# 1. SampleDynamics.mean_confidence computes correctly
# ---------------------------------------------------------------------------


def test_sample_dynamics_mean_confidence():
    d = SampleDynamics(sample_idx=0, confidences=[0.2, 0.4, 0.6], correctness=[True, False, True])
    expected = (0.2 + 0.4 + 0.6) / 3
    assert abs(d.mean_confidence - expected) < 1e-6


# ---------------------------------------------------------------------------
# 2. SampleDynamics.variability returns std of confidences
# ---------------------------------------------------------------------------


def test_sample_dynamics_variability():
    confidences = [0.2, 0.4, 0.6]
    d = SampleDynamics(sample_idx=0, confidences=confidences, correctness=[True, False, True])
    mean = sum(confidences) / len(confidences)
    expected_std = math.sqrt(sum((c - mean) ** 2 for c in confidences) / (len(confidences) - 1))
    assert abs(d.variability - expected_std) < 1e-6


# ---------------------------------------------------------------------------
# 3. SampleDynamics.region = "easy-to-learn" for high conf / low var
# ---------------------------------------------------------------------------


def test_sample_dynamics_region_easy_to_learn():
    # high confidence (> 0.7), low variability (<= 0.2)
    d = SampleDynamics(sample_idx=0, confidences=[0.85, 0.88, 0.87], correctness=[True, True, True])
    assert d.mean_confidence > 0.7
    assert d.variability <= 0.2
    assert d.region == "easy-to-learn"


# ---------------------------------------------------------------------------
# 4. SampleDynamics.region = "ambiguous" for high variability
# ---------------------------------------------------------------------------


def test_sample_dynamics_region_ambiguous():
    # alternating between very high and very low confidence → high variability
    d = SampleDynamics(
        sample_idx=0, confidences=[0.05, 0.95, 0.05, 0.95], correctness=[False, True, False, True]
    )
    assert d.variability > 0.2
    assert d.region == "ambiguous"


# ---------------------------------------------------------------------------
# 5. SampleDynamics.region = "hard-to-learn" for low conf / low var
# ---------------------------------------------------------------------------


def test_sample_dynamics_region_hard_to_learn():
    # low confidence (<= 0.7), low variability (<= 0.2)
    d = SampleDynamics(
        sample_idx=0, confidences=[0.1, 0.12, 0.11], correctness=[False, False, False]
    )
    assert d.mean_confidence <= 0.7
    assert d.variability <= 0.2
    assert d.region == "hard-to-learn"


# ---------------------------------------------------------------------------
# 6. CartographyTracker.record_epoch runs without error
# ---------------------------------------------------------------------------


def test_cartography_tracker_record_epoch_no_error():
    torch.manual_seed(0)
    cfg = CartographyConfig(n_epochs=1)
    tracker = CartographyTracker(n_samples=5, config=cfg)
    logits = torch.randn(5, N_CLASSES)
    labels = torch.randint(0, N_CLASSES, (5,))
    tracker.record_epoch(list(range(5)), logits, labels)  # should not raise


# ---------------------------------------------------------------------------
# 7. After record_epoch, get_dynamics returns SampleDynamics
# ---------------------------------------------------------------------------


def test_cartography_tracker_get_dynamics_returns_sample_dynamics():
    torch.manual_seed(1)
    cfg = CartographyConfig(n_epochs=1)
    tracker = CartographyTracker(n_samples=5, config=cfg)
    logits = torch.randn(5, N_CLASSES)
    labels = torch.randint(0, N_CLASSES, (5,))
    tracker.record_epoch(list(range(5)), logits, labels)

    dyn = tracker.get_dynamics(0)
    assert isinstance(dyn, SampleDynamics)
    assert dyn.sample_idx == 0
    assert len(dyn.confidences) == 1
    assert len(dyn.correctness) == 1


# ---------------------------------------------------------------------------
# 8. get_all_dynamics returns list of SampleDynamics
# ---------------------------------------------------------------------------


def test_cartography_tracker_get_all_dynamics():
    tracker = _make_tracker(n_samples=10, n_epochs=2)
    all_dyn = tracker.get_all_dynamics()
    assert isinstance(all_dyn, list)
    assert len(all_dyn) == 10
    assert all(isinstance(d, SampleDynamics) for d in all_dyn)


# ---------------------------------------------------------------------------
# 9. select_by_region returns subset of indices
# ---------------------------------------------------------------------------


def test_cartography_tracker_select_by_region():
    tracker = _make_tracker(n_samples=30, n_epochs=3)
    for region in ("easy-to-learn", "ambiguous", "hard-to-learn"):
        indices = tracker.select_by_region(region)
        assert isinstance(indices, list)
        # all returned indices must be valid sample indices
        for idx in indices:
            assert 0 <= idx < 30


# ---------------------------------------------------------------------------
# 10. cartography_summary returns dict with all 3 regions
# ---------------------------------------------------------------------------


def test_cartography_tracker_cartography_summary_keys():
    tracker = _make_tracker(n_samples=20, n_epochs=3)
    summary = tracker.cartography_summary()
    assert isinstance(summary, dict)
    assert set(summary.keys()) == {"easy-to-learn", "ambiguous", "hard-to-learn"}


def test_cartography_tracker_cartography_summary_counts():
    tracker = _make_tracker(n_samples=20, n_epochs=3)
    summary = tracker.cartography_summary()
    total = sum(summary.values())
    assert total == 20


# ---------------------------------------------------------------------------
# 11. select_training_subset returns list of indices
# ---------------------------------------------------------------------------


def test_select_training_subset_returns_list():
    tracker = _make_tracker(n_samples=30, n_epochs=3)
    result = select_training_subset(tracker, strategy="ambiguous", fraction=0.5)
    assert isinstance(result, list)
    for idx in result:
        assert isinstance(idx, int)


# ---------------------------------------------------------------------------
# 12. select_training_subset returns <= fraction * total count
# ---------------------------------------------------------------------------


def test_select_training_subset_respects_fraction():
    tracker = _make_tracker(n_samples=30, n_epochs=3)
    fraction = 0.33
    total = len(tracker.dynamics)
    budget = max(1, int(fraction * total))

    for strategy in ("easy-to-learn", "ambiguous", "hard-to-learn"):
        result = select_training_subset(tracker, strategy=strategy, fraction=fraction)
        assert len(result) <= budget


# ---------------------------------------------------------------------------
# 13. compute_forgetting_events returns dict with correct counts
# ---------------------------------------------------------------------------


def test_compute_forgetting_events_counts():
    # Manually craft dynamics with known forgetting events
    d0 = SampleDynamics(
        sample_idx=0,
        confidences=[0.9, 0.1, 0.9],
        correctness=[True, False, True],  # 1 forgetting event (T→F)
    )
    d1 = SampleDynamics(
        sample_idx=1,
        confidences=[0.9, 0.9, 0.9],
        correctness=[True, True, True],  # 0 forgetting events
    )
    d2 = SampleDynamics(
        sample_idx=2,
        confidences=[0.8, 0.1, 0.8, 0.1],
        correctness=[True, False, True, False],  # 2 forgetting events
    )
    result = compute_forgetting_events([d0, d1, d2])
    assert isinstance(result, dict)
    assert result[0] == 1
    assert result[1] == 0
    assert result[2] == 2


def test_compute_forgetting_events_keys_match_sample_idx():
    tracker = _make_tracker(n_samples=10, n_epochs=3)
    all_dyn = tracker.get_all_dynamics()
    forgetting = compute_forgetting_events(all_dyn)
    expected_keys = {d.sample_idx for d in all_dyn}
    assert set(forgetting.keys()) == expected_keys


# ---------------------------------------------------------------------------
# SampleDynamics edge cases
# ---------------------------------------------------------------------------


def test_sample_dynamics_single_epoch_variability_zero():
    d = SampleDynamics(sample_idx=0, confidences=[0.5], correctness=[True])
    assert d.variability == 0.0


def test_sample_dynamics_correctness_rate():
    d = SampleDynamics(
        sample_idx=0, confidences=[0.5, 0.5, 0.5, 0.5], correctness=[True, True, False, True]
    )
    assert abs(d.correctness_rate - 0.75) < 1e-6
