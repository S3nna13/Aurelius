"""Tests for src/eval/ood_detection.py"""

import torch
from aurelius.eval.ood_detection import (
    EnergyScorer,
    KNNScorer,
    MahalanobisScorer,
    MaxSoftmaxScorer,
    OODDetector,
)

SEED = 42
B, C, D = 8, 16, 32  # batch, classes, feature dim


# ---------------------------------------------------------------------------
# MaxSoftmaxScorer
# ---------------------------------------------------------------------------


def test_msp_output_range():
    torch.manual_seed(SEED)
    scorer = MaxSoftmaxScorer()
    logits = torch.randn(B, C)
    scores = scorer.score(logits)
    assert scores.shape == (B,)
    assert (scores >= 0).all() and (scores <= 1).all()


def test_msp_shape_batch1():
    scorer = MaxSoftmaxScorer()
    logits = torch.randn(1, C)
    scores = scorer.score(logits)
    assert scores.shape == (1,)


def test_msp_no_nan():
    scorer = MaxSoftmaxScorer()
    logits = torch.zeros(B, C)
    scores = scorer.score(logits)
    assert torch.isfinite(scores).all()


def test_msp_peaked_high_score():
    scorer = MaxSoftmaxScorer()
    logits = torch.zeros(1, C)
    logits[0, 0] = 100.0  # very peaked
    scores = scorer.score(logits)
    assert scores[0] > 0.99


# ---------------------------------------------------------------------------
# EnergyScorer
# ---------------------------------------------------------------------------


def test_energy_output_finite():
    torch.manual_seed(SEED)
    scorer = EnergyScorer(temperature=1.0)
    logits = torch.randn(B, C)
    scores = scorer.score(logits)
    assert scores.shape == (B,)
    assert torch.isfinite(scores).all()


def test_energy_higher_temp_lower_magnitude():
    torch.manual_seed(SEED)
    logits = torch.randn(B, C)
    s1 = EnergyScorer(temperature=1.0).score(logits)
    s10 = EnergyScorer(temperature=10.0).score(logits)
    # Higher T → log-sum-exp of logits/T is smaller magnitude → less negative
    assert s10.mean() > s1.mean()


def test_energy_default_matches_manual():
    torch.manual_seed(SEED)
    logits = torch.randn(4, C)
    scorer = EnergyScorer(temperature=1.0)
    expected = -torch.logsumexp(logits, dim=-1)
    actual = scorer.score(logits)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_energy_no_nan_zeros():
    scorer = EnergyScorer()
    scores = scorer.score(torch.zeros(B, C))
    assert torch.isfinite(scores).all()


# ---------------------------------------------------------------------------
# MahalanobisScorer
# ---------------------------------------------------------------------------


def test_mahalanobis_fit_score_shape():
    torch.manual_seed(SEED)
    scorer = MahalanobisScorer()
    train_feat = torch.randn(20, D)
    labels = torch.zeros(20, dtype=torch.long)
    scorer.fit(train_feat, labels)
    scores = scorer.score(torch.randn(B, D))
    assert scores.shape == (B,)
    assert torch.isfinite(scores).all()


def test_mahalanobis_in_dist_scores_higher():
    torch.manual_seed(SEED)
    # Two separable classes
    c0 = torch.randn(30, D) + torch.full((D,), 10.0)
    c1 = torch.randn(30, D) - torch.full((D,), 10.0)
    train = torch.cat([c0, c1])
    labels = torch.cat([torch.zeros(30, dtype=torch.long), torch.ones(30, dtype=torch.long)])
    scorer = MahalanobisScorer()
    scorer.fit(train, labels)

    in_dist = torch.randn(5, D) + torch.full((D,), 10.0)
    ood = torch.randn(5, D) * 0.01 + torch.full((D,), 0.0)  # midpoint — far from both clusters

    s_in = scorer.score(in_dist).mean()
    s_ood = scorer.score(ood).mean()
    assert s_in > s_ood


def test_mahalanobis_no_nan_zeros():
    scorer = MahalanobisScorer()
    scorer.fit(torch.randn(10, D))
    scores = scorer.score(torch.zeros(B, D))
    assert torch.isfinite(scores).all()


def test_mahalanobis_singular_cov_no_crash():
    # All identical training samples → singular covariance
    scorer = MahalanobisScorer(reg=1e-5)
    feat = torch.ones(10, D)
    scorer.fit(feat)
    scores = scorer.score(torch.randn(4, D))
    assert torch.isfinite(scores).all()


def test_mahalanobis_batch1():
    scorer = MahalanobisScorer()
    scorer.fit(torch.randn(10, D))
    scores = scorer.score(torch.randn(1, D))
    assert scores.shape == (1,)


# ---------------------------------------------------------------------------
# KNNScorer
# ---------------------------------------------------------------------------


def test_knn_fit_score_shape():
    torch.manual_seed(SEED)
    scorer = KNNScorer(k=3)
    scorer.fit(torch.randn(20, D))
    scores = scorer.score(torch.randn(B, D))
    assert scores.shape == (B,)
    assert torch.isfinite(scores).all()


def test_knn_in_dist_scores_higher():
    torch.manual_seed(SEED)
    train = torch.randn(50, D)
    scorer = KNNScorer(k=3, metric="cosine")
    scorer.fit(train)

    in_dist = train[:5] + torch.randn(5, D) * 0.01  # near training points
    ood = torch.randn(5, D) * 0.01 + torch.full((D,), 100.0)  # far from all training

    s_in = scorer.score(in_dist).mean()
    s_ood = scorer.score(ood).mean()
    assert s_in > s_ood


def test_knn_no_nan_zeros():
    scorer = KNNScorer(k=2)
    scorer.fit(torch.randn(10, D))
    scores = scorer.score(torch.zeros(B, D))
    assert torch.isfinite(scores).all()


def test_knn_k_clamp():
    # k > n_train: should not crash
    scorer = KNNScorer(k=100)
    scorer.fit(torch.randn(3, D))
    scores = scorer.score(torch.randn(4, D))
    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()


def test_knn_l2_metric():
    torch.manual_seed(SEED)
    scorer = KNNScorer(k=3, metric="l2")
    scorer.fit(torch.randn(20, D))
    scores = scorer.score(torch.randn(B, D))
    assert scores.shape == (B,)
    assert torch.isfinite(scores).all()


def test_knn_batch1():
    scorer = KNNScorer(k=2)
    scorer.fit(torch.randn(10, D))
    scores = scorer.score(torch.randn(1, D))
    assert scores.shape == (1,)


# ---------------------------------------------------------------------------
# OODDetector
# ---------------------------------------------------------------------------


def test_detector_predict_shape():
    torch.manual_seed(SEED)
    scorer = MaxSoftmaxScorer()
    detector = OODDetector(scorer, threshold=0.5)
    logits = torch.randn(B, C)
    preds = detector.predict(logits)
    assert preds.shape == (B,)
    assert preds.dtype == torch.bool


def test_detector_clear_in_dist_not_flagged():
    # All-same logit peaked distribution → max softmax near 1 → score >> threshold
    scorer = MaxSoftmaxScorer()
    detector = OODDetector(scorer, threshold=0.5)
    logits = torch.full((4, C), -10.0)
    logits[:, 0] = 10.0  # strongly peaked
    preds = detector.predict(logits)
    assert not preds.any()


def test_detector_clear_ood_flagged():
    # Uniform logits → max softmax = 1/C ≈ 0.0625 (C=16) → score < 0.5 → OOD
    scorer = MaxSoftmaxScorer()
    detector = OODDetector(scorer, threshold=0.5)
    logits = torch.zeros(4, C)
    preds = detector.predict(logits)
    assert preds.all()


def test_detector_determinism():
    torch.manual_seed(SEED)
    scorer = KNNScorer(k=3)
    feat = torch.randn(20, D)
    scorer.fit(feat)
    detector = OODDetector(scorer, threshold=-0.5)

    query = torch.randn(B, D)
    preds1 = detector.predict(query)
    preds2 = detector.predict(query)
    assert (preds1 == preds2).all()
