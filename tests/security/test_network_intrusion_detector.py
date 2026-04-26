"""Tests for the network intrusion detector module."""

from __future__ import annotations

import pytest
import torch
import torch.optim as optim

from src.security.network_intrusion_detector import (
    IntrusionConfig,
    IntrusionDetector,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg() -> IntrusionConfig:
    return IntrusionConfig(
        n_features=8,
        d_model=32,
        n_heads=2,
        n_layers=1,
        n_classes=2,
        max_seq_len=16,
        dropout=0.0,
    )


@pytest.fixture
def detector(cfg: IntrusionConfig) -> IntrusionDetector:
    torch.manual_seed(0)
    return IntrusionDetector(cfg)


@pytest.fixture
def x(cfg: IntrusionConfig) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(2, 8, cfg.n_features)  # B=2, T=8


@pytest.fixture
def labels() -> torch.Tensor:
    return torch.tensor([0, 1])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_instantiates(detector: IntrusionDetector) -> None:
    assert detector is not None


def test_forward_shape(detector: IntrusionDetector, x: torch.Tensor, cfg: IntrusionConfig) -> None:
    logits = detector.forward(x)
    assert logits.shape == (2, cfg.n_classes)


def test_forward_finite(detector: IntrusionDetector, x: torch.Tensor) -> None:
    logits = detector.forward(x)
    assert torch.isfinite(logits).all()


def test_predict_shape(detector: IntrusionDetector, x: torch.Tensor) -> None:
    preds = detector.predict(x)
    assert preds.shape == (2,)


def test_predict_dtype(detector: IntrusionDetector, x: torch.Tensor) -> None:
    preds = detector.predict(x)
    assert preds.dtype == torch.long


def test_predict_in_range(
    detector: IntrusionDetector, x: torch.Tensor, cfg: IntrusionConfig
) -> None:
    preds = detector.predict(x)
    assert (preds >= 0).all() and (preds < cfg.n_classes).all()


def test_anomaly_score_shape(detector: IntrusionDetector, x: torch.Tensor) -> None:
    scores = detector.anomaly_score(x)
    assert scores.shape == (2,)


def test_anomaly_score_in_0_1(detector: IntrusionDetector, x: torch.Tensor) -> None:
    scores = detector.anomaly_score(x)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_anomaly_score_sums_to_one(detector: IntrusionDetector, x: torch.Tensor) -> None:
    import torch.nn.functional as F

    logits = detector.forward(x)
    probs = F.softmax(logits.detach(), dim=-1)
    class0 = probs[:, 0]
    class1 = detector.anomaly_score(x)
    total = class0 + class1
    assert torch.allclose(total, torch.ones(2), atol=1e-5)


def test_train_step_returns_float(
    detector: IntrusionDetector,
    x: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    optimizer = optim.Adam(detector.model.parameters(), lr=1e-3)
    loss = detector.train_step(x, labels, optimizer)
    assert isinstance(loss, float)


def test_train_step_loss_positive_finite(
    detector: IntrusionDetector,
    x: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    optimizer = optim.Adam(detector.model.parameters(), lr=1e-3)
    loss = detector.train_step(x, labels, optimizer)
    assert loss > 0.0
    assert not (loss != loss)  # not NaN


def test_train_step_updates_params(
    detector: IntrusionDetector,
    x: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    optimizer = optim.Adam(detector.model.parameters(), lr=1e-2)
    params_before = [p.clone() for p in detector.model.parameters()]
    detector.train_step(x, labels, optimizer)
    params_after = list(detector.model.parameters())
    changed = any(
        not torch.equal(before, after) for before, after in zip(params_before, params_after)
    )
    assert changed


def test_batch_size_one(cfg: IntrusionConfig) -> None:
    torch.manual_seed(2)
    detector = IntrusionDetector(cfg)
    x_single = torch.randn(1, 8, cfg.n_features)
    logits = detector.forward(x_single)
    assert logits.shape == (1, cfg.n_classes)
    assert torch.isfinite(logits).all()


def test_single_token_sequence(cfg: IntrusionConfig) -> None:
    torch.manual_seed(3)
    detector = IntrusionDetector(cfg)
    x_single_tok = torch.randn(2, 1, cfg.n_features)
    logits = detector.forward(x_single_tok)
    assert logits.shape == (2, cfg.n_classes)
    assert torch.isfinite(logits).all()


def test_config_fields() -> None:
    cfg = IntrusionConfig(
        n_features=8,
        d_model=32,
        n_heads=2,
        n_layers=1,
        n_classes=2,
        max_seq_len=16,
        dropout=0.0,
    )
    assert cfg.n_features == 8
    assert cfg.d_model == 32
    assert cfg.n_heads == 2
    assert cfg.n_layers == 1
    assert cfg.n_classes == 2
    assert cfg.max_seq_len == 16
    assert cfg.dropout == 0.0
