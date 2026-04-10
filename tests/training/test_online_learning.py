"""Tests for online_learning — drift-aware adaptive-LR online learning."""
from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.online_learning import (
    OnlineLearningConfig,
    LossWindow,
    DriftDetector,
    AdaptiveLRScheduler,
    OnlineLearner,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


@pytest.fixture
def config():
    return OnlineLearningConfig(
        window_size=10,
        drift_threshold=2.0,
        base_lr=1e-4,
        lr_increase_factor=2.0,
        min_lr=1e-6,
        max_lr=1e-2,
        forgetting_factor=0.99,
    )


@pytest.fixture
def optimizer(small_model):
    return torch.optim.AdamW(small_model.parameters(), lr=1e-4)


@pytest.fixture
def learner(small_model, optimizer, config):
    torch.manual_seed(42)
    return OnlineLearner(small_model, optimizer, config)


def _make_input(batch_size: int = 2, seq_len: int = 8, vocab_size: int = 256) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randint(0, vocab_size, (batch_size, seq_len))


# ---------------------------------------------------------------------------
# 1. OnlineLearningConfig defaults
# ---------------------------------------------------------------------------

def test_online_learning_config_defaults():
    cfg = OnlineLearningConfig()
    assert cfg.window_size == 100
    assert cfg.drift_threshold == 2.0
    assert cfg.base_lr == 1e-4
    assert cfg.lr_increase_factor == 2.0
    assert cfg.min_lr == 1e-6
    assert cfg.max_lr == 1e-2
    assert cfg.forgetting_factor == 0.99


# ---------------------------------------------------------------------------
# 2. LossWindow starts empty
# ---------------------------------------------------------------------------

def test_loss_window_starts_empty():
    w = LossWindow(window_size=10)
    assert len(w) == 0


# ---------------------------------------------------------------------------
# 3. LossWindow.add increases length
# ---------------------------------------------------------------------------

def test_loss_window_add_increases_length():
    w = LossWindow(window_size=10)
    w.add(1.0)
    assert len(w) == 1
    w.add(2.0)
    assert len(w) == 2


# ---------------------------------------------------------------------------
# 4. LossWindow.mean correct average
# ---------------------------------------------------------------------------

def test_loss_window_mean_correct():
    w = LossWindow(window_size=10)
    w.add(1.0)
    w.add(3.0)
    assert w.mean() == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 5. LossWindow.is_full when at capacity
# ---------------------------------------------------------------------------

def test_loss_window_is_full():
    w = LossWindow(window_size=3)
    assert not w.is_full()
    w.add(1.0)
    w.add(2.0)
    assert not w.is_full()
    w.add(3.0)
    assert w.is_full()


# ---------------------------------------------------------------------------
# 6. DriftDetector.update returns bool
# ---------------------------------------------------------------------------

def test_drift_detector_update_returns_bool(config):
    detector = DriftDetector(config)
    result = detector.update(1.0)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# 7. DriftDetector.update no drift when losses stable
# ---------------------------------------------------------------------------

def test_drift_detector_no_drift_stable(config):
    detector = DriftDetector(config)
    # Feed stable losses — should never trigger drift
    any_drift = False
    for _ in range(config.window_size * 2):
        if detector.update(1.0):
            any_drift = True
    assert not any_drift


# ---------------------------------------------------------------------------
# 8. DriftDetector.update detects drift with large loss jump
# ---------------------------------------------------------------------------

def test_drift_detector_detects_drift_large_jump(config):
    detector = DriftDetector(config)
    # Fill reference window with stable low losses
    for _ in range(config.window_size):
        detector.update(0.5)
    # Now feed very high losses to trigger drift
    drift_found = False
    for _ in range(config.window_size):
        if detector.update(100.0):
            drift_found = True
            break
    assert drift_found


# ---------------------------------------------------------------------------
# 9. DriftDetector.drift_score returns float
# ---------------------------------------------------------------------------

def test_drift_detector_drift_score_returns_float(config):
    detector = DriftDetector(config)
    detector.update(1.0)
    score = detector.drift_score()
    assert isinstance(score, float)


# ---------------------------------------------------------------------------
# 10. AdaptiveLRScheduler.on_drift_detected increases LR
# ---------------------------------------------------------------------------

def test_adaptive_lr_on_drift_detected_increases_lr(optimizer, config):
    scheduler = AdaptiveLRScheduler(optimizer, config)
    before = scheduler._current_lr
    scheduler.on_drift_detected()
    assert scheduler._current_lr > before


# ---------------------------------------------------------------------------
# 11. AdaptiveLRScheduler.on_stable decays LR toward base
# ---------------------------------------------------------------------------

def test_adaptive_lr_on_stable_decays_toward_base(optimizer, config):
    scheduler = AdaptiveLRScheduler(optimizer, config)
    # Boost first so LR > base_lr
    scheduler.on_drift_detected()
    boosted = scheduler._current_lr
    scheduler.on_stable()
    assert scheduler._current_lr < boosted


# ---------------------------------------------------------------------------
# 12. AdaptiveLRScheduler LR stays within [min_lr, max_lr]
# ---------------------------------------------------------------------------

def test_adaptive_lr_stays_within_bounds(optimizer, config):
    scheduler = AdaptiveLRScheduler(optimizer, config)
    for _ in range(50):
        scheduler.on_drift_detected()
    assert scheduler._current_lr <= config.max_lr

    # decay many times — should not go below min_lr
    for _ in range(1000):
        scheduler.on_stable()
    assert scheduler._current_lr >= config.min_lr


# ---------------------------------------------------------------------------
# 13. OnlineLearner.train_step returns required keys
# ---------------------------------------------------------------------------

def test_online_learner_train_step_required_keys(learner):
    input_ids = _make_input()
    result = learner.train_step(input_ids)
    for key in ("loss", "drift_detected", "current_lr", "drift_score", "step"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 14. OnlineLearner.train_step loss is positive
# ---------------------------------------------------------------------------

def test_online_learner_train_step_loss_positive(learner):
    input_ids = _make_input()
    result = learner.train_step(input_ids)
    assert result["loss"] > 0.0


# ---------------------------------------------------------------------------
# 15. OnlineLearner.get_stats returns required keys
# ---------------------------------------------------------------------------

def test_online_learner_get_stats_required_keys(learner):
    input_ids = _make_input()
    learner.train_step(input_ids)
    stats = learner.get_stats()
    for key in ("steps", "drift_events", "current_lr", "mean_loss"):
        assert key in stats, f"Missing key: {key}"
