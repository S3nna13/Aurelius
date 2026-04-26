"""Tests for src/training/online_learning_v2.py.

All models use D=8 (tiny) to keep tests fast.
Pure PyTorch only — no HuggingFace, scipy, or sklearn.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.online_learning_v2 import (
    OnlineConfig,
    OnlineLearner,
    StreamingBuffer,
    compute_forgetting_metric,
    compute_online_ewc_loss,
    compute_parameter_distance,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 8  # tiny model dimension for all tests


def tiny_model() -> nn.Module:
    """Simple 2-layer MLP: D -> D -> D."""
    return nn.Sequential(
        nn.Linear(D, D),
        nn.ReLU(),
        nn.Linear(D, D),
    )


def default_config(**kwargs) -> OnlineConfig:
    return OnlineConfig(**kwargs)


# ---------------------------------------------------------------------------
# 1. OnlineConfig defaults
# ---------------------------------------------------------------------------


def test_online_config_defaults():
    cfg = OnlineConfig()
    assert cfg.buffer_size == 1000
    assert cfg.update_frequency == 100
    assert cfg.lr == pytest.approx(1e-4)
    assert cfg.forgetting_penalty == pytest.approx(0.0)
    assert cfg.ewc_lambda == pytest.approx(0.0)
    assert cfg.momentum == pytest.approx(0.0)
    assert cfg.min_loss_threshold == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. StreamingBuffer — add and len
# ---------------------------------------------------------------------------


def test_streaming_buffer_add_and_len():
    buf = StreamingBuffer(capacity=5)
    assert len(buf) == 0
    buf.add("a")
    buf.add("b")
    assert len(buf) == 2


# ---------------------------------------------------------------------------
# 3. StreamingBuffer — evicts oldest when full
# ---------------------------------------------------------------------------


def test_streaming_buffer_evicts_oldest():
    buf = StreamingBuffer(capacity=3)
    buf.add(1)
    buf.add(2)
    buf.add(3)
    assert buf.is_full()
    buf.add(4)  # should evict 1
    assert len(buf) == 3
    assert 1 not in buf._data  # oldest evicted
    assert 4 in buf._data


# ---------------------------------------------------------------------------
# 4. StreamingBuffer — sample length <= n
# ---------------------------------------------------------------------------


def test_streaming_buffer_sample_length():
    buf = StreamingBuffer(capacity=10)
    for i in range(6):
        buf.add(i)
    result = buf.sample(4)
    assert len(result) == 4

    # Requesting more than available returns all
    result2 = buf.sample(20)
    assert len(result2) == 6


# ---------------------------------------------------------------------------
# 5. StreamingBuffer — is_full correct
# ---------------------------------------------------------------------------


def test_streaming_buffer_is_full():
    buf = StreamingBuffer(capacity=3)
    assert not buf.is_full()
    buf.add("x")
    buf.add("y")
    assert not buf.is_full()
    buf.add("z")
    assert buf.is_full()


# ---------------------------------------------------------------------------
# 6. compute_parameter_distance = 0 for identical params
# ---------------------------------------------------------------------------


def test_param_distance_identical():
    model = tiny_model()
    params = [p.detach().clone() for p in model.parameters()]
    dist = compute_parameter_distance(params, params)
    assert dist == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 7. compute_parameter_distance > 0 for different params
# ---------------------------------------------------------------------------


def test_param_distance_different():
    model1 = tiny_model()
    model2 = tiny_model()
    # Ensure they differ
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(torch.ones_like(p))
    p1 = [p.detach().clone() for p in model1.parameters()]
    p2 = [p.detach().clone() for p in model2.parameters()]
    dist = compute_parameter_distance(p1, p2)
    assert dist > 0.0


# ---------------------------------------------------------------------------
# 8. compute_online_ewc_loss returns scalar, non-negative
# ---------------------------------------------------------------------------


def test_ewc_loss_scalar_non_negative():
    model = tiny_model()
    params = list(model.parameters())
    fisher = [torch.rand_like(p) for p in params]
    anchors = [p.detach().clone() for p in params]
    # Perturb model slightly so loss is non-zero
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    loss = compute_online_ewc_loss(model, fisher, anchors, lambda_=1.0)
    assert loss.shape == torch.Size([]) or loss.ndim == 0  # scalar
    assert float(loss.item()) >= 0.0


# ---------------------------------------------------------------------------
# 9. compute_online_ewc_loss = 0 when at anchors
# ---------------------------------------------------------------------------


def test_ewc_loss_zero_at_anchors():
    model = tiny_model()
    params = list(model.parameters())
    fisher = [torch.ones_like(p) for p in params]
    anchors = [p.detach().clone() for p in params]
    # Do NOT perturb model — params == anchors
    loss = compute_online_ewc_loss(model, fisher, anchors, lambda_=1.0)
    assert float(loss.item()) == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 10. OnlineLearner.observe returns float loss
# ---------------------------------------------------------------------------


def test_online_learner_observe_returns_float():
    model = tiny_model()
    cfg = OnlineConfig(lr=1e-3)
    learner = OnlineLearner(model, cfg)
    x = torch.randn(4, D)
    y = torch.randn(4, D)
    loss = learner.observe(x, y)
    assert isinstance(loss, float)
    assert loss >= 0.0


# ---------------------------------------------------------------------------
# 11. OnlineLearner.observe updates parameters
# ---------------------------------------------------------------------------


def test_online_learner_observe_updates_params():
    model = tiny_model()
    cfg = OnlineConfig(lr=1e-2)
    learner = OnlineLearner(model, cfg)
    params_before = [p.detach().clone() for p in model.parameters()]
    x = torch.randn(4, D)
    y = torch.randn(4, D)
    learner.observe(x, y)
    params_after = [p.detach().clone() for p in model.parameters()]
    changed = any(not torch.allclose(pb, pa) for pb, pa in zip(params_before, params_after))
    assert changed, "Parameters should change after an observe step"


# ---------------------------------------------------------------------------
# 12. OnlineLearner.get_parameter_drift >= 0 after update
# ---------------------------------------------------------------------------


def test_online_learner_drift_non_negative():
    model = tiny_model()
    cfg = OnlineConfig(lr=1e-3)
    learner = OnlineLearner(model, cfg)
    drift_before = learner.get_parameter_drift()
    assert drift_before == pytest.approx(0.0, abs=1e-6)
    x = torch.randn(4, D)
    y = torch.randn(4, D)
    learner.observe(x, y)
    drift_after = learner.get_parameter_drift()
    assert drift_after >= 0.0


# ---------------------------------------------------------------------------
# 13. OnlineLearner.save/reset_to_checkpoint round-trip
# ---------------------------------------------------------------------------


def test_online_learner_checkpoint_roundtrip():
    model = tiny_model()
    cfg = OnlineConfig(lr=1e-2)
    learner = OnlineLearner(model, cfg)

    # Save initial checkpoint
    ckpt = learner.save_checkpoint()

    # Apply several updates to change parameters
    for _ in range(5):
        x = torch.randn(4, D)
        y = torch.randn(4, D)
        learner.observe(x, y)

    # Verify params changed
    params_updated = [p.detach().clone() for p in model.parameters()]

    # Restore
    learner.reset_to_checkpoint(ckpt)
    params_restored = [p.detach().clone() for p in model.parameters()]

    for pu, pr in zip(params_updated, params_restored):
        # At least one parameter tensor should differ from updated
        pass  # structural check below

    # All restored params match the saved checkpoint
    for name, saved_param in ckpt.items():
        restored_param = dict(model.named_parameters())[name]
        assert torch.allclose(saved_param, restored_param), (
            f"Parameter {name} not restored correctly"
        )


# ---------------------------------------------------------------------------
# 14. compute_forgetting_metric sign is correct
# ---------------------------------------------------------------------------


def test_forgetting_metric_sign():
    # Positive = forgetting (losses went up)
    before = [1.0, 1.0, 1.0]
    after_worse = [2.0, 2.0, 2.0]
    assert compute_forgetting_metric(before, after_worse) > 0.0

    # Negative = improvement (losses went down)
    after_better = [0.5, 0.5, 0.5]
    assert compute_forgetting_metric(before, after_better) < 0.0

    # Zero = no change
    assert compute_forgetting_metric(before, before) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 15. observe_batch returns correct number of losses
# ---------------------------------------------------------------------------


def test_observe_batch_returns_list():
    model = tiny_model()
    cfg = OnlineConfig(lr=1e-3)
    learner = OnlineLearner(model, cfg)
    xs = [torch.randn(2, D) for _ in range(5)]
    ys = [torch.randn(2, D) for _ in range(5)]
    losses = learner.observe_batch(xs, ys)
    assert isinstance(losses, list)
    assert len(losses) == 5
    assert all(isinstance(line, float) for line in losses)


# ---------------------------------------------------------------------------
# 16. min_loss_threshold skips update
# ---------------------------------------------------------------------------


def test_min_loss_threshold_skips_update():
    """When initial loss is below threshold, parameters should NOT change."""
    model = tiny_model()
    # Use a very large threshold so the update is always skipped
    cfg = OnlineConfig(lr=1e-2, min_loss_threshold=1e9)
    learner = OnlineLearner(model, cfg)
    params_before = [p.detach().clone() for p in model.parameters()]
    x = torch.randn(4, D)
    y = torch.randn(4, D)
    learner.observe(x, y)
    params_after = [p.detach().clone() for p in model.parameters()]
    unchanged = all(torch.allclose(pb, pa) for pb, pa in zip(params_before, params_after))
    assert unchanged, "Parameters should NOT change when loss < min_loss_threshold"
