"""Tests for src/training/loss_spike_recovery.py.

Run with:
    cd ~/Desktop/Aurelius && .venv/bin/python3.13 -m pytest tests/training/test_loss_spike_recovery.py -v
"""  # noqa: E501

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.training.loss_spike_recovery import (
    CheckpointBuffer,
    LossHistory,
    LossSpikeRecovery,
    SpikeConfig,
    adaptive_grad_clip,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_mlp(seed: int = 0) -> nn.Module:
    """Tiny 2-layer MLP for checkpoint tests."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


@pytest.fixture
def mlp():
    return make_mlp()


@pytest.fixture
def optimizer(mlp):
    return optim.SGD(mlp.parameters(), lr=1e-3)


# ---------------------------------------------------------------------------
# 1. LossHistory.mean and std after N updates
# ---------------------------------------------------------------------------


def test_loss_history_mean_and_std():
    """mean and std are correct after several updates."""
    history = LossHistory(window=10)
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in values:
        history.update(v)

    expected_mean = sum(values) / len(values)  # 3.0
    assert abs(history.mean - expected_mean) < 1e-6

    # population std
    variance = sum((x - expected_mean) ** 2 for x in values) / len(values)
    expected_std = variance**0.5
    assert abs(history.std - expected_std) < 1e-6


# ---------------------------------------------------------------------------
# 2. z_score > threshold → is_spike True
# ---------------------------------------------------------------------------


def test_is_spike_true_when_z_score_exceeds_threshold():
    """is_spike returns True when the z-score is above threshold."""
    history = LossHistory(window=20)
    # Populate with stable losses around 1.0
    for _ in range(15):
        history.update(1.0)
    # std ≈ 0 here but add slight noise so std > 0
    history.update(1.01)
    history.update(0.99)
    history.update(1.005)
    history.update(0.995)

    # Inject a massive spike — should far exceed any reasonable threshold
    spike_loss = 1000.0
    assert history.is_spike(spike_loss, threshold=2.5)


# ---------------------------------------------------------------------------
# 3. Normal loss → is_spike False
# ---------------------------------------------------------------------------


def test_is_spike_false_for_normal_loss():
    """is_spike returns False for a value well within the normal range."""
    history = LossHistory(window=20)
    for v in [1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.01, 0.99, 1.0]:
        history.update(v)

    # A value within 1 std of the mean should not be a spike
    assert not history.is_spike(1.0, threshold=2.5)


# ---------------------------------------------------------------------------
# 4. CheckpointBuffer saves and restores model state
# ---------------------------------------------------------------------------


def test_checkpoint_buffer_save_and_restore(mlp, optimizer):
    """save() followed by restore_latest() completes without error."""
    buf = CheckpointBuffer(capacity=3)
    buf.save(mlp, optimizer, step=10)
    assert len(buf) == 1
    restored_step = buf.restore_latest(mlp, optimizer)
    assert restored_step == 10


# ---------------------------------------------------------------------------
# 5. Restored model has same weights as saved
# ---------------------------------------------------------------------------


def test_checkpoint_buffer_weights_match_after_restore(mlp, optimizer):
    """Weights after restore_latest() equal the weights at save time."""
    buf = CheckpointBuffer(capacity=3)

    # Record the weights at save time
    saved_state = copy.deepcopy(mlp.state_dict())
    buf.save(mlp, optimizer, step=5)

    # Mutate the model weights
    with torch.no_grad():
        for p in mlp.parameters():
            p.add_(torch.ones_like(p) * 999.0)

    # Restore and compare
    buf.restore_latest(mlp, optimizer)
    for key in saved_state:
        assert torch.allclose(mlp.state_dict()[key], saved_state[key]), (
            f"Parameter '{key}' does not match after restore"
        )


# ---------------------------------------------------------------------------
# 6. CheckpointBuffer capacity evicts oldest
# ---------------------------------------------------------------------------


def test_checkpoint_buffer_capacity_evicts_oldest(mlp, optimizer):
    """Once capacity is reached, the oldest checkpoint is evicted."""
    buf = CheckpointBuffer(capacity=2)
    buf.save(mlp, optimizer, step=1)
    buf.save(mlp, optimizer, step=2)
    buf.save(mlp, optimizer, step=3)  # should evict step=1

    assert len(buf) == 2
    # restore_latest should give step=3
    restored = buf.restore_latest(mlp, optimizer)
    assert restored == 3


# ---------------------------------------------------------------------------
# 7. LossSpikeRecovery.step returns dict with correct keys
# ---------------------------------------------------------------------------


def test_loss_spike_recovery_step_returns_correct_keys(mlp, optimizer):
    """step() always returns a dict with exactly the required keys."""
    config = SpikeConfig(min_steps_before_check=0)
    recovery = LossSpikeRecovery(mlp, optimizer, config)

    result = recovery.step(loss=1.0, grad_norm=0.5, step=10)

    required_keys = {"spike", "recovered", "n_recoveries", "z_score", "step_restored_to"}
    assert set(result.keys()) == required_keys
    assert isinstance(result["spike"], bool)
    assert isinstance(result["recovered"], bool)
    assert isinstance(result["n_recoveries"], int)
    assert isinstance(result["z_score"], float)
    # step_restored_to is int or None
    assert result["step_restored_to"] is None or isinstance(result["step_restored_to"], int)


# ---------------------------------------------------------------------------
# 8. step() returns recovered=True after spike (with saved checkpoint)
# ---------------------------------------------------------------------------


def test_loss_spike_recovery_recovered_true_after_spike(mlp, optimizer):
    """recovered=True when a checkpoint exists and a spike is detected."""
    config = SpikeConfig(
        window_size=10,
        spike_threshold=2.0,
        grad_norm_limit=1e9,  # disable grad-norm spike so only loss spike fires
        min_steps_before_check=0,
        cooldown_steps=0,
        max_recoveries=5,
    )
    recovery = LossSpikeRecovery(mlp, optimizer, config)

    # Save a checkpoint so rollback has something to restore
    recovery.checkpoint_buffer.save(mlp, optimizer, step=1)

    # Warm up history with stable losses
    for i in range(10):
        recovery.step(loss=1.0, grad_norm=0.1, step=i + 2)

    # Inject a massive spike
    result = recovery.step(loss=1000.0, grad_norm=0.1, step=100)

    assert result["spike"] is True
    assert result["recovered"] is True


# ---------------------------------------------------------------------------
# 9. After recovery, step_restored_to is set
# ---------------------------------------------------------------------------


def test_loss_spike_recovery_step_restored_to_set(mlp, optimizer):
    """step_restored_to is the step number of the saved checkpoint."""
    config = SpikeConfig(
        window_size=10,
        spike_threshold=2.0,
        grad_norm_limit=1e9,
        min_steps_before_check=0,
        cooldown_steps=0,
        max_recoveries=5,
    )
    recovery = LossSpikeRecovery(mlp, optimizer, config)

    checkpoint_step = 42
    recovery.checkpoint_buffer.save(mlp, optimizer, step=checkpoint_step)

    # Warm up
    for i in range(10):
        recovery.step(loss=1.0, grad_norm=0.1, step=i)

    result = recovery.step(loss=9999.0, grad_norm=0.1, step=100)

    assert result["spike"] is True
    assert result["step_restored_to"] == checkpoint_step


# ---------------------------------------------------------------------------
# 10. recovery_count increments after each spike
# ---------------------------------------------------------------------------


def test_recovery_count_increments(mlp, optimizer):
    """recovery_count goes up by 1 for each detected spike."""
    config = SpikeConfig(
        window_size=5,
        spike_threshold=2.0,
        grad_norm_limit=1e9,
        min_steps_before_check=0,
        cooldown_steps=0,
        max_recoveries=10,
    )
    recovery = LossSpikeRecovery(mlp, optimizer, config)
    recovery.checkpoint_buffer.save(mlp, optimizer, step=0)

    assert recovery.recovery_count == 0

    for spike_num in range(1, 4):
        # Reset history with stable losses
        recovery._loss_history = LossHistory(5)
        for i in range(5):
            recovery._loss_history.update(1.0)
        recovery._cooldown_remaining = 0

        recovery.step(loss=9999.0, grad_norm=0.1, step=100 + spike_num)
        assert recovery.recovery_count == spike_num


# ---------------------------------------------------------------------------
# 11. max_recoveries exceeded → raises RuntimeError
# ---------------------------------------------------------------------------


def test_max_recoveries_exceeded_raises(mlp, optimizer):
    """RuntimeError is raised when max_recoveries is exceeded."""
    config = SpikeConfig(
        window_size=5,
        spike_threshold=2.0,
        grad_norm_limit=1e9,
        min_steps_before_check=0,
        cooldown_steps=0,
        max_recoveries=2,
    )
    recovery = LossSpikeRecovery(mlp, optimizer, config)
    recovery.checkpoint_buffer.save(mlp, optimizer, step=0)

    def do_spike(step_num: int):
        recovery._loss_history = LossHistory(5)
        for _ in range(5):
            recovery._loss_history.update(1.0)
        recovery._cooldown_remaining = 0
        recovery.step(loss=9999.0, grad_norm=0.1, step=step_num)

    do_spike(1)  # recovery 1
    do_spike(2)  # recovery 2

    with pytest.raises(RuntimeError, match="max_recoveries"):
        do_spike(3)  # recovery 3 → should raise


# ---------------------------------------------------------------------------
# 12. should_save_checkpoint False during cooldown
# ---------------------------------------------------------------------------


def test_should_save_checkpoint_false_during_cooldown(mlp, optimizer):
    """should_save_checkpoint returns False while in the post-recovery cooldown."""
    config = SpikeConfig(
        window_size=10,
        spike_threshold=2.0,
        grad_norm_limit=1e9,
        min_steps_before_check=0,
        cooldown_steps=5,
        max_recoveries=5,
    )
    recovery = LossSpikeRecovery(mlp, optimizer, config)
    recovery.checkpoint_buffer.save(mlp, optimizer, step=0)

    # Warm up history
    for i in range(10):
        recovery.step(loss=1.0, grad_norm=0.1, step=i)

    # Trigger a spike → enters cooldown
    recovery.step(loss=9999.0, grad_norm=0.1, step=100)
    assert recovery._cooldown_remaining > 0

    # During cooldown, should_save_checkpoint must be False
    assert not recovery.should_save_checkpoint(step=101, loss=1.0)


# ---------------------------------------------------------------------------
# 13. adaptive_grad_clip returns smaller value when grad_norm is high
# ---------------------------------------------------------------------------


def test_adaptive_grad_clip_smaller_than_raw_norm():
    """adaptive_grad_clip gives a clip value smaller than the raw grad norm
    when the gradient is much larger than the typical loss scale."""
    history = LossHistory(window=20)
    # Populate with stable losses around 1.0
    for v in [1.0, 0.9, 1.1, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01, 1.0]:
        history.update(v)

    high_grad_norm = 500.0
    clip_val = adaptive_grad_clip(high_grad_norm, history, multiplier=3.0)

    # With mean ≈ 1.0 and multiplier=3.0, clip_val ≈ 3.0 < 500.0
    assert clip_val < high_grad_norm, (
        f"Expected clip_val ({clip_val}) < grad_norm ({high_grad_norm})"
    )


# ---------------------------------------------------------------------------
# Bonus: adaptive_grad_clip with empty history returns grad_norm unchanged
# ---------------------------------------------------------------------------


def test_adaptive_grad_clip_no_history_returns_grad_norm():
    """With no history, adaptive_grad_clip returns grad_norm unchanged."""
    history = LossHistory(window=10)
    grad_norm = 42.5
    result = adaptive_grad_clip(grad_norm, history, multiplier=3.0)
    assert result == grad_norm


# ---------------------------------------------------------------------------
# Bonus: is_stable property
# ---------------------------------------------------------------------------


def test_is_stable_false_after_recovery(mlp, optimizer):
    """is_stable is False after at least one recovery."""
    config = SpikeConfig(
        window_size=5,
        spike_threshold=2.0,
        grad_norm_limit=1e9,
        min_steps_before_check=0,
        cooldown_steps=0,
        max_recoveries=5,
    )
    recovery = LossSpikeRecovery(mlp, optimizer, config)
    recovery.checkpoint_buffer.save(mlp, optimizer, step=0)
    assert recovery.is_stable

    # Warm up + spike
    for i in range(5):
        recovery._loss_history.update(1.0)
    recovery.step(loss=9999.0, grad_norm=0.1, step=10)

    assert not recovery.is_stable
