"""Tests for online_learning module — streaming learning with experience replay."""
from __future__ import annotations

import torch
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.online_learning import (
    OnlineLearningConfig,
    ReplayBuffer,
    detect_concept_drift,
    compute_fisher_diagonal,
    ewc_penalty,
    OnlineLearner,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_cfg():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def small_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def optimizer(small_model):
    return torch.optim.AdamW(small_model.parameters(), lr=1e-4)


@pytest.fixture
def online_config():
    return OnlineLearningConfig()


@pytest.fixture
def learner(small_model, optimizer, online_config):
    return OnlineLearner(small_model, online_config, optimizer)


def _make_batch(batch_size: int = 2, seq_len: int = 8, vocab_size: int = 256) -> tuple:
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, labels


def _make_item(batch_size: int = 2, seq_len: int = 8, vocab_size: int = 256) -> dict:
    ids, lbl = _make_batch(batch_size, seq_len, vocab_size)
    return {"input_ids": ids, "labels": lbl}


# ---------------------------------------------------------------------------
# OnlineLearningConfig tests
# ---------------------------------------------------------------------------

def test_online_learning_config_defaults():
    """Test that OnlineLearningConfig has the correct default values."""
    cfg = OnlineLearningConfig()
    assert cfg.replay_buffer_size == 1000
    assert cfg.replay_batch_size == 16
    assert cfg.ewc_lambda == 1.0
    assert cfg.drift_detection_window == 50
    assert cfg.drift_threshold == 2.0
    assert cfg.learning_rate == 1e-4


# ---------------------------------------------------------------------------
# ReplayBuffer tests
# ---------------------------------------------------------------------------

def test_replay_buffer_add_and_len():
    """Test that ReplayBuffer.add increments len correctly."""
    buf = ReplayBuffer(max_size=10)
    assert len(buf) == 0
    buf.add(_make_item())
    assert len(buf) == 1
    buf.add(_make_item())
    assert len(buf) == 2


def test_replay_buffer_evicts_oldest_when_full():
    """Test that oldest items are evicted (FIFO) when buffer is full."""
    buf = ReplayBuffer(max_size=3)
    items = [{"input_ids": torch.tensor([[i]]), "labels": torch.tensor([[i]])} for i in range(5)]
    for item in items:
        buf.add(item)
    # Buffer should only have 3 items and the first 2 (index 0,1) should be gone
    assert len(buf) == 3
    # The remaining items should be the last 3 added (indices 2,3,4)
    remaining_ids = [buf._buffer[j]["input_ids"].item() for j in range(3)]
    assert remaining_ids == [2, 3, 4]


def test_replay_buffer_sample_returns_correct_count():
    """Test that sample returns exactly n items when buffer has enough."""
    buf = ReplayBuffer(max_size=50)
    for _ in range(20):
        buf.add(_make_item())
    samples = buf.sample(5)
    assert len(samples) == 5


def test_replay_buffer_sample_with_replacement_when_n_gt_len():
    """Test that sample uses replacement when n > buffer size."""
    buf = ReplayBuffer(max_size=50)
    buf.add(_make_item())
    samples = buf.sample(10)
    assert len(samples) == 10


# ---------------------------------------------------------------------------
# detect_concept_drift tests
# ---------------------------------------------------------------------------

def test_detect_concept_drift_returns_false_when_not_enough_data():
    """Test that drift detection returns False when history shorter than window."""
    loss_history = [1.0, 1.1, 1.05]
    result = detect_concept_drift(loss_history, window=10, threshold=2.0)
    assert result is False


def test_detect_concept_drift_returns_true_on_sharp_increase():
    """Test that drift detection returns True when loss sharply increases."""
    # First half: stable low losses, second half: much higher
    first_half = [0.5] * 25
    second_half = [5.0] * 25
    loss_history = first_half + second_half
    result = detect_concept_drift(loss_history, window=50, threshold=2.0)
    assert result is True


def test_detect_concept_drift_stable_loss_returns_false():
    """Test that drift detection returns False for stable loss."""
    loss_history = [1.0] * 60
    result = detect_concept_drift(loss_history, window=50, threshold=2.0)
    assert result is False


# ---------------------------------------------------------------------------
# compute_fisher_diagonal tests
# ---------------------------------------------------------------------------

def test_compute_fisher_diagonal_keys_match_param_names(small_model):
    """Test that compute_fisher_diagonal returns dict with same keys as model params."""
    def loss_fn(model, input_ids, labels):
        import torch.nn.functional as F
        _, logits, _ = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, logits.size(-1)),
            shift_labels.view(-1),
        )

    data_batch = [_make_item(batch_size=1, seq_len=4) for _ in range(3)]
    fisher = compute_fisher_diagonal(small_model, data_batch, loss_fn)

    expected_keys = {name for name, p in small_model.named_parameters() if p.requires_grad}
    assert set(fisher.keys()) == expected_keys


# ---------------------------------------------------------------------------
# ewc_penalty tests
# ---------------------------------------------------------------------------

def test_ewc_penalty_scalar_output(small_model):
    """Test that ewc_penalty returns a scalar tensor."""
    # Build dummy fisher and optimal_params
    fisher = {
        name: torch.ones_like(param.data)
        for name, param in small_model.named_parameters()
        if param.requires_grad
    }
    optimal_params = {
        name: param.data.clone()
        for name, param in small_model.named_parameters()
        if param.requires_grad
    }
    # Perturb params slightly
    with torch.no_grad():
        for param in small_model.parameters():
            param.add_(0.01)

    penalty = ewc_penalty(small_model, fisher, optimal_params, ewc_lambda=1.0)

    assert penalty.ndim == 0  # scalar
    assert penalty.item() > 0.0

    # Restore params
    with torch.no_grad():
        for param in small_model.parameters():
            param.sub_(0.01)


def test_ewc_penalty_zero_when_at_optimal_params(small_model):
    """Test that ewc_penalty is zero when model is at optimal parameters."""
    fisher = {
        name: torch.ones_like(param.data)
        for name, param in small_model.named_parameters()
        if param.requires_grad
    }
    optimal_params = {
        name: param.data.clone()
        for name, param in small_model.named_parameters()
        if param.requires_grad
    }

    penalty = ewc_penalty(small_model, fisher, optimal_params, ewc_lambda=1.0)
    assert penalty.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# OnlineLearner tests
# ---------------------------------------------------------------------------

def test_online_learner_update_returns_correct_keys(learner):
    """Test that OnlineLearner.update returns dict with all required keys."""
    ids, lbl = _make_batch(batch_size=2, seq_len=8)
    result = learner.update(ids, lbl)
    for key in ("task_loss", "replay_loss", "drift_detected", "buffer_size"):
        assert key in result, f"Missing key: {key}"


def test_online_learner_update_buffer_size_increases(learner):
    """Test that buffer_size increases with each update call."""
    ids, lbl = _make_batch(batch_size=2, seq_len=8)
    result1 = learner.update(ids, lbl)
    result2 = learner.update(ids, lbl)
    assert result2["buffer_size"] > result1["buffer_size"]
