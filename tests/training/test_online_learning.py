"""Tests for online_learning module."""
import torch
import pytest

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.online_learning import (
    OnlineLearningConfig,
    StreamingDataBuffer,
    OnlineLearner,
    DataStreamSimulator,
    cosine_similarity_params,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
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


@pytest.fixture
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
    return OnlineLearner(small_model, optimizer, online_config)


def _make_batch(batch_size: int = 1, seq_len: int = 8, vocab_size: int = 256) -> tuple:
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    return input_ids, labels


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_online_learning_config_defaults():
    cfg = OnlineLearningConfig()
    assert cfg.buffer_size == 1000
    assert cfg.replay_ratio == 0.3
    assert cfg.lr_warmup_steps == 100
    assert cfg.forgetting_threshold == 0.1
    assert cfg.use_ewc is True
    assert cfg.ewc_lambda == 0.4


def test_streaming_buffer_add():
    buf = StreamingDataBuffer(capacity=5)
    assert len(buf) == 0
    ids, lbl = _make_batch()
    buf.add(ids, lbl)
    assert len(buf) == 1
    buf.add(ids, lbl)
    assert len(buf) == 2


def test_streaming_buffer_capacity():
    buf = StreamingDataBuffer(capacity=3)
    ids, lbl = _make_batch()
    for _ in range(10):
        buf.add(ids, lbl)
    assert len(buf) <= 3


def test_streaming_buffer_sample():
    buf = StreamingDataBuffer(capacity=20)
    ids, lbl = _make_batch()
    for _ in range(15):
        buf.add(ids, lbl)
    samples = buf.sample(5)
    assert len(samples) == 5
    for item in samples:
        assert len(item) == 2


def test_streaming_buffer_is_ready():
    buf = StreamingDataBuffer(capacity=50)
    assert not buf.is_ready(min_size=10)
    ids, lbl = _make_batch()
    for _ in range(10):
        buf.add(ids, lbl)
    assert buf.is_ready(min_size=10)


def test_online_learner_train_keys(learner):
    ids, lbl = _make_batch(batch_size=1, seq_len=8)
    result = learner.train_on_batch(ids, lbl)
    for key in ("loss", "replay_loss", "ewc_loss", "buffer_size"):
        assert key in result, f"Missing key: {key}"


def test_online_learner_buffer_grows(learner):
    ids, lbl = _make_batch(batch_size=1, seq_len=8)
    result1 = learner.train_on_batch(ids, lbl)
    result2 = learner.train_on_batch(ids, lbl)
    assert result2["buffer_size"] > result1["buffer_size"]


def test_online_learner_ewc_penalty_zero_no_fisher(learner):
    penalty = learner.ewc_penalty()
    assert penalty.item() == 0.0


def test_data_stream_simulator_shape():
    sim = DataStreamSimulator(vocab_size=256, seq_len=16, seed=42)
    input_ids, labels = sim.next_batch(task_id=0, batch_size=4)
    assert input_ids.shape == (4, 16)
    assert labels.shape == (4, 16)


def test_evaluate_forgetting_keys(learner):
    ids, lbl = _make_batch(batch_size=1, seq_len=8)
    baseline_losses = [5.0]
    result = learner.evaluate_forgetting([(ids, lbl)], baseline_losses)
    for key in ("mean_forgetting", "max_forgetting", "catastrophic"):
        assert key in result, f"Missing key: {key}"


def test_cosine_similarity_same_params(small_model):
    params = {name: param.data.clone() for name, param in small_model.named_parameters()}
    sim = cosine_similarity_params(params, params)
    assert abs(sim - 1.0) < 1e-5, f"Expected ~1.0, got {sim}"
