"""Tests for continual_replay: DER++, ring buffer replay, and EWC consolidation."""

from __future__ import annotations

import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.continual_replay import (
    ContinualReplayTrainer,
    ReplayConfig,
    ReplayExample,
    RingBufferReplay,
    compute_ewc_penalty,
    der_loss,
    estimate_fisher_diagonal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_model():
    torch.manual_seed(42)
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


def make_example(
    seq_len: int = 16, vocab_size: int = 256, task_id: int = 0, with_logits: bool = False
):
    ids = torch.randint(0, vocab_size, (seq_len,))
    labels = ids.clone()
    logits = torch.randn(seq_len, vocab_size) if with_logits else None
    return ReplayExample(
        input_ids=ids,
        labels=labels,
        logits=logits,
        task_id=task_id,
        timestamp=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_replay_config_defaults():
    """Test ReplayConfig has correct default values."""
    cfg = ReplayConfig()
    assert cfg.buffer_size == 1000
    assert cfg.replay_fraction == 0.25
    assert cfg.alpha == 0.1
    assert cfg.beta == 0.5
    assert cfg.ewc_lambda == 0.01
    assert cfg.strategy == "der++"
    assert cfg.reservoir_sampling is True


def test_replay_example_fields():
    """Test ReplayExample stores all required fields."""
    ids = torch.randint(0, 256, (16,))
    labels = ids.clone()
    logits = torch.randn(16, 256)
    ex = ReplayExample(input_ids=ids, labels=labels, logits=logits, task_id=1, timestamp=5)
    assert ex.input_ids.shape == (16,)
    assert ex.labels.shape == (16,)
    assert ex.logits is not None and ex.logits.shape == (16, 256)
    assert ex.task_id == 1
    assert ex.timestamp == 5


def test_ring_buffer_add_and_len():
    """Test adding examples and len() reporting."""
    cfg = ReplayConfig(buffer_size=10, reservoir_sampling=False)
    buf = RingBufferReplay(cfg)
    assert len(buf) == 0

    for i in range(5):
        buf.add(make_example(task_id=i))
    assert len(buf) == 5

    # Add more than buffer_size
    for i in range(10):
        buf.add(make_example(task_id=i))
    assert len(buf) == 10  # Should not exceed buffer_size


def test_ring_buffer_sample_count():
    """Test that sample returns at most min(n, len) examples."""
    cfg = ReplayConfig(buffer_size=20, reservoir_sampling=False)
    buf = RingBufferReplay(cfg)
    for i in range(8):
        buf.add(make_example(task_id=i))

    sampled = buf.sample(5)
    assert len(sampled) == 5

    sampled_all = buf.sample(100)
    assert len(sampled_all) == 8  # Only 8 in buffer


def test_ring_buffer_reservoir_sampling():
    """Test that reservoir sampling keeps buffer bounded at buffer_size."""
    cfg = ReplayConfig(buffer_size=10, reservoir_sampling=True)
    buf = RingBufferReplay(cfg)

    # Add many more examples than buffer_size
    for i in range(100):
        buf.add(make_example(task_id=i % 5))

    # Buffer must not exceed buffer_size
    assert len(buf) <= cfg.buffer_size
    assert len(buf) == cfg.buffer_size


def test_ring_buffer_task_distribution():
    """Test task_distribution counts examples per task_id."""
    cfg = ReplayConfig(buffer_size=50, reservoir_sampling=False)
    buf = RingBufferReplay(cfg)

    for _ in range(10):
        buf.add(make_example(task_id=0))
    for _ in range(5):
        buf.add(make_example(task_id=1))

    dist = buf.task_distribution()
    assert dist[0] == 10
    assert dist[1] == 5
    assert 2 not in dist


def test_der_loss_shape():
    """Test der_loss returns a scalar tensor."""
    B, T, V = 4, 8, 256
    current_logits = torch.randn(B, T, V)
    replay_logits = torch.randn(B, T, V)
    replay_labels = torch.randint(0, V, (B, T))

    loss, info = der_loss(current_logits, replay_logits, replay_labels)
    assert loss.shape == ()  # scalar
    assert loss.item() >= 0.0


def test_der_loss_keys():
    """Test der_loss info dict has mse_loss and ce_loss keys."""
    B, T, V = 2, 4, 32
    current_logits = torch.randn(B, T, V)
    replay_logits = torch.randn(B, T, V)
    replay_labels = torch.randint(0, V, (B, T))

    _, info = der_loss(current_logits, replay_logits, replay_labels)
    assert "mse_loss" in info
    assert "ce_loss" in info
    assert isinstance(info["mse_loss"], float)
    assert isinstance(info["ce_loss"], float)


def test_compute_ewc_penalty_zero():
    """Test that EWC penalty is ~0 when current params equal old params."""
    model = make_model()
    old_params = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    fisher_info = {
        name: torch.ones_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    penalty = compute_ewc_penalty(model, fisher_info, old_params, ewc_lambda=0.01)
    assert penalty.item() < 1e-5


def test_estimate_fisher_diagonal_keys():
    """Test estimate_fisher_diagonal returns dict with param names."""
    model = make_model()
    data = [torch.randint(0, 256, (2, 16)) for _ in range(3)]

    fisher = estimate_fisher_diagonal(model, data, n_samples=3)

    expected_keys = {name for name, p in model.named_parameters() if p.requires_grad}
    assert set(fisher.keys()) == expected_keys
    for name, val in fisher.items():
        assert val.shape == dict(model.named_parameters())[name].shape


def test_continual_trainer_step_keys():
    """Test train_step returns dict with loss, replay_loss, ewc_loss."""
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    cfg = ReplayConfig(buffer_size=100, strategy="der++")
    trainer = ContinualReplayTrainer(model, cfg, optimizer)

    input_ids = torch.randint(0, 256, (2, 16))
    labels = input_ids.clone()

    result = trainer.train_step(input_ids, labels, task_id=0)
    assert "loss" in result
    assert "replay_loss" in result
    assert "ewc_loss" in result
    assert isinstance(result["loss"], float)
    assert isinstance(result["replay_loss"], float)
    assert isinstance(result["ewc_loss"], float)


def test_continual_trainer_buffer_grows():
    """Test that buffer has entries after train_step calls."""
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    cfg = ReplayConfig(buffer_size=100, strategy="er")
    trainer = ContinualReplayTrainer(model, cfg, optimizer)

    assert len(trainer.buffer) == 0

    input_ids = torch.randint(0, 256, (2, 16))
    labels = input_ids.clone()

    trainer.train_step(input_ids, labels, task_id=0)
    assert len(trainer.buffer) == 2  # 2 examples from batch size 2

    trainer.train_step(input_ids, labels, task_id=1)
    assert len(trainer.buffer) == 4  # 4 total
