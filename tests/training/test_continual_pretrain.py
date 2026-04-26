"""Tests for src/training/continual_pretrain.py."""

from __future__ import annotations

import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.continual_pretrain import (
    ContinualPretrainConfig,
    ContinualPretrainer,
    DomainBatch,
    ExperienceReplayBuffer,
    ReplayConfig,
    compute_domain_mixing_weights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_model():
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


def make_seqs(n: int = 2, seq_len: int = 8, vocab_size: int = 256) -> list[torch.Tensor]:
    """Return n 1-D random token tensors."""
    return [torch.randint(1, vocab_size, (seq_len,)) for _ in range(n)]


def make_trainer(model=None):
    if model is None:
        model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    config = ContinualPretrainConfig(
        replay=ReplayConfig(buffer_size=100, replay_ratio=0.3),
    )
    return ContinualPretrainer(
        model=model,
        optimizer=optimizer,
        config=config,
        tokenizer_encode=lambda s: [ord(c) for c in s],
    )


# ---------------------------------------------------------------------------
# 1. ReplayConfig defaults
# ---------------------------------------------------------------------------


def test_replay_config_defaults():
    cfg = ReplayConfig()
    assert cfg.buffer_size == 1000
    assert cfg.replay_ratio == 0.3
    assert cfg.reservoir_sampling is True


# ---------------------------------------------------------------------------
# 2. ContinualPretrainConfig defaults
# ---------------------------------------------------------------------------


def test_continual_pretrain_config_defaults():
    cfg = ContinualPretrainConfig()
    assert cfg.learning_rate == 1e-4
    assert cfg.max_seq_len == 512
    assert cfg.warmup_steps == 100
    assert isinstance(cfg.replay, ReplayConfig)
    assert isinstance(cfg.domain_weights, dict)
    assert len(cfg.domain_weights) == 0


# ---------------------------------------------------------------------------
# 3. ExperienceReplayBuffer starts empty
# ---------------------------------------------------------------------------


def test_replay_buffer_starts_empty():
    buf = ExperienceReplayBuffer(ReplayConfig(buffer_size=50))
    assert len(buf) == 0


# ---------------------------------------------------------------------------
# 4. ExperienceReplayBuffer.add increases length
# ---------------------------------------------------------------------------


def test_replay_buffer_add_increases_length():
    buf = ExperienceReplayBuffer(ReplayConfig(buffer_size=50))
    for i in range(5):
        buf.add(torch.randint(1, 256, (8,)))
    assert len(buf) == 5


# ---------------------------------------------------------------------------
# 5. ExperienceReplayBuffer.add doesn't exceed buffer_size
# ---------------------------------------------------------------------------


def test_replay_buffer_add_does_not_exceed_capacity():
    buf = ExperienceReplayBuffer(ReplayConfig(buffer_size=10, reservoir_sampling=False))
    for i in range(25):
        buf.add(torch.randint(1, 256, (8,)))
    assert len(buf) == 10


# ---------------------------------------------------------------------------
# 6. ExperienceReplayBuffer.sample returns correct count
# ---------------------------------------------------------------------------


def test_replay_buffer_sample_returns_correct_count():
    buf = ExperienceReplayBuffer(ReplayConfig(buffer_size=50))
    for _ in range(20):
        buf.add(torch.randint(1, 256, (8,)))
    result = buf.sample(7)
    assert len(result) == 7
    assert all(isinstance(t, torch.Tensor) for t in result)


# ---------------------------------------------------------------------------
# 7. ExperienceReplayBuffer reservoir sampling — length stays capped
# ---------------------------------------------------------------------------


def test_replay_buffer_reservoir_sampling_capped():
    buf = ExperienceReplayBuffer(ReplayConfig(buffer_size=10, reservoir_sampling=True))
    for i in range(100):
        buf.add(torch.randint(1, 256, (8,)))
    assert len(buf) == 10


# ---------------------------------------------------------------------------
# 8. ExperienceReplayBuffer.sample from empty buffer returns empty list
# ---------------------------------------------------------------------------


def test_replay_buffer_sample_empty():
    buf = ExperienceReplayBuffer(ReplayConfig(buffer_size=50))
    result = buf.sample(5)
    assert result == []


# ---------------------------------------------------------------------------
# 9. ContinualPretrainer.train_step returns required keys
# ---------------------------------------------------------------------------


def test_train_step_returns_required_keys():
    trainer = make_trainer()
    seqs = make_seqs(n=2, seq_len=8)
    result = trainer.train_step(seqs, domain="test")
    assert "loss" in result
    assert "domain" in result
    assert "n_new" in result
    assert "n_replay" in result


# ---------------------------------------------------------------------------
# 10. ContinualPretrainer.train_step loss is positive float
# ---------------------------------------------------------------------------


def test_train_step_loss_is_positive_float():
    trainer = make_trainer()
    seqs = make_seqs(n=2, seq_len=8)
    result = trainer.train_step(seqs, domain="test")
    assert isinstance(result["loss"], float)
    assert result["loss"] > 0.0


# ---------------------------------------------------------------------------
# 11. ContinualPretrainer.train_step n_new matches input count
# ---------------------------------------------------------------------------


def test_train_step_n_new_matches_input():
    trainer = make_trainer()
    seqs = make_seqs(n=3, seq_len=8)
    result = trainer.train_step(seqs, domain="test")
    assert result["n_new"] == 3


# ---------------------------------------------------------------------------
# 12. ContinualPretrainer.train_step adds to replay buffer
# ---------------------------------------------------------------------------


def test_train_step_adds_to_replay_buffer():
    trainer = make_trainer()
    assert len(trainer.replay_buffer) == 0

    seqs = make_seqs(n=2, seq_len=8)
    trainer.train_step(seqs, domain="test")
    assert len(trainer.replay_buffer) == 2

    seqs2 = make_seqs(n=3, seq_len=8)
    trainer.train_step(seqs2, domain="test")
    assert len(trainer.replay_buffer) == 5


# ---------------------------------------------------------------------------
# 13. ContinualPretrainer.evaluate_forgetting returns required keys
# ---------------------------------------------------------------------------


def test_evaluate_forgetting_returns_required_keys():
    trainer = make_trainer()
    old_seqs = make_seqs(n=2, seq_len=8)
    new_seqs = make_seqs(n=2, seq_len=8)
    result = trainer.evaluate_forgetting(old_seqs, new_seqs)
    assert "old_loss" in result
    assert "new_loss" in result
    assert "forgetting" in result


# ---------------------------------------------------------------------------
# 14. ContinualPretrainer.evaluate_forgetting forgetting = old_loss - new_loss
# ---------------------------------------------------------------------------


def test_evaluate_forgetting_formula():
    trainer = make_trainer()
    old_seqs = make_seqs(n=2, seq_len=8)
    new_seqs = make_seqs(n=2, seq_len=8)
    result = trainer.evaluate_forgetting(old_seqs, new_seqs)
    expected_forgetting = result["old_loss"] - result["new_loss"]
    assert abs(result["forgetting"] - expected_forgetting) < 1e-6


# ---------------------------------------------------------------------------
# 15. compute_domain_mixing_weights sums to 1.0
# ---------------------------------------------------------------------------


def test_domain_mixing_weights_sum_to_one():
    losses = {"wiki": 2.5, "books": 1.8, "code": 3.1}
    weights = compute_domain_mixing_weights(losses)
    assert set(weights.keys()) == set(losses.keys())
    total = sum(weights.values())
    assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 16. compute_domain_mixing_weights higher-loss domain gets higher weight
# ---------------------------------------------------------------------------


def test_domain_mixing_weights_higher_loss_higher_weight():
    losses = {"easy": 0.5, "hard": 3.0}
    weights = compute_domain_mixing_weights(losses)
    assert weights["hard"] > weights["easy"]


# ---------------------------------------------------------------------------
# Extra: sample with replacement when n > buffer size
# ---------------------------------------------------------------------------


def test_replay_buffer_sample_with_replacement():
    buf = ExperienceReplayBuffer(ReplayConfig(buffer_size=50))
    for _ in range(5):
        buf.add(torch.randint(1, 256, (8,)))
    result = buf.sample(10)  # more than 5 in buffer
    assert len(result) == 10


# ---------------------------------------------------------------------------
# Extra: domain label is passed through train_step
# ---------------------------------------------------------------------------


def test_train_step_domain_label():
    trainer = make_trainer()
    seqs = make_seqs(n=1, seq_len=8)
    result = trainer.train_step(seqs, domain="science")
    assert result["domain"] == "science"


# ---------------------------------------------------------------------------
# Extra: DomainBatch dataclass instantiation
# ---------------------------------------------------------------------------


def test_domain_batch_creation():
    tokens = torch.randint(1, 256, (4, 8))
    batch = DomainBatch(tokens=tokens, domain="news")
    assert batch.domain == "news"
    assert batch.tokens.shape == (4, 8)
    assert batch.weights is None


# ---------------------------------------------------------------------------
# Extra: compute_domain_mixing_weights with temperature
# ---------------------------------------------------------------------------


def test_domain_mixing_weights_temperature():
    losses = {"a": 1.0, "b": 3.0}
    w_low_temp = compute_domain_mixing_weights(losses, temperature=0.1)
    w_high_temp = compute_domain_mixing_weights(losses, temperature=10.0)
    # Low temperature → sharper distribution → "b" gets even more weight
    assert w_low_temp["b"] > w_high_temp["b"]
