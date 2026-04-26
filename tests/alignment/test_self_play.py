"""Tests for Self-Play Training (SPIN) implementation."""

from __future__ import annotations

import pytest
import torch

from src.alignment.self_play import (
    SelfPlayConfig,
    SelfPlayTrainer,
    _compute_token_log_probs,
    compute_win_rate,
    generate_self_play_response,
    spin_loss,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_cfg():
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
def model(tiny_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(tiny_cfg)


@pytest.fixture
def opponent_model(tiny_cfg):
    torch.manual_seed(0)
    m = AureliusTransformer(tiny_cfg)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@pytest.fixture
def sp_config():
    return SelfPlayConfig(
        n_rounds=2,
        beta=0.1,
        max_gen_tokens=4,
        temperature=0.8,
        improvement_threshold=0.01,
    )


@pytest.fixture
def trainer(model, sp_config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    return SelfPlayTrainer(model=model, config=sp_config, optimizer=optimizer)


# ---------------------------------------------------------------------------
# SelfPlayConfig tests
# ---------------------------------------------------------------------------


class TestSelfPlayConfig:
    def test_default_values(self):
        """SelfPlayConfig defaults must match specification."""
        cfg = SelfPlayConfig()
        assert cfg.n_rounds == 3
        assert cfg.beta == 0.1
        assert cfg.max_gen_tokens == 64
        assert cfg.temperature == 0.8
        assert cfg.improvement_threshold == 0.01

    def test_custom_values(self):
        """SelfPlayConfig must accept custom values."""
        cfg = SelfPlayConfig(
            n_rounds=5, beta=0.2, max_gen_tokens=32, temperature=1.0, improvement_threshold=0.05
        )
        assert cfg.n_rounds == 5
        assert cfg.beta == 0.2
        assert cfg.max_gen_tokens == 32
        assert cfg.temperature == 1.0
        assert cfg.improvement_threshold == 0.05


# ---------------------------------------------------------------------------
# generate_self_play_response tests
# ---------------------------------------------------------------------------


class TestGenerateSelfPlayResponse:
    def test_output_shape(self, model):
        """Generated response must have shape (1, max_tokens)."""
        torch.manual_seed(42)
        prompt = torch.randint(0, 256, (1, 8))
        resp = generate_self_play_response(model, prompt, max_tokens=6, temperature=0.8)
        assert resp.shape == (1, 6)

    def test_output_dtype(self, model):
        """Generated tokens must be long integers."""
        torch.manual_seed(42)
        prompt = torch.randint(0, 256, (1, 8))
        resp = generate_self_play_response(model, prompt, max_tokens=4, temperature=0.8)
        assert resp.dtype == torch.long

    def test_tokens_in_vocab_range(self, model):
        """All generated tokens must be in [0, vocab_size)."""
        torch.manual_seed(42)
        prompt = torch.randint(0, 256, (1, 8))
        resp = generate_self_play_response(model, prompt, max_tokens=8, temperature=0.8)
        assert (resp >= 0).all()
        assert (resp < 256).all()

    def test_invalid_temperature_raises(self, model):
        """Temperature <= 0 must raise ValueError."""
        prompt = torch.randint(0, 256, (1, 4))
        with pytest.raises(ValueError, match="temperature must be > 0"):
            generate_self_play_response(model, prompt, max_tokens=4, temperature=0.0)
        with pytest.raises(ValueError, match="temperature must be > 0"):
            generate_self_play_response(model, prompt, max_tokens=4, temperature=-1.0)

    def test_invalid_max_tokens_raises(self, model):
        """max_tokens <= 0 must raise ValueError."""
        prompt = torch.randint(0, 256, (1, 4))
        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            generate_self_play_response(model, prompt, max_tokens=0, temperature=0.8)


# ---------------------------------------------------------------------------
# compute_win_rate tests
# ---------------------------------------------------------------------------


class TestComputeWinRate:
    def test_returns_float(self, model, opponent_model):
        """compute_win_rate must return a float."""
        prompts = [torch.randint(0, 256, (1, 6)) for _ in range(4)]

        def judge(a, b):
            return True

        wr = compute_win_rate(model, opponent_model, prompts, judge)
        assert isinstance(wr, float)

    def test_all_wins(self, model, opponent_model):
        """If judge always returns True, win rate is 1.0."""
        prompts = [torch.randint(0, 256, (1, 6)) for _ in range(5)]
        wr = compute_win_rate(model, opponent_model, prompts, lambda a, b: True)
        assert wr == 1.0

    def test_no_wins(self, model, opponent_model):
        """If judge always returns False, win rate is 0.0."""
        prompts = [torch.randint(0, 256, (1, 6)) for _ in range(5)]
        wr = compute_win_rate(model, opponent_model, prompts, lambda a, b: False)
        assert wr == 0.0

    def test_empty_prompts_raises(self, model, opponent_model):
        """Empty prompts list must raise ValueError."""
        with pytest.raises(ValueError, match="prompts must be non-empty"):
            compute_win_rate(model, opponent_model, [], lambda a, b: True)

    def test_win_rate_bounded(self, model, opponent_model):
        """Win rate must be in [0, 1]."""
        torch.manual_seed(99)
        prompts = [torch.randint(0, 256, (1, 6)) for _ in range(10)]

        def judge(a, b):
            return a.sum().item() > b.sum().item()

        wr = compute_win_rate(model, opponent_model, prompts, judge)
        assert 0.0 <= wr <= 1.0


# ---------------------------------------------------------------------------
# spin_loss tests
# ---------------------------------------------------------------------------


class TestSpinLoss:
    def test_scalar_output(self):
        """spin_loss must return a scalar tensor."""
        B = 4
        loss = spin_loss(
            torch.randn(B),
            torch.randn(B),
            torch.randn(B),
            torch.randn(B),
            beta=0.1,
        )
        assert loss.ndim == 0

    def test_non_negative(self):
        """spin_loss is -log sigmoid(...) which is always >= 0."""
        torch.manual_seed(7)
        B = 8
        loss = spin_loss(
            torch.randn(B),
            torch.randn(B),
            torch.randn(B),
            torch.randn(B),
            beta=0.1,
        )
        assert loss.item() >= 0.0

    def test_finite(self):
        """Loss must be finite for normal inputs."""
        B = 4
        loss = spin_loss(
            torch.randn(B),
            torch.randn(B),
            torch.randn(B),
            torch.randn(B),
            beta=0.1,
        )
        assert torch.isfinite(loss)

    def test_gradient_flows(self):
        """Gradients must flow through spin_loss."""
        B = 4
        policy_real = torch.randn(B, requires_grad=True)
        policy_gen = torch.randn(B, requires_grad=True)
        opp_real = torch.randn(B)
        opp_gen = torch.randn(B)
        loss = spin_loss(policy_real, policy_gen, opp_real, opp_gen, beta=0.1)
        loss.backward()
        assert policy_real.grad is not None
        assert policy_gen.grad is not None

    def test_loss_decreases_when_policy_prefers_real(self):
        """Loss should be lower when policy strongly prefers real over generated."""
        B = 4
        opp_real = torch.zeros(B)
        opp_gen = torch.zeros(B)

        # Policy strongly prefers real
        loss_good = spin_loss(
            torch.full((B,), 0.0),
            torch.full((B,), -5.0),
            opp_real,
            opp_gen,
            beta=0.1,
        )
        # Policy prefers generated (bad)
        loss_bad = spin_loss(
            torch.full((B,), -5.0),
            torch.full((B,), 0.0),
            opp_real,
            opp_gen,
            beta=0.1,
        )
        assert loss_good.item() < loss_bad.item()


# ---------------------------------------------------------------------------
# _compute_token_log_probs tests
# ---------------------------------------------------------------------------


class TestComputeTokenLogProbs:
    def test_shape(self, model):
        """Must return (B,) shaped tensor."""
        B, S, R = 2, 8, 4
        inp = torch.randint(0, 256, (B, S))
        resp = torch.randint(0, 256, (B, R))
        lp = _compute_token_log_probs(model, inp, resp)
        assert lp.shape == (B,)

    def test_negative_values(self, model):
        """Log probs must be <= 0."""
        B, S, R = 2, 8, 4
        inp = torch.randint(0, 256, (B, S))
        resp = torch.randint(0, 256, (B, R))
        lp = _compute_token_log_probs(model, inp, resp)
        assert (lp <= 0).all()


# ---------------------------------------------------------------------------
# SelfPlayTrainer tests
# ---------------------------------------------------------------------------


class TestSelfPlayTrainer:
    def test_init_creates_opponent(self, trainer):
        """Trainer must create a frozen opponent model on init."""
        assert trainer.opponent_model is not None
        for p in trainer.opponent_model.parameters():
            assert not p.requires_grad

    def test_train_round_returns_keys(self, trainer):
        """train_round must return dict with round_loss and win_rate."""
        torch.manual_seed(10)
        real_data = [
            {
                "prompt_ids": torch.randint(0, 256, (1, 6)),
                "response_ids": torch.randint(0, 256, (1, 4)),
            }
        ]
        prompts = [torch.randint(0, 256, (1, 6)) for _ in range(2)]
        result = trainer.train_round(real_data, prompts)
        assert "round_loss" in result
        assert "win_rate" in result
        assert isinstance(result["round_loss"], float)
        assert isinstance(result["win_rate"], float)

    def test_train_round_loss_finite(self, trainer):
        """round_loss from train_round must be finite."""
        torch.manual_seed(11)
        real_data = [
            {
                "prompt_ids": torch.randint(0, 256, (1, 6)),
                "response_ids": torch.randint(0, 256, (1, 4)),
            }
        ]
        prompts = [torch.randint(0, 256, (1, 6))]
        result = trainer.train_round(real_data, prompts)
        assert result["round_loss"] >= 0.0
        import math

        assert math.isfinite(result["round_loss"])

    def test_train_round_updates_opponent(self, trainer):
        """After train_round, opponent weights should match policy weights."""
        torch.manual_seed(12)
        real_data = [
            {
                "prompt_ids": torch.randint(0, 256, (1, 6)),
                "response_ids": torch.randint(0, 256, (1, 4)),
            }
        ]
        prompts = [torch.randint(0, 256, (1, 6))]
        trainer.train_round(real_data, prompts)

        for (n, pp), (_, op) in zip(
            trainer.model.named_parameters(),
            trainer.opponent_model.named_parameters(),
        ):
            assert torch.allclose(pp, op), f"Param {n} differs after train_round"

    def test_win_rate_in_range(self, trainer):
        """Win rate from train_round must be in [0, 1]."""
        torch.manual_seed(13)
        real_data = [
            {
                "prompt_ids": torch.randint(0, 256, (1, 6)),
                "response_ids": torch.randint(0, 256, (1, 4)),
            }
        ]
        prompts = [torch.randint(0, 256, (1, 6)) for _ in range(3)]
        result = trainer.train_round(real_data, prompts)
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_multiple_rounds(self, trainer):
        """Running multiple train_rounds must not error."""
        torch.manual_seed(14)
        real_data = [
            {
                "prompt_ids": torch.randint(0, 256, (1, 6)),
                "response_ids": torch.randint(0, 256, (1, 4)),
            }
        ]
        prompts = [torch.randint(0, 256, (1, 6))]
        for _ in range(2):
            result = trainer.train_round(real_data, prompts)
            assert "round_loss" in result
