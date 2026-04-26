"""Tests for src/alignment/rso.py"""

import pytest
import torch
from aurelius.alignment.rso import (
    RSOConfig,
    RSOLoss,
    RSOSampler,
    RSOTrainer,
)

SEED = 42
B = 8  # batch size


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(B: int, requires_grad: bool = False):
    torch.manual_seed(SEED)
    keys = [
        "log_probs_w",
        "log_probs_l",
        "ref_log_probs_w",
        "ref_log_probs_l",
        "reward_w",
        "reward_l",
    ]
    batch = {}
    for k in keys:
        t = torch.randn(B)
        if requires_grad and k.startswith("log_probs"):
            t = t.requires_grad_(True)
        batch[k] = t
    # Ensure reward_w > reward_l for most items (preferred wins)
    batch["reward_w"] = batch["reward_l"] + torch.abs(torch.randn(B)) + 0.1
    return batch


# ---------------------------------------------------------------------------
# RSOSampler
# ---------------------------------------------------------------------------


class TestRSOSampler:
    def test_returns_long_tensor(self):
        sampler = RSOSampler(seed=SEED)
        rewards = torch.randn(10)
        log_probs = torch.randn(10)
        idx = sampler.rejection_sample(rewards, log_probs, n_samples=3)
        assert idx.dtype == torch.long

    def test_output_within_range(self):
        sampler = RSOSampler(seed=SEED)
        N = 20
        rewards = torch.randn(N)
        log_probs = torch.randn(N)
        idx = sampler.rejection_sample(rewards, log_probs, n_samples=5)
        assert (idx >= 0).all() and (idx < N).all()

    def test_n_samples_clamped_to_N(self):
        sampler = RSOSampler(seed=SEED)
        N = 4
        idx = sampler.rejection_sample(torch.randn(N), torch.randn(N), n_samples=100)
        assert idx.numel() <= N

    def test_at_least_one_returned(self):
        # Even with very low acceptance probability, sampler falls back to argmax
        sampler = RSOSampler(seed=SEED)
        # Very high R_max means most accept_probs ≈ 0
        rewards = torch.tensor([-1000.0, -1000.0, 0.0])
        log_probs = torch.zeros(3)
        idx = sampler.rejection_sample(rewards, log_probs, n_samples=1)
        assert idx.numel() >= 1

    def test_sorted_descending_reward(self):
        torch.manual_seed(SEED)
        sampler = RSOSampler(seed=SEED)
        rewards = torch.arange(10, dtype=torch.float)
        log_probs = torch.zeros(10)
        idx = sampler.rejection_sample(rewards, log_probs, n_samples=5)
        selected_rewards = rewards[idx]
        # Returned indices should be in descending reward order
        assert (selected_rewards[:-1] >= selected_rewards[1:]).all()

    def test_highest_reward_always_selected(self):
        # The highest-reward candidate should always appear
        sampler = RSOSampler(seed=0)
        rewards = torch.zeros(10)
        rewards[7] = 10.0  # clear winner
        log_probs = torch.zeros(10)
        idx = sampler.rejection_sample(rewards, log_probs, n_samples=1)
        assert idx[0].item() == 7

    def test_raises_on_empty(self):
        sampler = RSOSampler()
        with pytest.raises(ValueError, match="non-empty"):
            sampler.rejection_sample(torch.tensor([]), torch.tensor([]), n_samples=1)

    def test_raises_on_n_samples_zero(self):
        sampler = RSOSampler()
        with pytest.raises(ValueError, match="n_samples"):
            sampler.rejection_sample(torch.randn(5), torch.randn(5), n_samples=0)

    def test_raises_on_shape_mismatch(self):
        sampler = RSOSampler()
        with pytest.raises(ValueError, match="equal length"):
            sampler.rejection_sample(torch.randn(5), torch.randn(3), n_samples=2)

    def test_raises_on_2d_input(self):
        sampler = RSOSampler()
        with pytest.raises(ValueError, match="1-D"):
            sampler.rejection_sample(torch.randn(5, 2), torch.randn(5, 2), n_samples=2)


# ---------------------------------------------------------------------------
# RSOLoss
# ---------------------------------------------------------------------------


class TestRSOLoss:
    def test_loss_is_scalar(self):
        loss_fn = RSOLoss(beta=0.1)
        batch = _make_batch(B)
        loss, metrics = loss_fn(**batch)
        assert loss.shape == ()

    def test_loss_finite(self):
        torch.manual_seed(SEED)
        loss_fn = RSOLoss(beta=0.1)
        batch = _make_batch(B)
        loss, _ = loss_fn(**batch)
        assert torch.isfinite(loss)

    def test_loss_positive(self):
        # -log sigmoid is always > 0
        loss_fn = RSOLoss(beta=0.1)
        batch = _make_batch(B)
        loss, _ = loss_fn(**batch)
        assert loss.item() > 0

    def test_metrics_keys(self):
        loss_fn = RSOLoss()
        batch = _make_batch(B)
        _, metrics = loss_fn(**batch)
        assert "reward_margin" in metrics
        assert "importance_weight_mean" in metrics
        assert "accuracy" in metrics

    def test_accuracy_in_range(self):
        loss_fn = RSOLoss()
        batch = _make_batch(B)
        _, metrics = loss_fn(**batch)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_importance_weight_mean_positive(self):
        loss_fn = RSOLoss(use_importance_weights=True)
        batch = _make_batch(B)
        _, metrics = loss_fn(**batch)
        assert metrics["importance_weight_mean"] > 0

    def test_no_importance_weights(self):
        torch.manual_seed(SEED)
        batch = _make_batch(B)
        RSOLoss(use_importance_weights=True)(**{**batch})[0]
        loss_no_iw = RSOLoss(use_importance_weights=False)(**{**batch})[0]
        # Without IW it's pure DPO; values will differ
        assert torch.isfinite(loss_no_iw)

    def test_gradients_flow(self):
        loss_fn = RSOLoss(beta=0.1)
        batch = _make_batch(B, requires_grad=True)
        loss, _ = loss_fn(**batch)
        loss.backward()
        assert batch["log_probs_w"].grad is not None
        assert batch["log_probs_l"].grad is not None

    def test_batch1(self):
        loss_fn = RSOLoss()
        batch = _make_batch(1)
        loss, metrics = loss_fn(**batch)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_raises_on_shape_mismatch(self):
        loss_fn = RSOLoss()
        batch = _make_batch(B)
        batch["log_probs_l"] = torch.randn(B + 1)
        with pytest.raises(ValueError):
            loss_fn(**batch)

    def test_reward_margin_correct(self):
        loss_fn = RSOLoss()
        torch.manual_seed(SEED)
        rw = torch.tensor([1.0, 2.0])
        rl = torch.tensor([0.0, 1.0])
        batch = {
            "log_probs_w": torch.zeros(2),
            "log_probs_l": torch.zeros(2),
            "ref_log_probs_w": torch.zeros(2),
            "ref_log_probs_l": torch.zeros(2),
            "reward_w": rw,
            "reward_l": rl,
        }
        _, metrics = loss_fn(**batch)
        assert abs(metrics["reward_margin"] - 1.0) < 1e-5

    def test_beta_scaling(self):
        # Higher beta → larger margin → lower loss (more confident correct direction)
        torch.manual_seed(SEED)
        batch = _make_batch(B)
        # Ensure log_probs_w > log_probs_l on average (correct direction)
        batch["log_probs_w"] = torch.ones(B)
        batch["log_probs_l"] = -torch.ones(B)
        batch["ref_log_probs_w"] = torch.zeros(B)
        batch["ref_log_probs_l"] = torch.zeros(B)
        loss_low = RSOLoss(beta=0.01, use_importance_weights=False)(**batch)[0]
        loss_high = RSOLoss(beta=10.0, use_importance_weights=False)(**batch)[0]
        assert loss_high.item() < loss_low.item()


# ---------------------------------------------------------------------------
# RSOConfig
# ---------------------------------------------------------------------------


class TestRSOConfig:
    def test_defaults(self):
        cfg = RSOConfig()
        assert cfg.beta == 0.1
        assert cfg.use_importance_weights is True
        assert cfg.seed == 42

    def test_custom(self):
        cfg = RSOConfig(beta=0.5, use_importance_weights=False, seed=0)
        assert cfg.beta == 0.5
        assert not cfg.use_importance_weights
        assert cfg.seed == 0


# ---------------------------------------------------------------------------
# RSOTrainer
# ---------------------------------------------------------------------------


class TestRSOTrainer:
    def _make_tiny_model(self):
        return torch.nn.Linear(4, 1)

    def test_compute_loss_returns_scalar(self):
        model = self._make_tiny_model()
        ref_model = self._make_tiny_model()
        trainer = RSOTrainer(model, ref_model)
        batch = _make_batch(B)
        loss = trainer.compute_loss(batch)
        assert loss.shape == ()

    def test_compute_loss_finite(self):
        model = self._make_tiny_model()
        ref_model = self._make_tiny_model()
        trainer = RSOTrainer(model, ref_model)
        batch = _make_batch(B)
        loss = trainer.compute_loss(batch)
        assert torch.isfinite(loss)

    def test_ref_model_frozen(self):
        model = self._make_tiny_model()
        ref_model = self._make_tiny_model()
        RSOTrainer(model, ref_model)
        for p in ref_model.parameters():
            assert not p.requires_grad

    def test_missing_key_raises(self):
        model = self._make_tiny_model()
        ref_model = self._make_tiny_model()
        trainer = RSOTrainer(model, ref_model)
        batch = _make_batch(B)
        del batch["reward_w"]
        with pytest.raises(KeyError, match="reward_w"):
            trainer.compute_loss(batch)

    def test_sampler_is_rso_sampler(self):
        model = self._make_tiny_model()
        ref_model = self._make_tiny_model()
        trainer = RSOTrainer(model, ref_model)
        assert isinstance(trainer.sampler, RSOSampler)

    def test_determinism(self):
        model = self._make_tiny_model()
        ref_model = self._make_tiny_model()
        trainer = RSOTrainer(model, ref_model, RSOConfig(seed=SEED))
        batch = _make_batch(B)
        loss1 = trainer.compute_loss(batch)
        loss2 = trainer.compute_loss(batch)
        assert torch.allclose(loss1, loss2)
