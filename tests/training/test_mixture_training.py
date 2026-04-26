"""Tests for src/training/mixture_training.py.

All tests use tiny dimensions (N=16, K=3, D=8, B=2, T=4) so they run on CPU
in milliseconds.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.mixture_training import (
    GatingNetwork,
    MixtureConfig,
    MixtureTrainer,
    compute_component_loss,
    compute_soft_assignments,
    em_step,
    mixture_weighted_loss,
    update_centroids,
)

# ---------------------------------------------------------------------------
# Tiny shared constants
# ---------------------------------------------------------------------------

N, K, D = 16, 3, 8
B, T = 2, 4


def _embeddings() -> torch.Tensor:
    """(N, D) random unit-norm embeddings."""
    torch.manual_seed(42)
    e = torch.randn(N, D)
    return e / e.norm(dim=-1, keepdim=True)


def _centroids() -> torch.Tensor:
    """(K, D) random unit-norm centroids."""
    torch.manual_seed(7)
    c = torch.randn(K, D)
    return c / c.norm(dim=-1, keepdim=True)


def _assignments() -> torch.Tensor:
    """(N, K) valid soft assignments (rows sum to 1)."""
    torch.manual_seed(0)
    a = torch.rand(N, K)
    return a / a.sum(dim=-1, keepdim=True)


def _per_sample_loss() -> torch.Tensor:
    """(N,) positive per-sample losses."""
    torch.manual_seed(1)
    return torch.rand(N) + 0.1  # ensure all > 0


def _dummy_model() -> nn.Module:
    """Tiny linear model — content doesn't matter for MixtureTrainer tests."""
    return nn.Linear(D, D)


# ---------------------------------------------------------------------------
# MixtureConfig
# ---------------------------------------------------------------------------


class TestMixtureConfig:
    def test_defaults(self) -> None:
        cfg = MixtureConfig()
        assert cfg.n_components == 4
        assert cfg.d_model == 512
        assert cfg.temperature == 1.0
        assert cfg.min_component_weight == 0.01
        assert cfg.em_iterations == 10

    def test_custom_values(self) -> None:
        cfg = MixtureConfig(n_components=K, d_model=D, temperature=0.5)
        assert cfg.n_components == K
        assert cfg.d_model == D
        assert cfg.temperature == 0.5


# ---------------------------------------------------------------------------
# compute_soft_assignments
# ---------------------------------------------------------------------------


class TestComputeSoftAssignments:
    def test_output_shape(self) -> None:
        a = compute_soft_assignments(_embeddings(), _centroids())
        assert a.shape == (N, K)

    def test_rows_sum_to_one(self) -> None:
        a = compute_soft_assignments(_embeddings(), _centroids())
        row_sums = a.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(N), atol=1e-5)

    def test_non_negative(self) -> None:
        a = compute_soft_assignments(_embeddings(), _centroids())
        assert (a >= 0).all()

    def test_temperature_effect(self) -> None:
        """Lower temperature → sharper (higher max) assignments."""
        a_low = compute_soft_assignments(_embeddings(), _centroids(), temperature=0.1)
        a_high = compute_soft_assignments(_embeddings(), _centroids(), temperature=10.0)
        assert a_low.max() > a_high.max()


# ---------------------------------------------------------------------------
# update_centroids
# ---------------------------------------------------------------------------


class TestUpdateCentroids:
    def test_output_shape(self) -> None:
        c = update_centroids(_embeddings(), _assignments())
        assert c.shape == (K, D)

    def test_centroids_finite(self) -> None:
        c = update_centroids(_embeddings(), _assignments())
        assert torch.isfinite(c).all()

    def test_weighted_mean_simple(self) -> None:
        """With one-hot assignments, centroid k == mean of assigned embeddings."""
        emb = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]], dtype=torch.float32)
        # Sample 0 -> component 0, sample 1 -> component 0, sample 2 -> component 1
        assign = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        c = update_centroids(emb, assign)
        expected_c0 = torch.tensor([2.0, 0.0])  # mean of rows 0 and 1
        expected_c1 = torch.tensor([0.0, 2.0])  # row 2
        assert torch.allclose(c[0], expected_c0, atol=1e-5)
        assert torch.allclose(c[1], expected_c1, atol=1e-5)


# ---------------------------------------------------------------------------
# em_step
# ---------------------------------------------------------------------------


class TestEmStep:
    def test_returns_assignments_shape(self) -> None:
        a, _ = em_step(_embeddings(), _centroids(), temperature=1.0)
        assert a.shape == (N, K)

    def test_returns_centroids_shape(self) -> None:
        _, c = em_step(_embeddings(), _centroids(), temperature=1.0)
        assert c.shape == (K, D)

    def test_assignments_sum_to_one(self) -> None:
        a, _ = em_step(_embeddings(), _centroids(), temperature=1.0)
        assert torch.allclose(a.sum(dim=-1), torch.ones(N), atol=1e-5)

    def test_centroids_finite(self) -> None:
        _, c = em_step(_embeddings(), _centroids(), temperature=1.0)
        assert torch.isfinite(c).all()


# ---------------------------------------------------------------------------
# compute_component_loss
# ---------------------------------------------------------------------------


class TestComputeComponentLoss:
    def test_output_shape(self) -> None:
        cl = compute_component_loss(_per_sample_loss(), _assignments())
        assert cl.shape == (K,)

    def test_all_losses_positive(self) -> None:
        cl = compute_component_loss(_per_sample_loss(), _assignments())
        assert (cl > 0).all()

    def test_finite(self) -> None:
        cl = compute_component_loss(_per_sample_loss(), _assignments())
        assert torch.isfinite(cl).all()

    def test_uniform_assignments_equal_mean_loss(self) -> None:
        """With uniform assignments each component loss == global mean loss."""
        losses = _per_sample_loss()
        uniform_assign = torch.full((N, K), 1.0 / K)
        cl = compute_component_loss(losses, uniform_assign)
        expected = losses.mean()
        assert torch.allclose(cl, expected.expand(K), atol=1e-5)


# ---------------------------------------------------------------------------
# mixture_weighted_loss
# ---------------------------------------------------------------------------


class TestMixtureWeightedLoss:
    def test_is_scalar(self) -> None:
        loss = mixture_weighted_loss(_per_sample_loss(), _assignments())
        assert loss.shape == ()

    def test_uniform_weights_match_mean_component_loss(self) -> None:
        losses = _per_sample_loss()
        a = _assignments()
        cl = compute_component_loss(losses, a)
        expected = cl.mean()
        total = mixture_weighted_loss(losses, a)
        assert torch.allclose(total, expected, atol=1e-5)

    def test_explicit_weights(self) -> None:
        losses = _per_sample_loss()
        a = _assignments()
        weights = torch.ones(K) / K
        loss_uniform = mixture_weighted_loss(losses, a, weights=None)
        loss_explicit = mixture_weighted_loss(losses, a, weights=weights)
        assert torch.allclose(loss_uniform, loss_explicit, atol=1e-5)

    def test_positive(self) -> None:
        loss = mixture_weighted_loss(_per_sample_loss(), _assignments())
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# GatingNetwork
# ---------------------------------------------------------------------------


class TestGatingNetwork:
    def _net(self) -> GatingNetwork:
        return GatingNetwork(d_model=D, n_components=K, temperature=1.0)

    def test_output_shape(self) -> None:
        net = self._net()
        x = torch.randn(B, T, D)
        out = net(x)
        assert out.shape == (B, T, K)

    def test_rows_sum_to_one(self) -> None:
        net = self._net()
        x = torch.randn(B, T, D)
        out = net(x)
        sums = out.sum(dim=-1)  # (B, T)
        assert torch.allclose(sums, torch.ones(B, T), atol=1e-5)

    def test_non_negative(self) -> None:
        net = self._net()
        x = torch.randn(B, T, D)
        out = net(x)
        assert (out >= 0).all()

    def test_temperature_sharpness(self) -> None:
        """Lower temperature -> sharper (higher peak) assignments."""
        torch.manual_seed(99)
        x = torch.randn(B, T, D)
        net_low = GatingNetwork(d_model=D, n_components=K, temperature=0.1)
        net_high = GatingNetwork(d_model=D, n_components=K, temperature=10.0)
        # Share weights so only temperature differs
        net_high.linear.weight = net_low.linear.weight
        assert net_low(x).max() > net_high(x).max()


# ---------------------------------------------------------------------------
# MixtureTrainer
# ---------------------------------------------------------------------------


class TestMixtureTrainer:
    def _trainer(self) -> MixtureTrainer:
        cfg = MixtureConfig(n_components=K, d_model=D, temperature=1.0)
        return MixtureTrainer(_dummy_model(), cfg)

    def test_init_centroids_shape(self) -> None:
        trainer = self._trainer()
        c = trainer.init_centroids(_embeddings())
        assert c.shape == (K, D)

    def test_init_centroids_are_subset_of_embeddings(self) -> None:
        trainer = self._trainer()
        emb = _embeddings()
        c = trainer.init_centroids(emb)
        # Each centroid must match at least one row of emb
        for k in range(K):
            diffs = (emb - c[k]).norm(dim=-1)
            assert diffs.min().item() < 1e-5

    def test_run_em_assignments_shape(self) -> None:
        trainer = self._trainer()
        a, _ = trainer.run_em(_embeddings(), n_iter=3)
        assert a.shape == (N, K)

    def test_run_em_centroids_shape(self) -> None:
        trainer = self._trainer()
        _, c = trainer.run_em(_embeddings(), n_iter=3)
        assert c.shape == (K, D)

    def test_run_em_assignments_sum_to_one(self) -> None:
        trainer = self._trainer()
        a, _ = trainer.run_em(_embeddings(), n_iter=3)
        assert torch.allclose(a.sum(dim=-1), torch.ones(N), atol=1e-5)

    def test_compute_loss_is_scalar(self) -> None:
        trainer = self._trainer()
        trainer.init_centroids(_embeddings())
        loss = trainer.compute_loss(_per_sample_loss(), _embeddings())
        assert loss.shape == ()

    def test_compute_loss_positive(self) -> None:
        trainer = self._trainer()
        trainer.init_centroids(_embeddings())
        loss = trainer.compute_loss(_per_sample_loss(), _embeddings())
        assert loss.item() > 0

    def test_compute_loss_finite(self) -> None:
        trainer = self._trainer()
        trainer.init_centroids(_embeddings())
        loss = trainer.compute_loss(_per_sample_loss(), _embeddings())
        assert torch.isfinite(loss)

    def test_centroids_updated_after_compute_loss(self) -> None:
        trainer = self._trainer()
        emb = _embeddings()
        c0 = trainer.init_centroids(emb).clone()
        trainer.compute_loss(_per_sample_loss(), emb)
        # Centroids should have changed after the M-step
        assert not torch.allclose(trainer._centroids, c0)
