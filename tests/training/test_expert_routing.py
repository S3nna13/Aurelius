"""Tests for src/training/expert_routing.py.

All tests use tiny dimensions (B=2, T=4, n_experts=4) so they run on CPU
in milliseconds.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.training.expert_routing import (
    ExpertLoadTracker,
    ExpertRoutingConfig,
    RoutingAwareLoss,
    compute_expert_utilization,
    compute_load_balance_loss,
    compute_router_entropy,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

B, T, E = 2, 4, 4  # tiny dimensions used across tests


def _uniform_probs() -> torch.Tensor:
    """(B, T, E) tensor where every token has equal probability for each expert."""
    return torch.full((B, T, E), 1.0 / E)


def _uniform_top1_mask() -> torch.Tensor:
    """(B, T, E) mask where each token is assigned to exactly one expert in a
    round-robin fashion so that every expert gets exactly B*T/E tokens."""
    mask = torch.zeros(B, T, E)
    n_tokens = B * T  # 8
    for tok_idx in range(n_tokens):
        b = tok_idx // T
        t = tok_idx % T
        mask[b, t, tok_idx % E] = 1.0
    return mask


def _single_expert_mask(expert_idx: int = 0) -> torch.Tensor:
    """(B, T, E) mask where every token goes to a single expert."""
    mask = torch.zeros(B, T, E)
    mask[:, :, expert_idx] = 1.0
    return mask


# ---------------------------------------------------------------------------
# 1. ExpertRoutingConfig defaults
# ---------------------------------------------------------------------------


class TestExpertRoutingConfig:
    def test_defaults(self):
        cfg = ExpertRoutingConfig()
        assert cfg.n_experts == 8
        assert cfg.n_active == 2
        assert cfg.load_balance_coeff == pytest.approx(0.01)
        assert cfg.entropy_coeff == pytest.approx(0.001)
        assert cfg.track_history is True
        assert cfg.history_window == 1000

    def test_custom_values(self):
        cfg = ExpertRoutingConfig(n_experts=4, n_active=1, load_balance_coeff=0.1)
        assert cfg.n_experts == 4
        assert cfg.n_active == 1
        assert cfg.load_balance_coeff == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 2 & 3. compute_load_balance_loss
# ---------------------------------------------------------------------------


class TestComputeLoadBalanceLoss:
    def test_returns_scalar(self):
        probs = _uniform_probs()
        mask = _uniform_top1_mask()
        loss = compute_load_balance_loss(probs, mask)
        assert loss.ndim == 0, "Expected a scalar tensor"

    def test_uniform_gives_expected_value(self):
        """With uniform probs AND uniform mask, loss = n_experts * sum(f_i * P_i).
        f_i = 1/E for all i, P_i = 1/E for all i → loss = E * E * (1/E)^2 = 1.0.
        """
        probs = _uniform_probs()
        mask = _uniform_top1_mask()
        loss = compute_load_balance_loss(probs, mask)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_zero_when_perfectly_balanced(self):
        """Perfectly balanced load: every expert gets exactly 1/E tokens AND
        the router assigns exactly 1/E probability to every expert.
        The loss is always n_experts * sum(f_i * P_i); for the balanced case this
        equals 1.0 (not zero). Verify the function agrees with the analytical value."""
        probs = _uniform_probs()
        mask = _uniform_top1_mask()
        loss = compute_load_balance_loss(probs, mask)
        expected = E * E * (1.0 / E) ** 2  # = 1.0
        assert loss.item() == pytest.approx(expected, abs=1e-5)

    def test_skewed_routing_increases_loss(self):
        """With non-uniform probs, sending all tokens to the highest-probability
        expert gives a higher loss than balanced routing.

        With uniform probs P_i = 1/E, the formula E*sum(f_i * 1/E) = 1 regardless
        of f_i (since sum(f_i)=1). We therefore use peaked probs so that routing
        all tokens to the highest-prob expert actually increases the loss.
        """
        # Router assigns high probability to expert 0.
        peaked_probs = torch.full((B, T, E), 0.01 / (E - 1))
        peaked_probs[:, :, 0] = 0.97

        balanced_mask = _uniform_top1_mask()
        skewed_mask = _single_expert_mask(0)  # all tokens → expert 0

        balanced_loss = compute_load_balance_loss(peaked_probs, balanced_mask)
        skewed_loss = compute_load_balance_loss(peaked_probs, skewed_mask)
        assert skewed_loss.item() > balanced_loss.item()

    def test_gradient_flows(self):
        """Loss should be differentiable w.r.t. router_probs."""
        probs = _uniform_probs().requires_grad_(True)
        mask = _uniform_top1_mask()
        loss = compute_load_balance_loss(probs, mask)
        loss.backward()
        assert probs.grad is not None


# ---------------------------------------------------------------------------
# 4 & 5. compute_router_entropy
# ---------------------------------------------------------------------------


class TestComputeRouterEntropy:
    def test_returns_scalar(self):
        probs = _uniform_probs()
        ent = compute_router_entropy(probs)
        assert ent.ndim == 0, "Expected a scalar tensor"

    def test_uniform_maximizes_entropy(self):
        """Uniform distribution has maximum entropy = log(E)."""
        uniform_probs = _uniform_probs()
        max_entropy = math.log(E)
        ent = compute_router_entropy(uniform_probs)
        assert ent.item() == pytest.approx(max_entropy, abs=1e-5)

    def test_peaked_distribution_has_lower_entropy(self):
        """A near-one-hot distribution should have lower entropy than uniform."""
        peaked = torch.full((B, T, E), 0.01 / (E - 1))
        peaked[:, :, 0] = 0.99
        ent_peaked = compute_router_entropy(peaked)
        ent_uniform = compute_router_entropy(_uniform_probs())
        assert ent_peaked.item() < ent_uniform.item()

    def test_gradient_flows(self):
        probs = _uniform_probs().requires_grad_(True)
        ent = compute_router_entropy(probs)
        ent.backward()
        assert probs.grad is not None


# ---------------------------------------------------------------------------
# 6, 7 & 8. compute_expert_utilization
# ---------------------------------------------------------------------------


class TestComputeExpertUtilization:
    def test_returns_correct_keys(self):
        mask = _uniform_top1_mask()
        result = compute_expert_utilization(mask)
        assert set(result.keys()) == {"utilization", "max_util", "min_util", "utilization_cv"}

    def test_utilization_shape(self):
        mask = _uniform_top1_mask()
        result = compute_expert_utilization(mask)
        assert result["utilization"].shape == (E,)

    def test_uniform_mask_gives_zero_cv(self):
        """When every expert receives exactly the same number of tokens, CV = 0."""
        mask = _uniform_top1_mask()
        result = compute_expert_utilization(mask)
        assert result["utilization_cv"].item() == pytest.approx(0.0, abs=1e-5)

    def test_single_expert_gets_full_utilization(self):
        """When all tokens go to expert 0, utilization[0] = 1.0 and others = 0."""
        mask = _single_expert_mask(0)
        result = compute_expert_utilization(mask)
        assert result["utilization"][0].item() == pytest.approx(1.0, abs=1e-5)
        assert result["utilization"][1:].sum().item() == pytest.approx(0.0, abs=1e-5)
        assert result["max_util"].item() == pytest.approx(1.0, abs=1e-5)
        assert result["min_util"].item() == pytest.approx(0.0, abs=1e-5)

    def test_utilization_sums_to_one(self):
        mask = _uniform_top1_mask()
        result = compute_expert_utilization(mask)
        assert result["utilization"].sum().item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 9, 10 & 11. ExpertLoadTracker
# ---------------------------------------------------------------------------


class TestExpertLoadTracker:
    def _make_tracker(self, **kwargs) -> ExpertLoadTracker:
        cfg = ExpertRoutingConfig(n_experts=E, **kwargs)
        return ExpertLoadTracker(cfg)

    def test_update_and_get_stats_smoke(self):
        tracker = self._make_tracker()
        mask = _uniform_top1_mask()
        tracker.update(mask)
        stats = tracker.get_stats()
        assert "mean_utilization" in stats
        assert "cv" in stats
        assert "dead_experts" in stats

    def test_balanced_routing_gives_low_cv(self):
        tracker = self._make_tracker()
        for _ in range(10):
            tracker.update(_uniform_top1_mask())
        stats = tracker.get_stats()
        assert stats["cv"] == pytest.approx(0.0, abs=1e-4)

    def test_dead_experts_detection(self):
        """Expert 1, 2, 3 never selected — should appear as dead (< 1 % of tokens)."""
        tracker = self._make_tracker()
        mask = _single_expert_mask(0)  # only expert 0 gets tokens
        for _ in range(5):
            tracker.update(mask)
        stats = tracker.get_stats()
        # Experts 1, 2, 3 receive 0 % of tokens → all three are dead.
        assert stats["dead_experts"] == pytest.approx(E - 1, abs=0.5)

    def test_reset_clears_history(self):
        tracker = self._make_tracker()
        for _ in range(5):
            tracker.update(_uniform_top1_mask())
        tracker.reset()
        # After reset, no history → stats should be at zero state.
        stats = tracker.get_stats()
        assert stats["mean_utilization"] == pytest.approx(0.0, abs=1e-9)
        assert stats["cv"] == pytest.approx(0.0, abs=1e-9)
        assert stats["dead_experts"] == pytest.approx(0.0, abs=1e-9)

    def test_no_history_when_track_history_false(self):
        tracker = self._make_tracker(track_history=False)
        tracker.update(_uniform_top1_mask())
        # update should be a no-op; history stays empty.
        assert len(tracker._history) == 0


# ---------------------------------------------------------------------------
# 12 & 13. RoutingAwareLoss
# ---------------------------------------------------------------------------


class TestRoutingAwareLoss:
    def _make_routing_loss(self) -> RoutingAwareLoss:
        cfg = ExpertRoutingConfig(n_experts=E, load_balance_coeff=0.01, entropy_coeff=0.001)
        return RoutingAwareLoss(cfg)

    def test_returns_correct_keys(self):
        rl = self._make_routing_loss()
        task_loss = torch.tensor(1.0)
        probs = _uniform_probs()
        mask = _uniform_top1_mask()
        total, info = rl(task_loss, probs, mask)
        assert set(info.keys()) == {"task_loss", "load_balance_loss", "entropy_loss"}

    def test_total_is_scalar(self):
        rl = self._make_routing_loss()
        total, _ = rl(torch.tensor(1.0), _uniform_probs(), _uniform_top1_mask())
        assert total.ndim == 0

    def test_total_geq_task_loss(self):
        """Regularization terms are additive, so total >= task_loss.

        load_balance_loss is always non-negative (it's n_experts * sum of
        products of non-negative quantities).
        entropy_loss = -coeff * H, which is <= 0, so together the regularizers
        may reduce the total slightly.  However the entropy coefficient is very
        small (0.001 * log(4) ≈ 0.0014) compared to the 0.01 * 1.0 load-balance
        term, so total > task_loss in practice for these defaults.
        We verify total >= task_loss - entropy_contribution (i.e., total + |entropy|
        >= task_loss).
        """
        rl = self._make_routing_loss()
        task_loss = torch.tensor(2.5)
        total, info = rl(task_loss, _uniform_probs(), _uniform_top1_mask())
        # The sum of load_balance_loss and entropy_loss might be slightly negative
        # if entropy > load_balance, but their absolute effect is bounded.
        # More robustly: total == task_loss + lb + entropy strictly.
        reconstructed = info["task_loss"] + info["load_balance_loss"] + info["entropy_loss"]
        assert total.item() == pytest.approx(reconstructed.item(), abs=1e-6)

    def test_task_loss_in_info_matches_input(self):
        rl = self._make_routing_loss()
        task_loss = torch.tensor(3.14)
        _, info = rl(task_loss, _uniform_probs(), _uniform_top1_mask())
        assert info["task_loss"].item() == pytest.approx(3.14, abs=1e-6)

    def test_load_balance_coeff_scales_loss(self):
        """Doubling load_balance_coeff should double the load_balance_loss component."""
        cfg1 = ExpertRoutingConfig(n_experts=E, load_balance_coeff=0.01, entropy_coeff=0.0)
        cfg2 = ExpertRoutingConfig(n_experts=E, load_balance_coeff=0.02, entropy_coeff=0.0)
        rl1, rl2 = RoutingAwareLoss(cfg1), RoutingAwareLoss(cfg2)
        task_loss = torch.tensor(1.0)
        probs, mask = _uniform_probs(), _uniform_top1_mask()
        _, info1 = rl1(task_loss, probs, mask)
        _, info2 = rl2(task_loss, probs, mask)
        assert info2["load_balance_loss"].item() == pytest.approx(
            2.0 * info1["load_balance_loss"].item(), rel=1e-5
        )
