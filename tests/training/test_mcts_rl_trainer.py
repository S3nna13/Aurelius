"""Unit and integration tests for src/training/mcts_rl_trainer.py.

Coverage:
    1.  test_config_defaults              -- MCTSRLConfig default field values
    2.  test_normalize_visits_sums_to_one -- output is a valid probability distribution
    3.  test_normalize_visits_temperature -- τ→0 sharpens to argmax, τ→∞ flattens
    4.  test_policy_loss_scalar           -- policy_loss returns a 0-dim tensor
    5.  test_policy_loss_zero             -- when log_policy == log(mcts_policy), loss = entropy
    6.  test_policy_loss_positive         -- cross-entropy >= entropy (never negative)
    7.  test_value_loss_scalar            -- value_loss returns a 0-dim tensor
    8.  test_value_loss_zero              -- predicted == target → loss = 0
    9.  test_compute_loss_keys            -- compute_loss returns required dict keys
    10. test_compute_loss_finite          -- all returned tensors are finite scalars
    11. test_uct_score_exploration        -- higher prior → higher UCT score
    12. test_uct_score_exploitation       -- higher Q → higher UCT score
    13. test_mcts_node_q_value            -- q_value = value_sum / visit_count
    14. test_gradient_flows               -- backward on total loss propagates gradients
    Integration:
        test_integration_forward_backward -- B=4, A=8 full compute_loss + backward
"""

from __future__ import annotations

import math

import pytest
import torch

from src.training.mcts_rl_trainer import (
    MCTSNode,
    MCTSRLConfig,
    MCTSRLTrainer,
    MCTSStats,
)
from src.training import TRAINING_REGISTRY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stats(B: int = 2, A: int = 4, *, seed: int = 0) -> list[MCTSStats]:
    """Build a list of synthetic MCTSStats with random visit counts and Q-values."""
    torch.manual_seed(seed)
    stats = []
    for i in range(B):
        visit_counts  = torch.randint(1, 20, (A,)).float()
        action_values = torch.randn(A)
        # Normalise visit counts into a valid policy target.
        mcts_policy   = (visit_counts / visit_counts.sum())
        stats.append(MCTSStats(
            state_id=i,
            visit_counts=visit_counts,
            action_values=action_values,
            mcts_policy=mcts_policy,
        ))
    return stats


def _make_log_policy(B: int, A: int, *, seed: int = 1) -> torch.Tensor:
    """Create a valid [B, A] log-softmax tensor (log-probabilities)."""
    torch.manual_seed(seed)
    return torch.nn.functional.log_softmax(torch.randn(B, A, requires_grad=True), dim=-1)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = MCTSRLConfig()
    assert cfg.temperature   == pytest.approx(1.0)
    assert cfg.value_weight  == pytest.approx(1.0)
    assert cfg.n_simulations == 50
    assert cfg.c_puct        == pytest.approx(1.4)
    assert cfg.discount      == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# 2. test_normalize_visits_sums_to_one
# ---------------------------------------------------------------------------

def test_normalize_visits_sums_to_one():
    trainer = MCTSRLTrainer()
    for seed in (0, 7, 42):
        torch.manual_seed(seed)
        counts = torch.randint(1, 50, (8,)).float()
        policy = trainer.normalize_visits(counts)
        assert policy.shape == (8,), f"unexpected shape {policy.shape}"
        assert policy.sum().item() == pytest.approx(1.0, abs=1e-5), (
            f"policy does not sum to 1: {policy.sum().item()}"
        )
        assert (policy >= 0).all(), "policy contains negative entries"


# ---------------------------------------------------------------------------
# 3. test_normalize_visits_temperature
# ---------------------------------------------------------------------------

def test_normalize_visits_temperature():
    # τ → 0: deterministic — one-hot at argmax.
    counts = torch.tensor([1.0, 5.0, 2.0, 8.0])
    trainer_cold = MCTSRLTrainer(MCTSRLConfig(temperature=0.0))
    policy_cold  = trainer_cold.normalize_visits(counts)
    assert policy_cold.argmax().item() == 3, "cold policy should argmax at index 3"
    assert policy_cold[3].item() == pytest.approx(1.0), "cold policy should be one-hot"

    # τ → large: near-uniform.
    trainer_hot = MCTSRLTrainer(MCTSRLConfig(temperature=100.0))
    policy_hot  = trainer_hot.normalize_visits(counts)
    expected_uniform = 1.0 / len(counts)
    for p in policy_hot.tolist():
        assert abs(p - expected_uniform) < 0.05, (
            f"high-temperature policy not near uniform: {policy_hot.tolist()}"
        )


# ---------------------------------------------------------------------------
# 4. test_policy_loss_scalar
# ---------------------------------------------------------------------------

def test_policy_loss_scalar():
    B, A = 4, 6
    trainer   = MCTSRLTrainer()
    log_p     = _make_log_policy(B, A)
    mcts_p    = torch.softmax(torch.randn(B, A), dim=-1)
    loss      = trainer.policy_loss(log_p, mcts_p)
    assert loss.dim() == 0, f"expected scalar loss (0-dim), got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 5. test_policy_loss_zero — when log_policy == log(mcts_policy) loss = H(π)
# ---------------------------------------------------------------------------

def test_policy_loss_zero():
    """When the model policy exactly matches the MCTS policy the loss equals
    the entropy of the target distribution (cross-entropy with itself = entropy)."""
    torch.manual_seed(3)
    B, A = 3, 5
    trainer = MCTSRLTrainer()

    mcts_p  = torch.softmax(torch.randn(B, A), dim=-1)
    log_p   = mcts_p.log()  # exact match

    loss    = trainer.policy_loss(log_p, mcts_p)
    entropy = -(mcts_p * log_p).sum(dim=1).mean()

    assert loss.item() == pytest.approx(entropy.item(), rel=1e-5), (
        f"expected loss == entropy ({entropy.item():.6f}), got {loss.item():.6f}"
    )


# ---------------------------------------------------------------------------
# 6. test_policy_loss_positive — cross-entropy >= entropy
# ---------------------------------------------------------------------------

def test_policy_loss_positive():
    """Cross-entropy H(π_MCTS, π_θ) >= H(π_MCTS) always.  The *additional*
    cost beyond entropy is non-negative (Gibbs inequality)."""
    torch.manual_seed(5)
    B, A = 4, 8
    trainer = MCTSRLTrainer()

    mcts_p  = torch.softmax(torch.randn(B, A), dim=-1)
    log_p   = torch.nn.functional.log_softmax(torch.randn(B, A), dim=-1)

    loss    = trainer.policy_loss(log_p, mcts_p)
    entropy = -(mcts_p * mcts_p.log()).sum(dim=1).mean()

    assert loss.item() >= entropy.item() - 1e-5, (
        f"cross-entropy ({loss.item():.6f}) must be >= entropy ({entropy.item():.6f})"
    )


# ---------------------------------------------------------------------------
# 7. test_value_loss_scalar
# ---------------------------------------------------------------------------

def test_value_loss_scalar():
    trainer = MCTSRLTrainer()
    B = 6
    pred   = torch.randn(B, requires_grad=True)
    target = torch.randn(B)
    loss   = trainer.value_loss(pred, target)
    assert loss.dim() == 0, f"expected scalar (0-dim), got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 8. test_value_loss_zero — predicted == target → loss = 0
# ---------------------------------------------------------------------------

def test_value_loss_zero():
    trainer = MCTSRLTrainer()
    B       = 4
    target  = torch.randn(B)
    loss    = trainer.value_loss(target.clone(), target)
    assert loss.item() == pytest.approx(0.0, abs=1e-6), (
        f"expected zero loss when pred == target, got {loss.item()}"
    )


# ---------------------------------------------------------------------------
# 9. test_compute_loss_keys
# ---------------------------------------------------------------------------

def test_compute_loss_keys():
    B, A  = 3, 5
    trainer = MCTSRLTrainer()
    log_p   = _make_log_policy(B, A)
    pred_v  = torch.randn(B, requires_grad=True)
    stats   = _make_stats(B, A)

    out = trainer.compute_loss(log_p, pred_v, stats)
    required = {"loss", "policy_loss", "value_loss", "kl_from_mcts"}
    assert required <= out.keys(), f"missing keys: {required - out.keys()}"


# ---------------------------------------------------------------------------
# 10. test_compute_loss_finite
# ---------------------------------------------------------------------------

def test_compute_loss_finite():
    B, A  = 4, 6
    trainer = MCTSRLTrainer()
    log_p   = _make_log_policy(B, A)
    pred_v  = torch.randn(B, requires_grad=True)
    stats   = _make_stats(B, A)

    out = trainer.compute_loss(log_p, pred_v, stats)
    for key, val in out.items():
        assert val.dim() == 0, f"'{key}' must be a scalar, got shape {val.shape}"
        assert math.isfinite(val.item()), f"'{key}' is not finite: {val.item()}"


# ---------------------------------------------------------------------------
# 11. test_uct_score_exploration — higher prior → higher UCT score
# ---------------------------------------------------------------------------

def test_uct_score_exploration():
    """With identical Q-values, the node with a higher prior should score higher."""
    trainer = MCTSRLTrainer()
    parent_visits = 10

    node_low  = MCTSNode(state_id=0, visit_count=2, value_sum=0.4, prior=0.1)
    node_high = MCTSNode(state_id=1, visit_count=2, value_sum=0.4, prior=0.9)

    score_low  = trainer.uct_score(node_low,  parent_visits)
    score_high = trainer.uct_score(node_high, parent_visits)

    assert score_high > score_low, (
        f"higher prior should yield higher UCT: low={score_low:.4f}, high={score_high:.4f}"
    )


# ---------------------------------------------------------------------------
# 12. test_uct_score_exploitation — higher Q → higher UCT score
# ---------------------------------------------------------------------------

def test_uct_score_exploitation():
    """With identical priors and visit counts, the node with higher Q scores higher."""
    trainer = MCTSRLTrainer()
    parent_visits = 20

    node_low  = MCTSNode(state_id=0, visit_count=5, value_sum=0.5,  prior=0.3)
    node_high = MCTSNode(state_id=1, visit_count=5, value_sum=4.5,  prior=0.3)
    # q_low  = 0.5 / 5  = 0.1
    # q_high = 4.5 / 5  = 0.9

    score_low  = trainer.uct_score(node_low,  parent_visits)
    score_high = trainer.uct_score(node_high, parent_visits)

    assert score_high > score_low, (
        f"higher Q should yield higher UCT: low={score_low:.4f}, high={score_high:.4f}"
    )


# ---------------------------------------------------------------------------
# 13. test_mcts_node_q_value
# ---------------------------------------------------------------------------

def test_mcts_node_q_value():
    """q_value property must equal value_sum / visit_count (with ε for zero visits)."""
    node = MCTSNode(state_id=7, visit_count=4, value_sum=2.0)
    expected = 2.0 / (4 + 1e-8)
    assert node.q_value == pytest.approx(expected, rel=1e-6), (
        f"q_value mismatch: expected {expected}, got {node.q_value}"
    )

    # Zero visits: should not raise; returns ~0.
    node_zero = MCTSNode(state_id=8, visit_count=0, value_sum=0.0)
    assert math.isfinite(node_zero.q_value), "q_value with zero visits should be finite"


# ---------------------------------------------------------------------------
# 14. test_gradient_flows
# ---------------------------------------------------------------------------

def test_gradient_flows():
    """Backward on the total loss must propagate finite gradients."""
    B, A = 3, 6
    trainer = MCTSRLTrainer()

    logits_p = torch.randn(B, A, requires_grad=True)
    log_p    = torch.nn.functional.log_softmax(logits_p, dim=-1)
    pred_v   = torch.randn(B, requires_grad=True)
    stats    = _make_stats(B, A, seed=99)

    out = trainer.compute_loss(log_p, pred_v, stats)
    out["loss"].backward()

    assert logits_p.grad is not None, "gradients did not reach logits_p"
    assert pred_v.grad   is not None, "gradients did not reach pred_v"
    assert torch.isfinite(logits_p.grad).all(), "non-finite gradient in logits_p"
    assert torch.isfinite(pred_v.grad).all(),   "non-finite gradient in pred_v"


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_integration_forward_backward():
    """Full compute_loss + backward with B=4, A=8; verifies shapes and numerics."""
    B, A = 4, 8
    torch.manual_seed(12345)

    cfg     = MCTSRLConfig(temperature=1.0, value_weight=0.5, n_simulations=50, c_puct=1.4)
    trainer = MCTSRLTrainer(cfg)

    # Build realistic MCTS stats: random visit counts and Q-values.
    stats = []
    for i in range(B):
        visit_counts  = torch.randint(1, 100, (A,)).float()
        action_values = torch.randn(A) * 0.5           # bounded Q-values
        mcts_policy   = trainer.normalize_visits(visit_counts)
        stats.append(MCTSStats(
            state_id=i,
            visit_counts=visit_counts,
            action_values=action_values,
            mcts_policy=mcts_policy,
        ))

    # Model outputs: log-policy and value prediction.
    logits = torch.randn(B, A, requires_grad=True)
    log_p  = torch.nn.functional.log_softmax(logits, dim=-1)   # [B, A]
    pred_v = torch.randn(B, requires_grad=True)                  # [B]

    out = trainer.compute_loss(log_p, pred_v, stats)

    # --- Shape checks ---
    required_keys = {"loss", "policy_loss", "value_loss", "kl_from_mcts"}
    assert required_keys <= out.keys(), f"missing keys: {required_keys - out.keys()}"
    for key, val in out.items():
        assert val.dim() == 0, f"'{key}' must be scalar (0-dim), got {val.shape}"

    # --- Finiteness ---
    for key, val in out.items():
        assert math.isfinite(val.item()), f"'{key}' is not finite: {val.item()}"

    # --- Loss ordering: total = policy_loss + 0.5 * value_loss ---
    expected_total = out["policy_loss"].item() + 0.5 * out["value_loss"].item()
    assert out["loss"].item() == pytest.approx(expected_total, rel=1e-5), (
        f"total loss mismatch: got {out['loss'].item()}, expected {expected_total}"
    )

    # --- KL non-negative (Gibbs inequality) ---
    assert out["kl_from_mcts"].item() >= -1e-4, (
        f"KL divergence must be >= 0, got {out['kl_from_mcts'].item()}"
    )

    # --- Backward ---
    out["loss"].backward()
    assert logits.grad is not None, "gradients missing from logits"
    assert pred_v.grad  is not None, "gradients missing from pred_v"
    assert torch.isfinite(logits.grad).all(), "non-finite gradient in logits"
    assert torch.isfinite(pred_v.grad).all(), "non-finite gradient in pred_v"

    # --- statistics() method ---
    diag = trainer.statistics(stats)
    assert "mean_visit_count" in diag
    assert "policy_entropy"   in diag
    assert "mean_q_value"     in diag
    for k, v in diag.items():
        assert math.isfinite(v), f"statistics['{k}'] is not finite: {v}"
    assert diag["mean_visit_count"] > 0, "mean_visit_count should be positive"
    assert diag["policy_entropy"]   >= 0, "entropy must be non-negative"

    # --- UCT score smoke test ---
    node = MCTSNode(state_id=0, visit_count=10, value_sum=3.0, prior=0.25)
    score = trainer.uct_score(node, parent_visits=50)
    assert math.isfinite(score), f"UCT score is not finite: {score}"

    # --- Registry ---
    assert "mcts_rl" in TRAINING_REGISTRY, "'mcts_rl' not found in TRAINING_REGISTRY"
    assert TRAINING_REGISTRY["mcts_rl"] is MCTSRLTrainer
