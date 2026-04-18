"""
Tests for src/alignment/multi_objective_rlhf.py
16+ tests covering all public classes and methods.

Tiny inline reward models; n_obj=2, B=2, T=4, d_model=16.
"""

import math
import torch
import torch.nn as nn
import pytest

from src.alignment.multi_objective_rlhf import (
    MultiObjectiveReward,
    ParetoFront,
    WeightSampler,
    MOPPOTrainer,
    RewardWeightScheduler,
    MOConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B = 2
T = 4
D = 16
N_OBJ = 2
VOCAB = 32


class TinyRewardModel(nn.Module):
    """Maps [B, T] token ids → [B] scalar rewards via embedding + mean + linear."""

    def __init__(self, d_model: int = D, vocab: int = VOCAB, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(vocab, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # [B, T] → [B, T, D] → [B, D] → [B, 1] → [B]
        x = self.emb(input_ids).mean(dim=1)
        return self.head(x).squeeze(-1)


class TinyPolicy(nn.Module):
    """Maps [B, T] token ids → [B, T, V] logits."""

    def __init__(self, d_model: int = D, vocab: int = VOCAB, seed: int = 42) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.emb = nn.Embedding(vocab, d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(input_ids)
        return self.head(x)


def make_reward_model() -> MultiObjectiveReward:
    rm1 = TinyRewardModel(seed=1)
    rm2 = TinyRewardModel(seed=2)
    return MultiObjectiveReward(
        reward_models=[rm1, rm2],
        objective_names=["helpfulness", "safety"],
    )


def make_input_ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, VOCAB, (B, T))


# ---------------------------------------------------------------------------
# MultiObjectiveReward tests
# ---------------------------------------------------------------------------


def test_multi_objective_reward_forward_keys():
    """forward() returns a dict containing all objective names."""
    rm = make_reward_model()
    ids = make_input_ids()
    out = rm(ids)
    assert set(out.keys()) == {"helpfulness", "safety"}


def test_multi_objective_reward_forward_shapes():
    """Each reward tensor has shape [B]."""
    rm = make_reward_model()
    ids = make_input_ids()
    out = rm(ids)
    for name, r in out.items():
        assert r.shape == (B,), f"Expected shape ({B},) for {name}, got {r.shape}"


def test_multi_objective_reward_scalarize_shape():
    """scalarize() returns tensor of shape [B]."""
    rm = make_reward_model()
    ids = make_input_ids()
    out = rm(ids)
    w = torch.tensor([0.6, 0.4])
    scalar = rm.scalarize(out, w)
    assert scalar.shape == (B,)


def test_multi_objective_reward_scalarize_unit_weights():
    """scalarize with [1, 0] extracts exactly the first objective."""
    rm = make_reward_model()
    ids = make_input_ids()
    out = rm(ids)
    w_first = torch.tensor([1.0, 0.0])
    scalar = rm.scalarize(out, w_first)
    assert torch.allclose(scalar, out["helpfulness"], atol=1e-5)


def test_multi_objective_reward_scalarize_second_unit():
    """scalarize with [0, 1] extracts exactly the second objective."""
    rm = make_reward_model()
    ids = make_input_ids()
    out = rm(ids)
    w_second = torch.tensor([0.0, 1.0])
    scalar = rm.scalarize(out, w_second)
    assert torch.allclose(scalar, out["safety"], atol=1e-5)


def test_multi_objective_reward_scalarize_weighted_sum():
    """scalarize produces weighted sum manually verified."""
    rm = make_reward_model()
    ids = make_input_ids()
    out = rm(ids)
    w = torch.tensor([0.3, 0.7])
    scalar = rm.scalarize(out, w)
    expected = 0.3 * out["helpfulness"] + 0.7 * out["safety"]
    assert torch.allclose(scalar, expected, atol=1e-5)


def test_multi_objective_reward_mismatched_args():
    """Mismatched reward_models / objective_names raises ValueError."""
    with pytest.raises(ValueError):
        MultiObjectiveReward(
            reward_models=[TinyRewardModel()],
            objective_names=["a", "b"],
        )


# ---------------------------------------------------------------------------
# ParetoFront tests
# ---------------------------------------------------------------------------


def test_pareto_front_update_adds_solution():
    """update() adds at least one solution to the front."""
    pf = ParetoFront()
    w = torch.tensor([0.5, 0.5])
    r = torch.tensor([0.8, 0.6])
    pf.update(w, r, loss=0.5)
    assert len(pf.solutions) >= 1


def test_pareto_front_is_pareto_dominant_true():
    """r_new = [1, 1] dominates r_old = [0.5, 0.5]."""
    r_new = torch.tensor([1.0, 1.0])
    r_old = torch.tensor([0.5, 0.5])
    assert ParetoFront.is_pareto_dominant(r_new, r_old) is True


def test_pareto_front_is_pareto_dominant_false_equal():
    """Equal rewards do not dominate each other."""
    r = torch.tensor([0.5, 0.5])
    assert ParetoFront.is_pareto_dominant(r, r.clone()) is False


def test_pareto_front_is_pareto_dominant_false_partial():
    """r_new = [1, 0] does NOT dominate r_old = [0, 1] (neither dominates)."""
    r_new = torch.tensor([1.0, 0.0])
    r_old = torch.tensor([0.0, 1.0])
    assert ParetoFront.is_pareto_dominant(r_new, r_old) is False


def test_pareto_front_get_pareto_front_removes_dominated():
    """get_pareto_front() drops dominated solutions."""
    pf = ParetoFront()
    # A dominates B
    w_a = torch.tensor([0.5, 0.5])
    r_a = torch.tensor([1.0, 1.0])   # dominates everything below

    w_b = torch.tensor([0.5, 0.5])
    r_b = torch.tensor([0.3, 0.3])   # dominated by A

    pf.update(w_a, r_a, loss=0.2)
    pf.update(w_b, r_b, loss=0.8)

    front = pf.get_pareto_front()
    rewards_in_front = [tuple(s["rewards"].tolist()) for s in front]
    # The dominated solution should not be in the front
    assert tuple(r_a.tolist()) in rewards_in_front
    assert tuple(r_b.tolist()) not in rewards_in_front


def test_pareto_front_hypervolume_positive():
    """hypervolume_indicator returns a positive float for a valid front."""
    pf = ParetoFront()
    pf.update(torch.tensor([1.0, 0.0]), torch.tensor([0.9, 0.2]), loss=0.1)
    pf.update(torch.tensor([0.0, 1.0]), torch.tensor([0.2, 0.9]), loss=0.1)
    ref = torch.tensor([0.0, 0.0])
    hv = pf.hypervolume_indicator(ref)
    assert hv > 0.0, f"Expected positive hypervolume, got {hv}"


def test_pareto_front_hypervolume_empty_returns_zero():
    """hypervolume_indicator on empty front returns 0."""
    pf = ParetoFront()
    hv = pf.hypervolume_indicator(torch.tensor([0.0, 0.0]))
    assert hv == 0.0


# ---------------------------------------------------------------------------
# WeightSampler tests
# ---------------------------------------------------------------------------


def test_weight_sampler_uniform_simplex_shape():
    """uniform_simplex returns [n_samples, n_obj] tensor."""
    ws = WeightSampler()
    out = ws.uniform_simplex(N_OBJ, 8)
    assert out.shape == (8, N_OBJ)


def test_weight_sampler_uniform_simplex_sums_to_one():
    """Each row of uniform_simplex sums to 1."""
    ws = WeightSampler()
    out = ws.uniform_simplex(N_OBJ, 20)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(20), atol=1e-5)


def test_weight_sampler_linear_scalarization_shape():
    """linear_scalarization_weights returns [n_steps, n_obj]."""
    ws = WeightSampler()
    out = ws.linear_scalarization_weights(N_OBJ, 5)
    assert out.shape == (5, N_OBJ)


def test_weight_sampler_linear_scalarization_sums_to_one():
    """Each row of linear_scalarization_weights sums to 1."""
    ws = WeightSampler()
    out = ws.linear_scalarization_weights(N_OBJ, 5)
    row_sums = out.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones(5), atol=1e-5)


def test_weight_sampler_chebyshev_shape_and_sum():
    """chebyshev_weights returns [n_obj] summing to 1."""
    ws = WeightSampler()
    ideal = torch.tensor([1.0, 1.0])
    gap = torch.tensor([0.3, 0.7])
    w = ws.chebyshev_weights(ideal, gap)
    assert w.shape == (N_OBJ,)
    assert abs(float(w.sum()) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# MOPPOTrainer tests
# ---------------------------------------------------------------------------


def test_mopppo_trainer_policy_step_finite_loss():
    """policy_step returns a finite scalar loss."""
    policy = TinyPolicy()
    rm = make_reward_model()
    trainer = MOPPOTrainer(policy, rm, lr=1e-4, n_obj=N_OBJ)
    ids = make_input_ids()
    w = torch.tensor([0.5, 0.5])
    loss, rewards_dict = trainer.policy_step(ids, w)
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.shape == torch.Size([])


def test_mopppo_trainer_policy_step_rewards_dict():
    """policy_step returns a rewards_dict with correct objective keys."""
    policy = TinyPolicy()
    rm = make_reward_model()
    trainer = MOPPOTrainer(policy, rm, lr=1e-4, n_obj=N_OBJ)
    ids = make_input_ids()
    w = torch.tensor([0.5, 0.5])
    _, rewards_dict = trainer.policy_step(ids, w)
    assert set(rewards_dict.keys()) == {"helpfulness", "safety"}


def test_mopppo_trainer_multi_objective_step_updates_pareto():
    """multi_objective_step adds entries to the Pareto front."""
    policy = TinyPolicy()
    rm = make_reward_model()
    trainer = MOPPOTrainer(policy, rm, lr=1e-4, n_obj=N_OBJ)
    ids = make_input_ids()
    result = trainer.multi_objective_step(ids, n_weight_samples=3)
    assert result["pareto_size"] >= 1
    assert math.isfinite(result["total_loss"])


def test_mopppo_trainer_get_best_weights_shape():
    """get_best_weights returns [n_obj] tensor after Pareto front is populated."""
    policy = TinyPolicy()
    rm = make_reward_model()
    trainer = MOPPOTrainer(policy, rm, lr=1e-4, n_obj=N_OBJ)
    ids = make_input_ids()
    trainer.multi_objective_step(ids, n_weight_samples=4)
    pref = torch.tensor([1.0, 0.0])
    best_w = trainer.get_best_weights(pref)
    assert best_w.shape == (N_OBJ,)


def test_mopppo_trainer_get_best_weights_no_front():
    """get_best_weights on empty Pareto front returns uniform weights."""
    policy = TinyPolicy()
    rm = make_reward_model()
    trainer = MOPPOTrainer(policy, rm, lr=1e-4, n_obj=N_OBJ)
    pref = torch.tensor([1.0, 0.0])
    best_w = trainer.get_best_weights(pref)
    expected = torch.ones(N_OBJ) / N_OBJ
    assert torch.allclose(best_w, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# RewardWeightScheduler tests
# ---------------------------------------------------------------------------


def test_reward_weight_scheduler_uniform_constant():
    """'uniform' schedule always returns equal weights."""
    sched = RewardWeightScheduler(N_OBJ, schedule="uniform")
    w0 = sched.step(0)
    w5 = sched.step(5)
    assert torch.allclose(w0, w5, atol=1e-5)
    assert torch.allclose(w0, torch.ones(N_OBJ) / N_OBJ, atol=1e-5)


def test_reward_weight_scheduler_alternating_focuses():
    """'alternating' schedule focuses on a different objective each step."""
    sched = RewardWeightScheduler(N_OBJ, schedule="alternating")
    w0 = sched.step(0)
    w1 = sched.step(1)
    # Different maximum-weight indices
    assert int(w0.argmax()) != int(w1.argmax())
    # Each step has exactly one non-zero weight
    assert float(w0.max()) == 1.0
    assert float(w1.max()) == 1.0


def test_reward_weight_scheduler_alternating_cycles():
    """'alternating' schedule cycles back to objective 0 after n_obj steps."""
    n = 3
    sched = RewardWeightScheduler(n, schedule="alternating")
    for t in range(6):
        w = sched.step(t)
        assert int(w.argmax()) == t % n


def test_reward_weight_scheduler_output_sums_to_one():
    """All schedule types produce weight vectors summing to 1."""
    ids = make_input_ids()
    for schedule in ("uniform", "alternating", "random"):
        sched = RewardWeightScheduler(N_OBJ, schedule=schedule)
        for t in range(5):
            w = sched.step(t)
            assert abs(float(w.sum()) - 1.0) < 1e-5, (
                f"schedule={schedule} step={t}: sum={float(w.sum())}"
            )


def test_reward_weight_scheduler_random_shape():
    """'random' schedule returns [n_obj] tensor."""
    sched = RewardWeightScheduler(N_OBJ, schedule="random")
    w = sched.step(0)
    assert w.shape == (N_OBJ,)


def test_reward_weight_scheduler_invalid_schedule():
    """Invalid schedule name raises ValueError."""
    with pytest.raises(ValueError):
        RewardWeightScheduler(N_OBJ, schedule="bogus")


# ---------------------------------------------------------------------------
# MOConfig tests
# ---------------------------------------------------------------------------


def test_moconfig_defaults():
    """MOConfig has the correct default field values."""
    cfg = MOConfig()
    assert cfg.n_objectives == 3
    assert cfg.objective_names == ["helpfulness", "safety", "harmlessness"]
    assert cfg.lr == pytest.approx(1e-4)
    assert cfg.n_weight_samples == 4
    assert cfg.schedule == "uniform"


def test_moconfig_custom():
    """MOConfig accepts custom values."""
    cfg = MOConfig(n_objectives=2, objective_names=["a", "b"], lr=3e-4)
    assert cfg.n_objectives == 2
    assert cfg.objective_names == ["a", "b"]
    assert cfg.lr == pytest.approx(3e-4)
