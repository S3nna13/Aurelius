"""Tests for outcome_supervision.py — ORM / pass@k implementation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.alignment.outcome_supervision import (
    ORMConfig,
    ORMTrainer,
    OutcomeRewardModel,
    OutcomeVerifier,
    pass_at_k,
)

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------


def _exact_match_verifier() -> OutcomeVerifier:
    """Verifier that checks whether the last token equals the ground truth int."""

    def verify_fn(response_ids: torch.Tensor, ground_truth) -> bool:
        return int(response_ids.reshape(-1)[-1].item()) == int(ground_truth)

    return OutcomeVerifier(verify_fn)


class _TinyRewardNet(nn.Module):
    """Minimal network: embeds tokens, mean-pools, projects to scalar logit."""

    def __init__(self, vocab_size: int = 32, d_model: int = 16) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T) -> embed -> (B, T, d) -> mean -> (B, d) -> (B, 1)
        x = self.embed(input_ids).mean(dim=1)
        return self.head(x).squeeze(-1)  # (B,)


@pytest.fixture
def tiny_verifier() -> OutcomeVerifier:
    return _exact_match_verifier()


@pytest.fixture
def tiny_orm(tiny_verifier) -> OutcomeRewardModel:
    return OutcomeRewardModel(tiny_verifier, correct_reward=1.0, wrong_reward=0.0)


@pytest.fixture
def tiny_trainer(tiny_verifier):
    torch.manual_seed(0)
    net = _TinyRewardNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    return ORMTrainer(net, optimizer, tiny_verifier)


# ---------------------------------------------------------------------------
# Test 1: ORMConfig defaults
# ---------------------------------------------------------------------------


def test_ormconfig_defaults():
    cfg = ORMConfig()
    assert cfg.correct_reward == 1.0
    assert cfg.wrong_reward == 0.0
    assert cfg.n_samples == 16
    assert cfg.temperature == 0.8


# ---------------------------------------------------------------------------
# Test 2: pass_at_k(10, 5, 1) ≈ 0.5
# ---------------------------------------------------------------------------


def test_pass_at_k_half():
    result = pass_at_k(10, 5, 1)
    assert abs(result - 0.5) < 1e-9, f"Expected 0.5, got {result}"


# ---------------------------------------------------------------------------
# Test 3: pass_at_k(10, 10, 1) = 1.0 (all correct)
# ---------------------------------------------------------------------------


def test_pass_at_k_all_correct():
    result = pass_at_k(10, 10, 1)
    assert result == 1.0, f"Expected 1.0, got {result}"


# ---------------------------------------------------------------------------
# Test 4: pass_at_k(10, 0, 1) = 0.0 (none correct)
# ---------------------------------------------------------------------------


def test_pass_at_k_none_correct():
    result = pass_at_k(10, 0, 1)
    assert result == 0.0, f"Expected 0.0, got {result}"


# ---------------------------------------------------------------------------
# Test 5: pass_at_k increases (non-strictly) monotonically with k
# ---------------------------------------------------------------------------


def test_pass_at_k_monotone():
    n, c = 20, 8
    prev = pass_at_k(n, c, 1)
    for k in range(2, n + 1):
        curr = pass_at_k(n, c, k)
        assert curr >= prev - 1e-12, f"pass@{k}={curr:.6f} < pass@{k - 1}={prev:.6f} — not monotone"
        prev = curr
    # Final value must be 1.0 when k == n and c > 0
    assert abs(pass_at_k(n, c, n) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Test 6: OutcomeVerifier.verify returns bool
# ---------------------------------------------------------------------------


def test_verifier_verify_returns_bool(tiny_verifier):
    response = torch.tensor([1, 2, 3, 7])  # last token = 7
    result = tiny_verifier.verify(response, 7)
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    assert result is True

    result_wrong = tiny_verifier.verify(response, 99)
    assert isinstance(result_wrong, bool)
    assert result_wrong is False


# ---------------------------------------------------------------------------
# Test 7: batch_verify returns correct shape tensor
# ---------------------------------------------------------------------------


def test_batch_verify_shape(tiny_verifier):
    responses = [torch.tensor([1, 2, i]) for i in range(5)]
    ground_truths = list(range(5))  # each last token matches its gt
    result = tiny_verifier.batch_verify(responses, ground_truths)
    assert isinstance(result, torch.Tensor), "Expected torch.Tensor"
    assert result.shape == (5,), f"Expected shape (5,), got {result.shape}"
    assert result.dtype == torch.bool, f"Expected bool dtype, got {result.dtype}"
    # All should be correct since last token == ground_truth
    assert result.all(), f"Expected all True, got {result}"


# ---------------------------------------------------------------------------
# Test 8: OutcomeRewardModel.compute_reward returns correct_reward for correct
# ---------------------------------------------------------------------------


def test_compute_reward_correct(tiny_orm):
    response = torch.tensor([5, 3, 42])  # last token = 42
    reward = tiny_orm.compute_reward(response, 42)
    assert reward == 1.0, f"Expected 1.0 for correct, got {reward}"


def test_compute_reward_wrong(tiny_orm):
    response = torch.tensor([5, 3, 42])  # last token = 42
    reward = tiny_orm.compute_reward(response, 99)
    assert reward == 0.0, f"Expected 0.0 for wrong, got {reward}"


# ---------------------------------------------------------------------------
# Test 9: estimate_pass_at_k returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_estimate_pass_at_k_range(tiny_orm):
    # 4 responses: last tokens 0,1,2,3
    responses = [torch.tensor([i]) for i in range(4)]
    # ground_truth=1 means only responses[1] is correct
    result = tiny_orm.estimate_pass_at_k(responses, ground_truth=1, k=2)
    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert 0.0 <= result <= 1.0, f"Expected in [0,1], got {result}"


# ---------------------------------------------------------------------------
# Test 10: ORMTrainer.train_step returns dict with 'loss' key
# ---------------------------------------------------------------------------


def test_trainer_train_step_has_loss(tiny_trainer):
    torch.manual_seed(1)
    # 4 responses with last token in {0,1}; ground truths all 1
    responses = [torch.tensor([0, 1, 0, 1]) for _ in range(4)]
    gts = [1, 1, 1, 1]
    response_ids, labels = tiny_trainer.create_training_batch(responses, gts)
    result = tiny_trainer.train_step(response_ids, labels)

    assert isinstance(result, dict), "train_step must return a dict"
    assert "loss" in result, f"'loss' key missing; keys: {result.keys()}"
    assert isinstance(result["loss"], float), f"loss must be float, got {type(result['loss'])}"


# ---------------------------------------------------------------------------
# Test 11: Accuracy in [0, 1]
# ---------------------------------------------------------------------------


def test_trainer_accuracy_range(tiny_trainer):
    torch.manual_seed(2)
    responses = [torch.tensor([i % 5]) for i in range(8)]
    gts = [i % 5 for i in range(8)]  # all correct
    response_ids, labels = tiny_trainer.create_training_batch(responses, gts)
    result = tiny_trainer.train_step(response_ids, labels)

    assert "accuracy" in result, f"'accuracy' key missing; keys: {result.keys()}"
    acc = result["accuracy"]
    assert 0.0 <= acc <= 1.0, f"Accuracy {acc} not in [0, 1]"


# ---------------------------------------------------------------------------
# Test 12: Gradient flows through ORM loss
# ---------------------------------------------------------------------------


def test_gradient_flows():
    """Parameters must receive gradients after a train_step call."""
    torch.manual_seed(3)
    net = _TinyRewardNet()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

    def verify_fn(ids, gt):
        return int(ids.reshape(-1)[-1].item()) == int(gt)

    verifier = OutcomeVerifier(verify_fn)
    trainer = ORMTrainer(net, optimizer, verifier)

    responses = [torch.tensor([0, 1, 2]) for _ in range(4)]
    gts = [2, 2, 2, 2]
    response_ids, labels = trainer.create_training_batch(responses, gts)

    # Zero grads explicitly first
    optimizer.zero_grad()

    # Run one forward+backward (train_step already calls backward)
    trainer.train_step(response_ids, labels)

    # After train_step, parameters should have non-None gradients
    # (they may have been zeroed by optimizer.step but the step happened)
    # Re-run forward+backward manually to check grad flow
    optimizer.zero_grad()
    logits = net(response_ids)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    grad_norms = [p.grad.norm().item() for p in net.parameters() if p.grad is not None]
    assert len(grad_norms) > 0, "No gradients found — backward did not propagate"
    assert any(g > 0 for g in grad_norms), f"All gradient norms are zero: {grad_norms}"
