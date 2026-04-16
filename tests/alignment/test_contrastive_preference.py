"""Tests for Contrastive Preference Optimization.

Tiny setup: vocab_size=256, seq_len=16, batch=2.
MockLM is self-contained -- no imports from the main transformer module.
"""

from __future__ import annotations

import copy
import math
import pytest
import torch
import torch.nn as nn

from src.alignment.contrastive_preference import (
    ContrastivePrefConfig,
    ContrastivePrefOptimizer,
    triplet_preference_loss,
    multi_negative_ranking_loss,
)


# ---------------------------------------------------------------------------
# Tiny mock language model
# ---------------------------------------------------------------------------

class MockLM(nn.Module):
    """Minimal LM for testing: input_ids (B, T) -> logits (B, T, vocab_size)."""

    def __init__(self, vocab_size: int = 256, d_model: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)   # (B, T, d_model)
        return self.proj(x)         # (B, T, vocab_size)


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 256
SEQ_LEN = 16
BATCH = 2
N_NEG = 4


def _make_model() -> MockLM:
    torch.manual_seed(0)
    return MockLM(vocab_size=VOCAB_SIZE)


def _make_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=1e-3)


def _make_optimizer_for(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=1e-3)


def _make_cpo(
    temperature: float = 0.07,
    n_negatives: int = N_NEG,
    beta: float = 0.1,
) -> tuple[ContrastivePrefOptimizer, MockLM, MockLM]:
    """Return (optimizer, policy, ref_policy)."""
    torch.manual_seed(0)
    policy = MockLM(vocab_size=VOCAB_SIZE)
    ref_policy = copy.deepcopy(policy)
    opt = ContrastivePrefOptimizer(
        policy=policy,
        ref_policy=ref_policy,
        temperature=temperature,
        n_negatives=n_negatives,
        beta=beta,
    )
    return opt, policy, ref_policy


def _random_ids(batch: int = BATCH, seq_len: int = SEQ_LEN) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (batch, seq_len))


def _labels_from_ids(ids: torch.Tensor) -> torch.Tensor:
    """All positions are valid labels (no ignore index)."""
    return ids.clone()


def _make_rejected_list(n: int = N_NEG, batch: int = BATCH, seq_len: int = SEQ_LEN):
    ids_list = [_random_ids(batch, seq_len) for _ in range(n)]
    labels_list = [_labels_from_ids(x) for x in ids_list]
    return ids_list, labels_list


# ---------------------------------------------------------------------------
# Test 1: ContrastivePrefConfig defaults correct
# ---------------------------------------------------------------------------

def test_config_defaults():
    """ContrastivePrefConfig should have the specified default values."""
    cfg = ContrastivePrefConfig()
    assert cfg.temperature == 0.07, f"Expected temperature=0.07, got {cfg.temperature}"
    assert cfg.n_negatives == 4, f"Expected n_negatives=4, got {cfg.n_negatives}"
    assert cfg.beta == 0.1, f"Expected beta=0.1, got {cfg.beta}"
    assert cfg.hard_negative_ratio == 0.5
    assert cfg.loss_type == "infonce"


# ---------------------------------------------------------------------------
# Test 2: compute_sequence_logps returns (batch,) tensor
# ---------------------------------------------------------------------------

def test_compute_sequence_logps_shape():
    """compute_sequence_logps must return a 1-D (batch,) tensor."""
    opt, policy, _ = _make_cpo()
    ids = _random_ids()
    labels = _labels_from_ids(ids)
    logps = opt.compute_sequence_logps(policy, ids, labels)
    assert logps.ndim == 1, f"Expected 1-D tensor, got shape {logps.shape}"
    assert logps.shape[0] == BATCH, f"Expected batch size {BATCH}, got {logps.shape[0]}"


# ---------------------------------------------------------------------------
# Test 3: info_nce_preference_loss returns scalar
# ---------------------------------------------------------------------------

def test_info_nce_loss_returns_scalar():
    """info_nce_preference_loss must return a scalar (0-dim) tensor."""
    opt, policy, ref_policy = _make_cpo()
    torch.manual_seed(1)

    ids = _random_ids()
    labels = _labels_from_ids(ids)
    chosen_logps = opt.compute_sequence_logps(policy, ids, labels)
    ref_chosen_logps = opt.compute_sequence_logps(ref_policy, ids, labels).detach()

    rej_ids_list, rej_labels_list = _make_rejected_list()
    rejected_logps_list = [
        opt.compute_sequence_logps(policy, r_ids, r_lbl)
        for r_ids, r_lbl in zip(rej_ids_list, rej_labels_list)
    ]
    ref_rejected_logps_list = [
        opt.compute_sequence_logps(ref_policy, r_ids, r_lbl).detach()
        for r_ids, r_lbl in zip(rej_ids_list, rej_labels_list)
    ]

    loss, metrics = opt.info_nce_preference_loss(
        chosen_logps, rejected_logps_list,
        ref_chosen_logps, ref_rejected_logps_list,
    )

    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), "Loss must be finite"


# ---------------------------------------------------------------------------
# Test 4: Accuracy > 0.5 when chosen_logp > all rejected_logps
# ---------------------------------------------------------------------------

def test_accuracy_high_when_chosen_dominates():
    """When chosen implicit reward clearly exceeds all rejected, accuracy should be 1.0."""
    opt, _, _ = _make_cpo(beta=1.0)

    # Chosen reward clearly higher than all rejected
    chosen_logps = torch.full((BATCH,), 0.0)
    ref_chosen_logps = torch.full((BATCH,), -5.0)  # chosen_ratio = +5.0

    rejected_logps_list = [torch.full((BATCH,), -5.0) for _ in range(N_NEG)]
    ref_rejected_logps_list = [torch.full((BATCH,), 0.0) for _ in range(N_NEG)]  # rej_ratio = -5.0

    _, metrics = opt.info_nce_preference_loss(
        chosen_logps, rejected_logps_list,
        ref_chosen_logps, ref_rejected_logps_list,
    )

    assert metrics["accuracy"] > 0.5, (
        f"Expected accuracy > 0.5, got {metrics['accuracy']}"
    )


# ---------------------------------------------------------------------------
# Test 5: Lower temperature -> sharper distribution -> higher loss gradient
# ---------------------------------------------------------------------------

def test_lower_temperature_sharper_distribution():
    """Lower temperature should produce a larger gradient magnitude on chosen_logps."""
    torch.manual_seed(42)
    # Use ambiguous chosen/rejected (small gap) to see temperature effect
    chosen_logps_high_T = torch.tensor([-1.0, -1.2], requires_grad=True)
    chosen_logps_low_T = torch.tensor([-1.0, -1.2], requires_grad=True)

    rejected_logps_list = [torch.full((BATCH,), -1.5) for _ in range(2)]
    ref_chosen = torch.zeros(BATCH)
    ref_rejected = [torch.zeros(BATCH) for _ in range(2)]

    opt_high_T, _, _ = _make_cpo(temperature=1.0)
    opt_low_T, _, _ = _make_cpo(temperature=0.07)

    loss_high, _ = opt_high_T.info_nce_preference_loss(
        chosen_logps_high_T, rejected_logps_list, ref_chosen, ref_rejected
    )
    loss_low, _ = opt_low_T.info_nce_preference_loss(
        chosen_logps_low_T, rejected_logps_list, ref_chosen, ref_rejected
    )

    loss_high.backward()
    loss_low.backward()

    grad_high = chosen_logps_high_T.grad.abs().mean().item()
    grad_low = chosen_logps_low_T.grad.abs().mean().item()

    assert grad_low > grad_high, (
        f"Lower temperature should produce larger gradient: low={grad_low:.4f} vs high={grad_high:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 6: select_hard_negatives returns correct count (n_hard)
# ---------------------------------------------------------------------------

def test_select_hard_negatives_count():
    """select_hard_negatives must return exactly n_hard items."""
    opt, policy, ref_policy = _make_cpo()

    chosen_logps = torch.zeros(BATCH)
    rej_list = [torch.full((BATCH,), float(-i)) for i in range(1, 7)]  # 6 negatives
    ref_rej_list = [torch.zeros(BATCH) for _ in range(6)]

    n_hard = 3
    sel_rej, sel_ref = opt.select_hard_negatives(
        rej_list, ref_rej_list, chosen_logps, n_hard=n_hard
    )

    assert len(sel_rej) == n_hard, f"Expected {n_hard} selected rejected, got {len(sel_rej)}"
    assert len(sel_ref) == n_hard, f"Expected {n_hard} selected ref, got {len(sel_ref)}"


# ---------------------------------------------------------------------------
# Test 7: Hard negatives are closest in reward to chosen
# ---------------------------------------------------------------------------

def test_hard_negatives_are_closest_in_reward():
    """The selected hard negatives should be those with reward closest to chosen."""
    opt, _, _ = _make_cpo(beta=1.0)

    # chosen reward = beta * (chosen_logps - 0) = chosen_logps
    chosen_logps = torch.zeros(BATCH)  # chosen reward ~ 0

    # Rewards for each rejected (ref_rej=0 so rej_reward = rej_logps)
    # Distances from 0: [10, 0.1, 5, 0.2, 8, 0.05]  -> hardest = indices 5,1,3
    rej_values = [-10.0, -0.1, -5.0, -0.2, -8.0, -0.05]
    rej_list = [torch.full((BATCH,), v) for v in rej_values]
    ref_rej_list = [torch.zeros(BATCH) for _ in range(6)]

    n_hard = 3
    sel_rej, _ = opt.select_hard_negatives(rej_list, ref_rej_list, chosen_logps, n_hard=n_hard)

    # The 3 hardest (closest to 0) are indices 5 (-0.05), 1 (-0.1), 3 (-0.2)
    hardest_expected = [-0.05, -0.1, -0.2]
    selected_means = sorted((t.mean().item() for t in sel_rej), reverse=True)
    hardest_expected_sorted = sorted(hardest_expected, reverse=True)
    for got, exp in zip(selected_means, hardest_expected_sorted):
        assert abs(got - exp) < 1e-4, (
            f"Expected hard negative mean ~{exp}, got {got}"
        )


# ---------------------------------------------------------------------------
# Test 8: triplet_preference_loss is 0 when margin satisfied
# ---------------------------------------------------------------------------

def test_triplet_loss_zero_when_margin_satisfied():
    """triplet_preference_loss should be exactly 0 when the margin is clearly satisfied."""
    # anchor - negative >> anchor - positive + margin
    anchor = torch.zeros(BATCH)
    positive = torch.full((BATCH,), -0.1)   # anchor - positive = 0.1
    negative = torch.full((BATCH,), -10.0)  # anchor - negative = 10.0
    margin = 1.0  # need (anchor-negative) - (anchor-positive) >= margin -> 9.9 >= 1.0 (yes)

    loss = triplet_preference_loss(anchor, positive, negative, margin=margin)
    assert loss.item() == 0.0, f"Expected loss=0, got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 9: multi_negative_ranking_loss is scalar
# ---------------------------------------------------------------------------

def test_multi_negative_ranking_loss_scalar():
    """multi_negative_ranking_loss must return a scalar (0-dim) tensor."""
    torch.manual_seed(7)
    chosen_logps = torch.randn(BATCH)
    rejected_stack = torch.randn(BATCH, N_NEG)
    loss = multi_negative_ranking_loss(chosen_logps, rejected_stack)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), "Loss must be finite"


# ---------------------------------------------------------------------------
# Test 10: train_step returns dict with all required keys
# ---------------------------------------------------------------------------

def test_train_step_returns_required_keys():
    """train_step must return a dict containing loss, accuracy, mean_chosen_reward,
    mean_rejected_reward."""
    opt, policy, _ = _make_cpo()
    optimizer = _make_optimizer(policy)

    chosen_ids = _random_ids()
    chosen_labels = _labels_from_ids(chosen_ids)
    rej_ids_list, rej_labels_list = _make_rejected_list()

    optimizer.zero_grad()
    result = opt.train_step(chosen_ids, chosen_labels, rej_ids_list, rej_labels_list)
    optimizer.step()

    required_keys = {"loss", "accuracy", "mean_chosen_reward", "mean_rejected_reward"}
    missing = required_keys - set(result.keys())
    assert not missing, f"Missing keys in train_step result: {missing}"

    for key in required_keys:
        assert math.isfinite(result[key]), f"Key '{key}' must be finite, got {result[key]}"


# ---------------------------------------------------------------------------
# Test 11: Gradient flows through InfoNCE loss
# ---------------------------------------------------------------------------

def test_gradient_flows_through_infonce():
    """Backward through info_nce_preference_loss must produce finite gradients."""
    opt, _, _ = _make_cpo()

    chosen_logps = torch.tensor([-1.0, -1.5], requires_grad=True)
    rejected_logps_list = [
        torch.tensor([-2.0, -2.5]) for _ in range(N_NEG)
    ]
    ref_chosen_logps = torch.zeros(BATCH)
    ref_rejected_logps_list = [torch.zeros(BATCH) for _ in range(N_NEG)]

    loss, _ = opt.info_nce_preference_loss(
        chosen_logps, rejected_logps_list,
        ref_chosen_logps, ref_rejected_logps_list,
    )
    loss.backward()

    assert chosen_logps.grad is not None, "Gradient must flow to chosen_logps"
    assert torch.isfinite(chosen_logps.grad).all(), (
        f"Gradients must be finite, got {chosen_logps.grad}"
    )
