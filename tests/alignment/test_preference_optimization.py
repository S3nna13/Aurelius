"""Tests for ORPO, KTO, and RRHF preference optimization."""

from __future__ import annotations

import math

import pytest
import torch
import torch.optim as optim

from src.alignment.preference_optimization import (
    PreferenceOptConfig,
    PreferenceOptTrainer,
    compute_sequence_log_probs,
    kto_loss,
    orpo_loss,
    rrhf_loss,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_cfg():
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
def policy_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def ref_model(small_cfg):
    torch.manual_seed(1)
    m = AureliusTransformer(small_cfg)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _make_ids(batch_size=2, seq_len=16, vocab_size=256, seed=42):
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (batch_size, seq_len))


# ---------------------------------------------------------------------------
# Test 1: PreferenceOptConfig defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = PreferenceOptConfig()
    assert cfg.method == "orpo"
    assert cfg.beta == 0.1
    assert cfg.lambda_ == 1.0
    assert cfg.desirable_weight == 1.0
    assert cfg.undesirable_weight == 1.0


# ---------------------------------------------------------------------------
# Test 2: compute_sequence_log_probs shape is (B,)
# ---------------------------------------------------------------------------


def test_compute_sequence_log_probs_shape(policy_model):
    B, seq_len = 2, 16
    input_ids = _make_ids(batch_size=B, seq_len=seq_len)
    labels = input_ids.clone()
    labels[:, -1] = -100  # mask last token

    lp = compute_sequence_log_probs(policy_model, input_ids, labels)
    assert lp.shape == (B,), f"Expected shape ({B},), got {lp.shape}"
    assert torch.isfinite(lp).all(), "Log-probs must be finite"
    assert (lp <= 0).all(), "Log-probs must be non-positive"


# ---------------------------------------------------------------------------
# Test 3: orpo_loss scalar output
# ---------------------------------------------------------------------------


def test_orpo_loss_scalar_output(policy_model):
    B, seq_len = 2, 16
    chosen_ids = _make_ids(batch_size=B, seq_len=seq_len, seed=10)
    rejected_ids = _make_ids(batch_size=B, seq_len=seq_len, seed=20)

    chosen_labels = chosen_ids.clone()
    chosen_labels[:, -1] = -100
    rejected_labels = rejected_ids.clone()
    rejected_labels[:, -1] = -100

    _, chosen_logits, _ = policy_model(chosen_ids)
    policy_chosen_logps = compute_sequence_log_probs(policy_model, chosen_ids, chosen_labels)
    policy_rejected_logps = compute_sequence_log_probs(policy_model, rejected_ids, rejected_labels)

    loss, metrics = orpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        chosen_logits,
        chosen_labels,
        lambda_=1.0,
        beta=0.1,
    )

    assert loss.ndim == 0, "ORPO loss must be a scalar"
    assert torch.isfinite(loss), "ORPO loss must be finite"


# ---------------------------------------------------------------------------
# Test 4: orpo_loss returns correct dict keys
# ---------------------------------------------------------------------------


def test_orpo_loss_dict_keys(policy_model):
    B, seq_len = 2, 16
    chosen_ids = _make_ids(batch_size=B, seq_len=seq_len, seed=11)
    rejected_ids = _make_ids(batch_size=B, seq_len=seq_len, seed=21)

    chosen_labels = chosen_ids.clone()
    chosen_labels[:, -1] = -100
    rejected_labels = rejected_ids.clone()
    rejected_labels[:, -1] = -100

    _, chosen_logits, _ = policy_model(chosen_ids)
    policy_chosen_logps = compute_sequence_log_probs(policy_model, chosen_ids, chosen_labels)
    policy_rejected_logps = compute_sequence_log_probs(policy_model, rejected_ids, rejected_labels)

    loss, metrics = orpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        chosen_logits,
        chosen_labels,
        lambda_=1.0,
        beta=0.1,
    )

    assert "sft_loss" in metrics, "ORPO metrics must include 'sft_loss'"
    assert "ratio" in metrics, "ORPO metrics must include 'ratio'"
    assert isinstance(metrics["sft_loss"], float)
    assert isinstance(metrics["ratio"], float)


# ---------------------------------------------------------------------------
# Test 5: kto_loss scalar output
# ---------------------------------------------------------------------------


def test_kto_loss_scalar_output():
    B = 4
    policy_chosen = torch.randn(B)
    policy_rejected = torch.randn(B)
    ref_chosen = torch.randn(B)
    ref_rejected = torch.randn(B)

    loss, metrics = kto_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        beta=0.1,
    )

    assert loss.ndim == 0, "KTO loss must be a scalar"
    assert torch.isfinite(loss), "KTO loss must be finite"


# ---------------------------------------------------------------------------
# Test 6: kto_loss returns correct dict keys
# ---------------------------------------------------------------------------


def test_kto_loss_dict_keys():
    B = 4
    policy_chosen = torch.randn(B)
    policy_rejected = torch.randn(B)
    ref_chosen = torch.randn(B)
    ref_rejected = torch.randn(B)

    loss, metrics = kto_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        beta=0.1,
    )

    assert "chosen_utility" in metrics, "KTO metrics must include 'chosen_utility'"
    assert "rejected_utility" in metrics, "KTO metrics must include 'rejected_utility'"
    assert isinstance(metrics["chosen_utility"], float)
    assert isinstance(metrics["rejected_utility"], float)


# ---------------------------------------------------------------------------
# Test 7: rrhf_loss with 3 ranked responses
# ---------------------------------------------------------------------------


def test_rrhf_loss_three_ranked_responses():
    B = 3
    # Three response tiers with clear separation
    best = torch.full((B,), -1.0)
    mid = torch.full((B,), -3.0)
    worst = torch.full((B,), -5.0)

    loss, metrics = rrhf_loss([best, mid, worst])

    assert loss.ndim == 0, "RRHF loss must be a scalar"
    assert torch.isfinite(loss), "RRHF loss must be finite"
    assert loss.item() >= 0.0, "RRHF loss (hinge) must be non-negative"


# ---------------------------------------------------------------------------
# Test 8: rrhf_loss correct ordering → 0 loss
# ---------------------------------------------------------------------------


def test_rrhf_loss_correct_ordering_zero_loss():
    B = 4
    # Perfectly ranked: logp_0 > logp_1 > logp_2 (all by large margin)
    best = torch.full((B,), -1.0)
    mid = torch.full((B,), -5.0)
    worst = torch.full((B,), -10.0)

    loss, _ = rrhf_loss([best, mid, worst])

    assert abs(loss.item()) < 1e-5, (
        f"RRHF loss should be 0 when ordering is satisfied, got {loss.item()}"
    )


# ---------------------------------------------------------------------------
# Test 9: rrhf_loss n_pairs correct count
# ---------------------------------------------------------------------------


def test_rrhf_loss_n_pairs_count():
    B = 2
    # 4 responses → C(4, 2) = 6 pairs
    responses = [torch.randn(B) for _ in range(4)]
    _, metrics = rrhf_loss(responses)

    assert metrics["n_pairs"] == 6, f"Expected 6 pairs for 4 responses, got {metrics['n_pairs']}"

    # 3 responses → C(3, 2) = 3 pairs
    responses3 = [torch.randn(B) for _ in range(3)]
    _, metrics3 = rrhf_loss(responses3)
    assert metrics3["n_pairs"] == 3, f"Expected 3 pairs for 3 responses, got {metrics3['n_pairs']}"


# ---------------------------------------------------------------------------
# Test 10: PreferenceOptTrainer with method="orpo" returns keys
# ---------------------------------------------------------------------------


def test_trainer_orpo_returns_keys(policy_model, ref_model):
    cfg = PreferenceOptConfig(method="orpo", beta=0.1, lambda_=1.0)
    optimizer = optim.SGD(policy_model.parameters(), lr=1e-4)
    trainer = PreferenceOptTrainer(policy_model, ref_model, cfg, optimizer)

    chosen_ids = _make_ids(batch_size=2, seq_len=16, seed=10)
    rejected_ids = _make_ids(batch_size=2, seq_len=16, seed=20)

    result = trainer.train_step(chosen_ids, rejected_ids)

    assert "loss" in result, "Result must contain 'loss'"
    assert "method" in result, "Result must contain 'method'"
    assert result["method"] == "orpo"
    assert isinstance(result["loss"], float)
    assert math.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# Test 11: PreferenceOptTrainer with method="kto" returns keys
# ---------------------------------------------------------------------------


def test_trainer_kto_returns_keys(policy_model, ref_model):
    cfg = PreferenceOptConfig(method="kto", beta=0.1)
    optimizer = optim.SGD(policy_model.parameters(), lr=1e-4)
    trainer = PreferenceOptTrainer(policy_model, ref_model, cfg, optimizer)

    chosen_ids = _make_ids(batch_size=2, seq_len=16, seed=11)
    rejected_ids = _make_ids(batch_size=2, seq_len=16, seed=21)

    result = trainer.train_step(chosen_ids, rejected_ids)

    assert "loss" in result, "Result must contain 'loss'"
    assert "method" in result, "Result must contain 'method'"
    assert result["method"] == "kto"
    assert isinstance(result["loss"], float)
    assert math.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# Test 12: PreferenceOptTrainer with method="rrhf"
# ---------------------------------------------------------------------------


def test_trainer_rrhf_returns_keys(policy_model, ref_model):
    cfg = PreferenceOptConfig(method="rrhf")
    optimizer = optim.SGD(policy_model.parameters(), lr=1e-4)
    trainer = PreferenceOptTrainer(policy_model, ref_model, cfg, optimizer)

    chosen_ids = _make_ids(batch_size=2, seq_len=16, seed=12)
    rejected_ids = _make_ids(batch_size=2, seq_len=16, seed=22)

    result = trainer.train_step(chosen_ids, rejected_ids)

    assert "loss" in result, "Result must contain 'loss'"
    assert "method" in result, "Result must contain 'method'"
    assert result["method"] == "rrhf"
    assert isinstance(result["loss"], float)
    assert math.isfinite(result["loss"])
    assert result["loss"] >= 0.0, "RRHF loss must be non-negative"
