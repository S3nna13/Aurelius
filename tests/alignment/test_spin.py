"""Tests for SPIN: Self-Play Fine-Tuning implementation."""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from src.alignment.spin import (
    SPINLoss,
    SPINTrainer,
    SPINDataCollator,
    compute_token_log_probs,
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
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )


@pytest.fixture
def policy_model(tiny_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(tiny_cfg)


@pytest.fixture
def ref_model(tiny_cfg):
    torch.manual_seed(0)
    model = AureliusTransformer(tiny_cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@pytest.fixture
def spin_trainer(policy_model, ref_model):
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-4)
    return SPINTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        optimizer=optimizer,
        beta=0.1,
        max_gen_tokens=4,
    )


# ---------------------------------------------------------------------------
# SPINLoss tests
# ---------------------------------------------------------------------------

def test_spin_loss_scalar():
    """SPINLoss forward must return a scalar loss."""
    loss_fn = SPINLoss(beta=0.1)
    B = 4
    policy_chosen   = torch.randn(B)
    policy_rejected = torch.randn(B)
    ref_chosen      = torch.randn(B)
    ref_rejected    = torch.randn(B)

    loss, _, _ = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

    assert loss.ndim == 0, f"Expected scalar (0-d tensor), got shape {loss.shape}"
    assert torch.isfinite(loss), "Loss must be finite"


def test_spin_loss_positive():
    """SPINLoss must be >= 0 (it is -log_sigmoid which is always non-negative)."""
    loss_fn = SPINLoss(beta=0.1)
    torch.manual_seed(42)
    B = 8
    policy_chosen   = torch.randn(B)
    policy_rejected = torch.randn(B)
    ref_chosen      = torch.randn(B)
    ref_rejected    = torch.randn(B)

    loss, _, _ = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

    assert loss.item() >= 0.0, f"Loss must be >= 0, got {loss.item()}"


def test_spin_chosen_rewards_shape():
    """chosen_rewards must be a (B,) tensor."""
    loss_fn = SPINLoss(beta=0.1)
    B = 6
    policy_chosen   = torch.randn(B)
    policy_rejected = torch.randn(B)
    ref_chosen      = torch.randn(B)
    ref_rejected    = torch.randn(B)

    _, chosen_rewards, _ = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

    assert chosen_rewards.shape == (B,), (
        f"chosen_rewards shape should be ({B},), got {chosen_rewards.shape}"
    )


def test_spin_reward_margin():
    """When chosen log-prob is clearly better, reward margin should be > 0."""
    loss_fn = SPINLoss(beta=0.1)
    B = 4
    # Policy much prefers the chosen; ref assigns same log-probs to both
    policy_chosen   = torch.full((B,), -1.0)   # higher log-prob for chosen
    policy_rejected = torch.full((B,), -5.0)   # lower log-prob for rejected
    ref_chosen      = torch.full((B,), -3.0)
    ref_rejected    = torch.full((B,), -3.0)

    _, chosen_rewards, rejected_rewards = loss_fn(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )
    margin = (chosen_rewards - rejected_rewards).mean().item()

    assert margin > 0.0, f"Expected positive reward margin, got {margin}"


# ---------------------------------------------------------------------------
# compute_token_log_probs tests
# ---------------------------------------------------------------------------

def test_compute_token_log_probs_shape(policy_model):
    """compute_token_log_probs must return a (B,) tensor."""
    torch.manual_seed(1)
    B, S, R = 2, 8, 4
    input_ids    = torch.randint(0, 256, (B, S))
    response_ids = torch.randint(0, 256, (B, R))

    log_probs = compute_token_log_probs(policy_model, input_ids, response_ids)

    assert log_probs.shape == (B,), (
        f"Expected shape ({B},), got {log_probs.shape}"
    )


def test_compute_token_log_probs_negative(policy_model):
    """Log probs of actual tokens must be <= 0."""
    torch.manual_seed(2)
    B, S, R = 2, 8, 4
    input_ids    = torch.randint(0, 256, (B, S))
    response_ids = torch.randint(0, 256, (B, R))

    log_probs = compute_token_log_probs(policy_model, input_ids, response_ids)

    assert (log_probs <= 0).all(), (
        f"All log probs must be <= 0, got max={log_probs.max().item()}"
    )


# ---------------------------------------------------------------------------
# SPINTrainer tests
# ---------------------------------------------------------------------------

def test_spin_trainer_step_keys(spin_trainer):
    """train_step must return dict with required keys."""
    torch.manual_seed(3)
    prompt_ids        = torch.randint(0, 256, (1, 6))
    real_response_ids = torch.randint(0, 256, (1, 4))

    result = spin_trainer.train_step(prompt_ids, real_response_ids)

    required_keys = {'loss', 'chosen_reward', 'rejected_reward', 'reward_margin'}
    assert required_keys == set(result.keys()), (
        f"Missing or extra keys. Expected {required_keys}, got {set(result.keys())}"
    )
    for k, v in result.items():
        assert isinstance(v, float), f"Key '{k}' should be float, got {type(v)}"


def test_spin_trainer_generates_synthetic(spin_trainer):
    """generate_synthetic must return a tensor of shape (1, max_gen_tokens)."""
    torch.manual_seed(4)
    prompt_ids = torch.randint(0, 256, (1, 6))

    synth = spin_trainer.generate_synthetic(prompt_ids)

    assert isinstance(synth, torch.Tensor), "generate_synthetic must return a Tensor"
    assert synth.shape == (1, spin_trainer.max_gen_tokens), (
        f"Expected shape (1, {spin_trainer.max_gen_tokens}), got {synth.shape}"
    )
    assert synth.dtype == torch.long, f"Expected dtype torch.long, got {synth.dtype}"


def test_spin_update_reference_copies_weights(spin_trainer):
    """After update_reference_model, ref_model must have the same weights as policy_model."""
    # Perturb the policy model so ref and policy differ
    with torch.no_grad():
        for p in spin_trainer.policy_model.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    spin_trainer.update_reference_model()

    for (name, p_param), (_, r_param) in zip(
        spin_trainer.policy_model.named_parameters(),
        spin_trainer.ref_model.named_parameters(),
    ):
        assert torch.allclose(p_param, r_param), (
            f"Parameter '{name}' differs between policy and ref after update_reference_model"
        )

    # Ref model must remain frozen
    for p in spin_trainer.ref_model.parameters():
        assert not p.requires_grad, "ref_model parameters must not require grad after update"


# ---------------------------------------------------------------------------
# SPINDataCollator tests
# ---------------------------------------------------------------------------

def test_spin_data_collator():
    """collate must return a list of dicts with 'prompt_ids' and 'real_response_ids'."""
    collator = SPINDataCollator()

    prompts = [torch.randint(0, 256, (1, 6)) for _ in range(3)]
    responses = [torch.randint(0, 256, (1, 4)) for _ in range(3)]

    batch = collator.collate(prompts, responses)

    assert isinstance(batch, list), "collate must return a list"
    assert len(batch) == 3, f"Expected 3 items, got {len(batch)}"

    for i, item in enumerate(batch):
        assert isinstance(item, dict), f"Item {i} must be a dict"
        assert 'prompt_ids' in item, f"Item {i} missing 'prompt_ids'"
        assert 'real_response_ids' in item, f"Item {i} missing 'real_response_ids'"
        assert torch.equal(item['prompt_ids'], prompts[i]), (
            f"Item {i} prompt_ids mismatch"
        )
        assert torch.equal(item['real_response_ids'], responses[i]), (
            f"Item {i} real_response_ids mismatch"
        )
