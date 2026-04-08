"""Tests for Online DPO implementation."""
from __future__ import annotations

import copy
import torch
import pytest
from src.alignment.online_dpo import OnlineDPOConfig, OnlineDPOTrainer
from src.alignment.reward_model import RewardModel
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_cfg():
    return AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=64, max_seq_len=32,
    )


@pytest.fixture
def policy(tiny_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(tiny_cfg)


@pytest.fixture
def ref_model(policy):
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


@pytest.fixture
def reward_model(tiny_cfg):
    torch.manual_seed(1)
    backbone = AureliusTransformer(tiny_cfg)
    rm = RewardModel(backbone, freeze_backbone=True)
    rm.training = False
    return rm


@pytest.fixture
def optimizer(policy):
    return torch.optim.AdamW(policy.parameters(), lr=1e-4)


@pytest.fixture
def dpo_cfg():
    return OnlineDPOConfig(
        n_candidates=4,
        max_new_tokens=8,
        temperature=1.0,
        dpo_beta=0.1,
        top_p=1.0,
        reward_batch_size=8,
    )


@pytest.fixture
def trainer(policy, ref_model, reward_model, optimizer, dpo_cfg):
    return OnlineDPOTrainer(policy, ref_model, reward_model, optimizer, dpo_cfg)


@pytest.fixture
def prompt_ids():
    torch.manual_seed(2)
    # Single prompt: shape (seq,) 1-D
    return torch.randint(0, 64, (6,))


@pytest.fixture
def prompt_ids_batch():
    torch.manual_seed(3)
    # Batch of 2 prompts, padded to same length, shape (B, seq)
    return torch.randint(0, 64, (2, 6))


# ---------------------------------------------------------------------------
# Test 1: OnlineDPOConfig defaults
# ---------------------------------------------------------------------------

def test_online_dpo_config_defaults():
    """Verify default field values for OnlineDPOConfig."""
    cfg = OnlineDPOConfig()
    assert cfg.n_candidates == 4
    assert cfg.max_new_tokens == 64
    assert cfg.temperature == 1.0
    assert cfg.dpo_beta == 0.1
    assert cfg.top_p == 1.0
    assert cfg.reward_batch_size == 8


# ---------------------------------------------------------------------------
# Test 2: generate_candidates returns exactly n_candidates completions
# ---------------------------------------------------------------------------

def test_generate_candidates_count(trainer, prompt_ids, dpo_cfg):
    """generate_candidates must return exactly n_candidates completions."""
    candidates = trainer.generate_candidates(prompt_ids)
    assert len(candidates) == dpo_cfg.n_candidates


# ---------------------------------------------------------------------------
# Test 3: Each candidate is a 1-D tensor
# ---------------------------------------------------------------------------

def test_generate_candidates_shape(trainer, prompt_ids):
    """Each candidate returned must be a 1-D tensor."""
    candidates = trainer.generate_candidates(prompt_ids)
    for cand in candidates:
        assert isinstance(cand, torch.Tensor)
        assert cand.ndim == 1, f"Expected 1-D tensor, got shape {cand.shape}"


# ---------------------------------------------------------------------------
# Test 4: select_pair chosen has higher reward than rejected
# ---------------------------------------------------------------------------

def test_select_pair_chosen_higher_reward(trainer, prompt_ids, reward_model):
    """Chosen sequence must have higher reward than rejected sequence."""
    candidates = trainer.generate_candidates(prompt_ids)
    chosen, rejected = trainer.select_pair(prompt_ids, candidates, reward_model)

    # Score chosen and rejected
    with torch.no_grad():
        r_chosen = reward_model(chosen.unsqueeze(0)).item()
        r_rejected = reward_model(rejected.unsqueeze(0)).item()

    assert r_chosen >= r_rejected, (
        f"chosen reward {r_chosen:.4f} should be >= rejected reward {r_rejected:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: select_pair returns full sequences (includes prompt prefix)
# ---------------------------------------------------------------------------

def test_select_pair_returns_full_sequences(trainer, prompt_ids, reward_model):
    """Chosen and rejected tensors must start with the prompt prefix."""
    candidates = trainer.generate_candidates(prompt_ids)
    chosen, rejected = trainer.select_pair(prompt_ids, candidates, reward_model)

    prompt_len = prompt_ids.shape[0]

    # Both should be at least as long as the prompt
    assert chosen.shape[0] >= prompt_len
    assert rejected.shape[0] >= prompt_len

    # Both should start with the prompt tokens
    assert torch.equal(chosen[:prompt_len], prompt_ids), "chosen does not start with prompt"
    assert torch.equal(rejected[:prompt_len], prompt_ids), "rejected does not start with prompt"


# ---------------------------------------------------------------------------
# Test 6: train_step returns correct metric keys
# ---------------------------------------------------------------------------

def test_train_step_returns_metrics(trainer, prompt_ids_batch):
    """train_step must return a dict with the required metric keys."""
    metrics = trainer.train_step(prompt_ids_batch)
    assert isinstance(metrics, dict)
    assert "loss" in metrics
    assert "chosen_reward" in metrics
    assert "rejected_reward" in metrics
    assert "reward_margin" in metrics


# ---------------------------------------------------------------------------
# Test 7: train_step loss is a Python float (not a tensor)
# ---------------------------------------------------------------------------

def test_train_step_loss_is_scalar(trainer, prompt_ids_batch):
    """The 'loss' value in train_step output must be a Python float."""
    metrics = trainer.train_step(prompt_ids_batch)
    assert isinstance(metrics["loss"], float), (
        f"Expected float, got {type(metrics['loss'])}"
    )


# ---------------------------------------------------------------------------
# Test 8: reward_margin is positive when chosen_reward > rejected_reward
# ---------------------------------------------------------------------------

def test_reward_margin_positive(trainer, prompt_ids_batch):
    """reward_margin must equal chosen_reward - rejected_reward and be >= 0."""
    metrics = trainer.train_step(prompt_ids_batch)
    margin = metrics["reward_margin"]
    chosen_r = metrics["chosen_reward"]
    rejected_r = metrics["rejected_reward"]

    assert abs(margin - (chosen_r - rejected_r)) < 1e-4, (
        f"margin {margin} != chosen_reward {chosen_r} - rejected_reward {rejected_r}"
    )
    assert margin >= 0, f"reward_margin {margin} should be >= 0"


# ---------------------------------------------------------------------------
# Test 9: train_step updates model parameters
# ---------------------------------------------------------------------------

def test_train_step_updates_model(tiny_cfg, reward_model):
    """Model parameters must change after a train_step call."""
    torch.manual_seed(10)
    model = AureliusTransformer(tiny_cfg)
    ref = copy.deepcopy(model)
    for p in ref.parameters():
        p.requires_grad_(False)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = OnlineDPOConfig(n_candidates=2, max_new_tokens=4)
    tr = OnlineDPOTrainer(model, ref, reward_model, opt, cfg)

    # Snapshot parameters before
    params_before = {
        name: p.data.clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    prompt_batch = torch.randint(0, 64, (1, 4))
    tr.train_step(prompt_batch)

    # At least one parameter should have changed
    changed = any(
        not torch.equal(p.data, params_before[name])
        for name, p in model.named_parameters()
        if p.requires_grad
    )
    assert changed, "No model parameters changed after train_step"


# ---------------------------------------------------------------------------
# Test 10: integration test with a tiny AureliusTransformer
# ---------------------------------------------------------------------------

def test_online_dpo_with_tiny_model():
    """Integration test: full OnlineDPOTrainer forward pass with tiny model."""
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=64, max_seq_len=32,
    )
    model = AureliusTransformer(cfg)
    ref = copy.deepcopy(model)
    for p in ref.parameters():
        p.requires_grad_(False)

    rm_backbone = AureliusTransformer(cfg)
    rm = RewardModel(rm_backbone, freeze_backbone=True)
    rm.training = False

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dpo_cfg = OnlineDPOConfig(
        n_candidates=3,
        max_new_tokens=6,
        temperature=1.0,
        dpo_beta=0.1,
    )
    trainer = OnlineDPOTrainer(model, ref, rm, opt, dpo_cfg)

    prompt_batch = torch.randint(0, 64, (2, 5))
    metrics = trainer.train_step(prompt_batch)

    # All metric values must be finite floats
    for key in ("loss", "chosen_reward", "rejected_reward", "reward_margin"):
        assert key in metrics
        assert isinstance(metrics[key], float)
        assert torch.isfinite(torch.tensor(metrics[key])), f"{key} is not finite"
