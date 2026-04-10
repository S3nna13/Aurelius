"""Tests for Online DPO implementation (src/alignment/online_dpo.py).

Covers:
  1.  OnlineDPOConfig defaults
  2.  sample_response returns (Tensor, Tensor) tuple
  3.  sample_response response_ids shape
  4.  sample_response log_probs shape
  5.  compute_sequence_log_probs returns (B, T) tensor
  6.  compute_sequence_log_probs values are <= 0 (log probs)
  7.  dpo_loss returns (Tensor, dict) with correct keys
  8.  dpo_loss loss is scalar and finite
  9.  dpo_loss reward_margin = chosen_reward - rejected_reward
  10. dpo_loss accuracy in [0, 1]
  11. dpo_loss with identical chosen/rejected gives reward_margin ~ 0
  12. OnlineDPOTrainer instantiates
  13. OnlineDPOTrainer.generate_preference_pair returns 4-tuple
  14. OnlineDPOTrainer.train_step returns dict with all required keys
  15. OnlineDPOTrainer.train_step loss is finite
"""
from __future__ import annotations

import copy
import torch
import pytest

from src.alignment.online_dpo import (
    OnlineDPOConfig,
    sample_response,
    compute_sequence_log_probs,
    dpo_loss,
    OnlineDPOTrainer,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_tiny_model(seed: int = 0) -> AureliusTransformer:
    torch.manual_seed(seed)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=512,
    )
    return AureliusTransformer(cfg)


def make_ref_model(policy: AureliusTransformer) -> AureliusTransformer:
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def dummy_reward_fn(response_ids: torch.Tensor) -> float:
    """Simple reward: sum of token ids mod 10 as float."""
    return float(response_ids.float().sum().item() % 10)


# ---------------------------------------------------------------------------
# Test 1: OnlineDPOConfig defaults
# ---------------------------------------------------------------------------

def test_online_dpo_config_defaults():
    cfg = OnlineDPOConfig()
    assert cfg.beta == 0.1
    assert cfg.n_samples == 2
    assert cfg.temperature == 1.0
    assert cfg.max_new_tokens == 16
    assert cfg.label_smoothing == 0.0


# ---------------------------------------------------------------------------
# Test 2: sample_response returns (Tensor, Tensor) tuple
# ---------------------------------------------------------------------------

def test_sample_response_returns_tuple():
    model = make_tiny_model(1)
    prompt = torch.randint(0, 256, (1, 4))
    result = sample_response(model, prompt, max_new_tokens=5)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], torch.Tensor)
    assert isinstance(result[1], torch.Tensor)


# ---------------------------------------------------------------------------
# Test 3: sample_response response_ids shape is (B, max_new_tokens)
# ---------------------------------------------------------------------------

def test_sample_response_response_ids_shape():
    model = make_tiny_model(2)
    B, prompt_len, T = 3, 4, 7
    prompt = torch.randint(0, 256, (B, prompt_len))
    response_ids, _ = sample_response(model, prompt, max_new_tokens=T)
    assert response_ids.shape == (B, T), f"Expected ({B}, {T}), got {response_ids.shape}"


# ---------------------------------------------------------------------------
# Test 4: sample_response log_probs shape is (B, max_new_tokens)
# ---------------------------------------------------------------------------

def test_sample_response_log_probs_shape():
    model = make_tiny_model(3)
    B, prompt_len, T = 2, 5, 8
    prompt = torch.randint(0, 256, (B, prompt_len))
    _, log_probs = sample_response(model, prompt, max_new_tokens=T)
    assert log_probs.shape == (B, T), f"Expected ({B}, {T}), got {log_probs.shape}"


# ---------------------------------------------------------------------------
# Test 5: compute_sequence_log_probs returns (B, max_new_tokens) tensor
# ---------------------------------------------------------------------------

def test_compute_sequence_log_probs_shape():
    model = make_tiny_model(4)
    B, prompt_len, T = 2, 6, 5
    prompt_ids = torch.randint(0, 256, (B, prompt_len))
    response_ids = torch.randint(0, 256, (B, T))
    lp = compute_sequence_log_probs(model, prompt_ids, response_ids)
    assert lp.shape == (B, T), f"Expected ({B}, {T}), got {lp.shape}"


# ---------------------------------------------------------------------------
# Test 6: compute_sequence_log_probs values are <= 0 (log probs)
# ---------------------------------------------------------------------------

def test_compute_sequence_log_probs_nonpositive():
    model = make_tiny_model(5)
    B, prompt_len, T = 2, 4, 6
    prompt_ids = torch.randint(0, 256, (B, prompt_len))
    response_ids = torch.randint(0, 256, (B, T))
    lp = compute_sequence_log_probs(model, prompt_ids, response_ids)
    assert (lp <= 0).all(), f"All log probs should be <= 0, got max={lp.max().item():.4f}"


# ---------------------------------------------------------------------------
# Test 7: dpo_loss returns (Tensor, dict) with correct keys
# ---------------------------------------------------------------------------

def test_dpo_loss_returns_tensor_and_dict():
    B = 4
    pi_c  = torch.randn(B)
    pi_r  = torch.randn(B)
    ref_c = torch.randn(B)
    ref_r = torch.randn(B)

    result = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
    assert isinstance(result, tuple) and len(result) == 2

    loss, metrics = result
    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics, dict)

    for key in ("chosen_reward", "rejected_reward", "reward_margin", "accuracy"):
        assert key in metrics, f"Missing key '{key}' in metrics"


# ---------------------------------------------------------------------------
# Test 8: dpo_loss loss is scalar and finite
# ---------------------------------------------------------------------------

def test_dpo_loss_scalar_and_finite():
    B = 4
    pi_c  = torch.randn(B)
    pi_r  = torch.randn(B)
    ref_c = torch.randn(B)
    ref_r = torch.randn(B)

    loss, _ = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
    assert loss.ndim == 0, "loss should be a scalar"
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# Test 9: dpo_loss reward_margin = chosen_reward - rejected_reward
# ---------------------------------------------------------------------------

def test_dpo_loss_reward_margin_consistency():
    B = 6
    pi_c  = torch.randn(B)
    pi_r  = torch.randn(B)
    ref_c = torch.randn(B)
    ref_r = torch.randn(B)

    _, metrics = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
    expected_margin = metrics["chosen_reward"] - metrics["rejected_reward"]
    assert abs(metrics["reward_margin"] - expected_margin) < 1e-5, (
        f"reward_margin {metrics['reward_margin']} != chosen-rejected {expected_margin}"
    )


# ---------------------------------------------------------------------------
# Test 10: dpo_loss accuracy in [0, 1]
# ---------------------------------------------------------------------------

def test_dpo_loss_accuracy_in_range():
    B = 8
    pi_c  = torch.randn(B)
    pi_r  = torch.randn(B)
    ref_c = torch.randn(B)
    ref_r = torch.randn(B)

    _, metrics = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
    acc = metrics["accuracy"]
    assert 0.0 <= acc <= 1.0, f"accuracy {acc} is out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 11: dpo_loss with identical chosen/rejected gives reward_margin ~ 0
# ---------------------------------------------------------------------------

def test_dpo_loss_identical_responses_zero_margin():
    B = 4
    same = torch.randn(B)

    _, metrics = dpo_loss(same, same, same, same, beta=0.1)
    assert abs(metrics["reward_margin"]) < 1e-5, (
        f"reward_margin should be ~0 for identical inputs, got {metrics['reward_margin']}"
    )


# ---------------------------------------------------------------------------
# Test 12: OnlineDPOTrainer instantiates
# ---------------------------------------------------------------------------

def test_online_dpo_trainer_instantiates():
    policy = make_tiny_model(10)
    ref = make_ref_model(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    cfg = OnlineDPOConfig(n_samples=2, max_new_tokens=4)

    trainer = OnlineDPOTrainer(policy, ref, dummy_reward_fn, cfg, optimizer)
    assert trainer is not None
    assert trainer.policy is policy
    assert trainer.ref_model is ref
    assert trainer.reward_fn is dummy_reward_fn


# ---------------------------------------------------------------------------
# Test 13: OnlineDPOTrainer.generate_preference_pair returns 4-tuple
# ---------------------------------------------------------------------------

def test_generate_preference_pair_returns_4_tuple():
    policy = make_tiny_model(11)
    ref = make_ref_model(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    cfg = OnlineDPOConfig(n_samples=3, max_new_tokens=5)

    trainer = OnlineDPOTrainer(policy, ref, dummy_reward_fn, cfg, optimizer)
    prompt_ids = torch.randint(0, 256, (1, 6))

    result = trainer.generate_preference_pair(prompt_ids)
    assert isinstance(result, tuple) and len(result) == 4

    chosen_ids, rejected_ids, chosen_reward, rejected_reward = result
    assert isinstance(chosen_ids, torch.Tensor)
    assert isinstance(rejected_ids, torch.Tensor)
    assert isinstance(chosen_reward, float)
    assert isinstance(rejected_reward, float)
    assert chosen_ids.shape == (cfg.max_new_tokens,)
    assert rejected_ids.shape == (cfg.max_new_tokens,)


# ---------------------------------------------------------------------------
# Test 14: OnlineDPOTrainer.train_step returns dict with all required keys
# ---------------------------------------------------------------------------

def test_train_step_returns_required_keys():
    policy = make_tiny_model(12)
    ref = make_ref_model(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    cfg = OnlineDPOConfig(n_samples=2, max_new_tokens=4)

    trainer = OnlineDPOTrainer(policy, ref, dummy_reward_fn, cfg, optimizer)
    prompt_ids = torch.randint(0, 256, (1, 5))

    metrics = trainer.train_step(prompt_ids)
    assert isinstance(metrics, dict)
    required_keys = {"loss", "chosen_reward", "rejected_reward", "reward_margin", "accuracy"}
    for key in required_keys:
        assert key in metrics, f"Missing key '{key}' in train_step output"


# ---------------------------------------------------------------------------
# Test 15: OnlineDPOTrainer.train_step loss is finite
# ---------------------------------------------------------------------------

def test_train_step_loss_is_finite():
    policy = make_tiny_model(13)
    ref = make_ref_model(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    cfg = OnlineDPOConfig(n_samples=2, max_new_tokens=4)

    trainer = OnlineDPOTrainer(policy, ref, dummy_reward_fn, cfg, optimizer)
    prompt_ids = torch.randint(0, 256, (1, 5))

    metrics = trainer.train_step(prompt_ids)
    loss_val = metrics["loss"]
    assert isinstance(loss_val, float), f"Expected float, got {type(loss_val)}"
    assert torch.isfinite(torch.tensor(loss_val)), f"loss is not finite: {loss_val}"
