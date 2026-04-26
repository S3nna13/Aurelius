"""Tests for rlvr_trainer.py — RLVRConfig, VerifiableProblem, verifiers,
generate_rollouts, rlvr_loss, and RLVRTrainer."""

import pytest
import torch

from src.alignment.rlvr_trainer import (
    RLVRConfig,
    RLVRTrainer,
    VerifiableProblem,
    format_verifier,
    generate_rollouts,
    math_verifier,
    rlvr_loss,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Tiny model config (matches architecture context)
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

RLVR_CFG = RLVRConfig(
    n_rollouts=2,
    max_new_tokens=4,
    max_seq_len=64,
    n_ppo_steps=1,
)


def DECODE(ids):
    return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


@pytest.fixture
def tiny_model():
    torch.manual_seed(42)
    return AureliusTransformer(TINY_CFG)


@pytest.fixture
def tiny_ref_model():
    torch.manual_seed(7)
    return AureliusTransformer(TINY_CFG)


@pytest.fixture
def prompt_ids():
    return torch.tensor([10, 20, 30], dtype=torch.long)


@pytest.fixture
def verifiable_problem(prompt_ids):
    return VerifiableProblem(
        prompt_ids=prompt_ids,
        ground_truth="42",
        problem_type="math",
    )


# ---------------------------------------------------------------------------
# 1. math_verifier: correct answer -> 1.0
# ---------------------------------------------------------------------------


def test_math_verifier_correct():
    assert math_verifier("The answer is 42", "42") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. math_verifier: wrong answer -> 0.0
# ---------------------------------------------------------------------------


def test_math_verifier_wrong():
    assert math_verifier("The answer is 99", "42") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. math_verifier: handles float answers
# ---------------------------------------------------------------------------


def test_math_verifier_float():
    assert math_verifier("Result: 3.14", "3.14") == pytest.approx(1.0)
    assert math_verifier("Result: 3.14", "2.71") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 4. format_verifier: matching substring -> 1.0
# ---------------------------------------------------------------------------


def test_format_verifier_match():
    assert format_verifier("Hello world", "world") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. format_verifier: non-matching -> 0.0
# ---------------------------------------------------------------------------


def test_format_verifier_no_match():
    assert format_verifier("Hello world", "xyz_zzz_abc_123") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. VerifiableProblem constructs correctly
# ---------------------------------------------------------------------------


def test_verifiable_problem_construction(prompt_ids):
    vp = VerifiableProblem(
        prompt_ids=prompt_ids,
        ground_truth="7",
        problem_type="math",
    )
    assert torch.equal(vp.prompt_ids, prompt_ids)
    assert vp.ground_truth == "7"
    assert vp.problem_type == "math"


# ---------------------------------------------------------------------------
# 7. generate_rollouts returns list of dicts
# ---------------------------------------------------------------------------


def test_generate_rollouts_returns_list(tiny_model, verifiable_problem):
    torch.manual_seed(0)
    result = generate_rollouts(
        tiny_model,
        [verifiable_problem],
        RLVR_CFG,
        DECODE,
        math_verifier,
    )
    assert isinstance(result, list)
    assert len(result) == RLVR_CFG.n_rollouts


# ---------------------------------------------------------------------------
# 8. Each rollout dict has prompt_ids, response_ids, reward
# ---------------------------------------------------------------------------


def test_generate_rollouts_dict_keys(tiny_model, verifiable_problem):
    torch.manual_seed(1)
    result = generate_rollouts(
        tiny_model,
        [verifiable_problem],
        RLVR_CFG,
        DECODE,
        math_verifier,
    )
    for rollout in result:
        assert "prompt_ids" in rollout
        assert "response_ids" in rollout
        assert "reward" in rollout
        assert "decoded_response" in rollout
        assert isinstance(rollout["response_ids"], list)
        assert isinstance(rollout["reward"], float)


# ---------------------------------------------------------------------------
# 9. rlvr_loss returns (Tensor, dict)
# ---------------------------------------------------------------------------


def test_rlvr_loss_return_types(tiny_model, tiny_ref_model, verifiable_problem):
    torch.manual_seed(2)
    rollouts = generate_rollouts(
        tiny_model,
        [verifiable_problem],
        RLVR_CFG,
        DECODE,
        math_verifier,
    )
    loss, metrics = rlvr_loss(tiny_model, tiny_ref_model, rollouts, RLVR_CFG)
    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# 10. dict has keys: policy_loss, kl, mean_reward, reward_std
# ---------------------------------------------------------------------------


def test_rlvr_loss_metric_keys(tiny_model, tiny_ref_model, verifiable_problem):
    torch.manual_seed(3)
    rollouts = generate_rollouts(
        tiny_model,
        [verifiable_problem],
        RLVR_CFG,
        DECODE,
        math_verifier,
    )
    _, metrics = rlvr_loss(tiny_model, tiny_ref_model, rollouts, RLVR_CFG)
    for key in ("policy_loss", "kl", "mean_reward", "reward_std"):
        assert key in metrics, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 11. RLVRTrainer constructs without error
# ---------------------------------------------------------------------------


def test_rlvr_trainer_constructs(tiny_model, tiny_ref_model):
    trainer = RLVRTrainer(
        model=tiny_model,
        ref_model=tiny_ref_model,
        config=RLVR_CFG,
        verifier=math_verifier,
        tokenizer_decode=DECODE,
    )
    assert trainer.model is tiny_model
    assert trainer.ref_model is tiny_ref_model
    assert trainer.config is RLVR_CFG


# ---------------------------------------------------------------------------
# 12. RLVRTrainer.step returns metrics dict
# ---------------------------------------------------------------------------


def test_rlvr_trainer_step_returns_metrics(tiny_model, tiny_ref_model, verifiable_problem):
    torch.manual_seed(4)
    trainer = RLVRTrainer(
        model=tiny_model,
        ref_model=tiny_ref_model,
        config=RLVR_CFG,
        verifier=math_verifier,
        tokenizer_decode=DECODE,
    )
    metrics = trainer.step([verifiable_problem])
    assert isinstance(metrics, dict)
    # Should have the standard keys
    for key in ("policy_loss", "kl", "mean_reward", "reward_std", "loss"):
        assert key in metrics, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 13. rlvr_loss loss is finite scalar
# ---------------------------------------------------------------------------


def test_rlvr_loss_finite_scalar(tiny_model, tiny_ref_model, verifiable_problem):
    torch.manual_seed(5)
    rollouts = generate_rollouts(
        tiny_model,
        [verifiable_problem],
        RLVR_CFG,
        DECODE,
        math_verifier,
    )
    loss, _ = rlvr_loss(tiny_model, tiny_ref_model, rollouts, RLVR_CFG)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# 14. generate_rollouts: response length matches max_new_tokens
# ---------------------------------------------------------------------------


def test_generate_rollouts_response_length(tiny_model, verifiable_problem):
    torch.manual_seed(6)
    result = generate_rollouts(
        tiny_model,
        [verifiable_problem],
        RLVR_CFG,
        DECODE,
        math_verifier,
    )
    for rollout in result:
        assert len(rollout["response_ids"]) == RLVR_CFG.max_new_tokens


# ---------------------------------------------------------------------------
# 15. RLVRConfig defaults are correct
# ---------------------------------------------------------------------------


def test_rlvr_config_defaults():
    cfg = RLVRConfig()
    assert cfg.lr == pytest.approx(1e-5)
    assert cfg.kl_coef == pytest.approx(0.04)
    assert cfg.n_rollouts == 4
    assert cfg.max_new_tokens == 16
    assert cfg.temperature == pytest.approx(1.0)
    assert cfg.clip_eps == pytest.approx(0.2)
    assert cfg.n_ppo_steps == 2
    assert cfg.format_reward == pytest.approx(0.1)
    assert cfg.correctness_reward == pytest.approx(1.0)
    assert cfg.max_seq_len == 64
