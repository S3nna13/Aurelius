"""Tests for online PPO RLHF training loop."""

from __future__ import annotations

import math

import torch

from src.alignment.reward_model import RewardModel
from src.alignment.value_head import PPOConfig, ValueHead
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.rlhf import RLHFBatch, RLHFConfig, RLHFTrainer

# ---------------------------------------------------------------------------
# Tiny config helpers
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=32,
)

RLHF_CFG = RLHFConfig(
    n_rollouts_per_prompt=2,
    max_new_tokens=4,
    ppo_epochs=1,
    mini_batch_size=4,
    lr=1e-4,
)


def make_policy():
    backbone = AureliusTransformer(TINY_CFG)
    ppo_cfg = PPOConfig(d_model=TINY_CFG.d_model)
    return ValueHead(backbone, ppo_cfg)


def make_reward_model():
    from src.alignment.reward_model import RewardModelConfig

    backbone = AureliusTransformer(TINY_CFG)
    # backbone returns (loss, logits, kv) tuple; logits dim == vocab_size
    cfg = RewardModelConfig(d_model=TINY_CFG.vocab_size)
    return RewardModel(backbone, cfg)


def make_prompts(n: int = 2, prompt_len: int = 4) -> list[torch.Tensor]:
    return [torch.randint(0, TINY_CFG.vocab_size, (prompt_len,)) for _ in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rlhf_config_defaults():
    cfg = RLHFConfig()
    assert cfg.n_rollouts_per_prompt == 4
    assert cfg.max_new_tokens == 64
    assert cfg.temperature == 1.0
    assert cfg.top_p == 0.9
    assert cfg.ppo_epochs == 4
    assert cfg.mini_batch_size == 8
    assert cfg.lr == 1e-5
    assert cfg.kl_coef == 0.1
    assert cfg.gamma == 1.0
    assert cfg.lam == 0.95
    assert cfg.clip_eps == 0.2
    assert cfg.value_loss_coef == 0.5
    assert cfg.entropy_coef == 0.01
    assert cfg.max_grad_norm == 1.0


def test_rlhf_batch_dataclass():
    n, s = 4, 6
    batch = RLHFBatch(
        prompt_ids=[torch.zeros(3, dtype=torch.long)] * n,
        response_ids=[torch.zeros(s, dtype=torch.long)] * n,
        rewards=torch.zeros(n),
        old_log_probs=torch.zeros(n, s),
        values=torch.zeros(n, s),
        advantages=torch.zeros(n, s),
        returns=torch.zeros(n, s),
    )
    assert batch.rewards.shape == (n,)
    assert batch.advantages.shape == (n, s)
    assert len(batch.prompt_ids) == n
    assert len(batch.response_ids) == n


def test_collect_rollouts_shapes():
    policy = make_policy()
    rm = make_reward_model()
    trainer = RLHFTrainer(policy, rm, RLHF_CFG)
    prompts = make_prompts(n=2, prompt_len=4)

    batch = trainer.collect_rollouts(prompts)

    # N = n_prompts * n_rollouts_per_prompt
    N = len(prompts) * RLHF_CFG.n_rollouts_per_prompt
    assert batch.rewards.shape == (N,), f"rewards shape {batch.rewards.shape} != ({N},)"
    # advantages should be (N, S_resp)
    assert batch.advantages.ndim == 2
    assert batch.advantages.shape[0] == N


def test_collect_rollouts_rewards_finite():
    policy = make_policy()
    rm = make_reward_model()
    trainer = RLHFTrainer(policy, rm, RLHF_CFG)
    prompts = make_prompts(n=2, prompt_len=4)

    batch = trainer.collect_rollouts(prompts)

    assert torch.isfinite(batch.rewards).all(), "Some rewards are not finite"
    assert torch.isfinite(batch.advantages).all(), "Some advantages are not finite"
    assert torch.isfinite(batch.returns).all(), "Some returns are not finite"


def test_train_step_returns_metrics():
    policy = make_policy()
    rm = make_reward_model()
    trainer = RLHFTrainer(policy, rm, RLHF_CFG)
    prompts = make_prompts(n=2, prompt_len=4)

    batch = trainer.collect_rollouts(prompts)
    metrics = trainer.train_step(batch)

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert "kl" in metrics


def test_train_step_metrics_finite():
    policy = make_policy()
    rm = make_reward_model()
    trainer = RLHFTrainer(policy, rm, RLHF_CFG)
    prompts = make_prompts(n=2, prompt_len=4)

    batch = trainer.collect_rollouts(prompts)
    metrics = trainer.train_step(batch)

    for key, val in metrics.items():
        assert isinstance(val, float), f"metric {key} is not a float: {type(val)}"
        assert math.isfinite(val), f"metric {key} is not finite: {val}"


def test_step_updates_weights():
    policy = make_policy()
    rm = make_reward_model()
    trainer = RLHFTrainer(policy, rm, RLHF_CFG)
    prompts = make_prompts(n=2, prompt_len=4)

    # Snapshot weights before
    before = {name: p.clone().detach() for name, p in policy.named_parameters()}

    trainer.step(prompts)

    # At least one parameter should have changed
    changed = any(
        not torch.equal(before[name], p.detach()) for name, p in policy.named_parameters()
    )
    assert changed, "No weights changed after step() — optimizer may not be working"
