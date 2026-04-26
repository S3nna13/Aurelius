"""
Tests for token_level_rl.py
============================
16+ tests covering all public classes.

Tiny policy: d_model=16, vocab_size=16, B=2, T=6.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.token_level_rl import (
    TokenAdvantageEstimator,
    TokenCreditModel,
    TokenLevelValueHead,
    TokenPPO,
    TokenRewardAssigner,
    TokenRLConfig,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_SIZE = 16
B = 2
T = 6


class TinyPolicy(nn.Module):
    """Minimal policy for testing: embedding → linear head."""

    def __init__(self, d_model: int = D_MODEL, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embedding(input_ids)  # [B, T, d_model]
        logits = self.head(hidden)  # [B, T, V]
        return {"logits": logits, "hidden_states": hidden}


def make_policy_and_value():
    policy = TinyPolicy(D_MODEL, VOCAB_SIZE)
    value_head = TokenLevelValueHead(D_MODEL)
    return policy, value_head


def make_ppo():
    policy, value_head = make_policy_and_value()
    return TokenPPO(
        policy=policy,
        value_head=value_head,
        lr=1e-3,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
    )


def sample_rewards(b: int = B) -> torch.Tensor:
    return torch.rand(b)


def sample_token_rewards(b: int = B, t: int = T) -> torch.Tensor:
    return torch.rand(b, t)


def sample_input_ids(b: int = B, t: int = T, v: int = VOCAB_SIZE) -> torch.Tensor:
    return torch.randint(0, v, (b, t))


# ===========================================================================
# TokenRewardAssigner tests
# ===========================================================================


def test_reward_assigner_dense_sum():
    """Dense assignment: total reward == B * T * original reward for uniform reward."""
    assigner = TokenRewardAssigner(method="dense")
    reward = torch.ones(B)  # uniform reward of 1.0
    token_r = assigner.assign(reward, T)
    assert token_r.shape == (B, T)
    expected_sum = B * T * 1.0
    assert abs(token_r.sum().item() - expected_sum) < 1e-5


def test_reward_assigner_dense_shape():
    """Dense assignment output shape is [B, T]."""
    assigner = TokenRewardAssigner(method="dense")
    token_r = assigner.assign(sample_rewards(), T)
    assert token_r.shape == (B, T)


def test_reward_assigner_sparse_only_last_nonzero():
    """Sparse: only last token position is nonzero."""
    assigner = TokenRewardAssigner(method="sparse")
    reward = torch.ones(B)
    token_r = assigner.assign(reward, T)
    assert token_r.shape == (B, T)
    # All positions except last should be zero
    assert token_r[:, :-1].abs().sum().item() == 0.0
    # Last position should equal the reward
    assert (token_r[:, -1] == reward).all()


def test_reward_assigner_gamma_decay_last_highest():
    """gamma_decay: last token should have the highest weight (exponent = 0)."""
    assigner = TokenRewardAssigner(method="gamma_decay", gamma=0.99)
    reward = torch.ones(B)
    token_r = assigner.assign(reward, T)
    assert token_r.shape == (B, T)
    # Last token gets gamma^0 = 1.0 * R; earlier tokens get smaller values
    last = token_r[:, -1]
    for t in range(T - 1):
        assert (token_r[:, t] <= last + 1e-6).all(), (
            f"Token {t} weight {token_r[:, t]} exceeds last token {last}"
        )


def test_reward_assigner_credit_propagation_shape():
    """credit_propagation: output shape is [B, T]."""
    assigner = TokenRewardAssigner(method="credit_propagation")
    token_r = assigner.assign(sample_rewards(), T)
    assert token_r.shape == (B, T)


def test_reward_assigner_credit_propagation_nonneg():
    """credit_propagation with positive reward produces nonneg values."""
    assigner = TokenRewardAssigner(method="credit_propagation")
    reward = torch.ones(B)
    token_r = assigner.assign(reward, T)
    assert (token_r >= 0).all()


# ===========================================================================
# TokenLevelValueHead tests
# ===========================================================================


def test_value_head_output_shape():
    """Value head forward: output shape is [B, T]."""
    vh = TokenLevelValueHead(D_MODEL)
    hidden = torch.randn(B, T, D_MODEL)
    values = vh(hidden)
    assert values.shape == (B, T)


def test_value_head_is_module():
    """Value head is an nn.Module."""
    vh = TokenLevelValueHead(D_MODEL)
    assert isinstance(vh, nn.Module)


# ===========================================================================
# TokenAdvantageEstimator tests
# ===========================================================================


def test_advantage_estimator_gae_shape():
    """GAE output shape is [B, T]."""
    est = TokenAdvantageEstimator(gamma=0.99, lam=0.95)
    rewards = sample_token_rewards()
    values = sample_token_rewards()
    dones = torch.zeros(B, T)
    dones[:, -1] = 1.0
    adv = est.gae(rewards, values, dones)
    assert adv.shape == (B, T)


def test_advantage_estimator_returns_shape():
    """Discounted returns output shape is [B, T]."""
    est = TokenAdvantageEstimator(gamma=0.99)
    rewards = sample_token_rewards()
    ret = est.returns(rewards)
    assert ret.shape == (B, T)


def test_advantage_estimator_returns_last_step():
    """Discounted returns: G_{T-1} == r_{T-1} (no future tokens)."""
    est = TokenAdvantageEstimator(gamma=0.99)
    rewards = sample_token_rewards()
    ret = est.returns(rewards)
    # At the last time step G_{T-1} = r_{T-1}
    assert torch.allclose(ret[:, -1], rewards[:, -1], atol=1e-5)


def test_advantage_estimator_returns_gamma_override():
    """returns() respects an explicit gamma argument."""
    est = TokenAdvantageEstimator(gamma=0.99)
    rewards = torch.ones(B, T)
    ret_09 = est.returns(rewards, gamma=0.9)
    ret_099 = est.returns(rewards, gamma=0.99)
    # With gamma=0.9 the sum converges faster — first step should be smaller
    assert ret_09[0, 0].item() < ret_099[0, 0].item()


# ===========================================================================
# TokenPPO tests
# ===========================================================================


def test_ppo_policy_loss_finite_scalar():
    """policy_loss returns a finite scalar tensor."""
    ppo = make_ppo()
    log_probs_new = torch.randn(B, T)
    log_probs_old = torch.randn(B, T)
    advantages = torch.randn(B, T)
    loss = ppo.policy_loss(log_probs_new, log_probs_old, advantages)
    assert loss.shape == ()
    assert math.isfinite(loss.item())


def test_ppo_value_loss_finite_scalar():
    """value_loss returns a finite scalar tensor."""
    ppo = make_ppo()
    values = torch.randn(B, T)
    returns = torch.randn(B, T)
    loss = ppo.value_loss(values, returns)
    assert loss.shape == ()
    assert math.isfinite(loss.item())


def test_ppo_entropy_bonus_nonneg():
    """entropy_bonus >= 0 for valid log-prob distributions."""
    ppo = make_ppo()
    logits = torch.randn(B, T, VOCAB_SIZE)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = ppo.entropy_bonus(log_probs)
    assert entropy.item() >= -1e-6  # allow tiny numerical noise


def test_ppo_total_loss_returns_four_tuple():
    """total_loss returns a 4-tuple of scalars."""
    ppo = make_ppo()
    log_probs_new = torch.randn(B, T)
    log_probs_old = torch.randn(B, T)
    values = torch.randn(B, T)
    returns = torch.randn(B, T)
    advantages = torch.randn(B, T)
    logits = torch.randn(B, T, VOCAB_SIZE)
    log_probs_dist = F.log_softmax(logits, dim=-1)
    result = ppo.total_loss(
        log_probs_new, log_probs_old, values, returns, advantages, log_probs_dist
    )
    assert len(result) == 4
    for item in result:
        assert math.isfinite(item.item())


def test_ppo_update_step_returns_dict():
    """update_step returns a dict containing the required loss keys."""
    ppo = make_ppo()
    input_ids = sample_input_ids()
    rewards = sample_rewards()
    result = ppo.update_step(input_ids, rewards)
    assert isinstance(result, dict)
    for key in ("loss", "policy_loss", "value_loss", "entropy_loss"):
        assert key in result, f"Missing key: {key}"
        assert math.isfinite(result[key]), f"{key} is not finite: {result[key]}"


def test_ppo_update_step_loss_is_finite():
    """update_step total loss is finite after a gradient step."""
    ppo = make_ppo()
    input_ids = sample_input_ids()
    rewards = sample_rewards()
    result = ppo.update_step(input_ids, rewards)
    assert math.isfinite(result["loss"])


# ===========================================================================
# TokenCreditModel tests
# ===========================================================================


def test_credit_model_output_shape():
    """TokenCreditModel forward: output shape is [B, T]."""
    model = TokenCreditModel(d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_layers=2)
    input_ids = sample_input_ids()
    reward = sample_rewards()
    credits = model(input_ids, reward)
    assert credits.shape == (B, T)


def test_credit_model_sums_to_one():
    """TokenCreditModel output is a probability distribution over T (softmax)."""
    model = TokenCreditModel(d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_layers=2)
    input_ids = sample_input_ids()
    reward = sample_rewards()
    credits = model(input_ids, reward)
    sums = credits.sum(dim=-1)  # [B]
    assert torch.allclose(sums, torch.ones(B), atol=1e-5), f"Credits do not sum to 1: {sums}"


def test_credit_model_nonneg():
    """TokenCreditModel outputs are non-negative (softmax outputs)."""
    model = TokenCreditModel(d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_layers=2)
    credits = model(sample_input_ids(), sample_rewards())
    assert (credits >= 0).all()


def test_credit_model_is_module():
    """TokenCreditModel is an nn.Module."""
    model = TokenCreditModel(d_model=D_MODEL, vocab_size=VOCAB_SIZE)
    assert isinstance(model, nn.Module)


# ===========================================================================
# TokenRLConfig tests
# ===========================================================================


def test_token_rl_config_defaults():
    """TokenRLConfig has correct default values."""
    cfg = TokenRLConfig()
    assert cfg.gamma == 0.99
    assert cfg.lam == 0.95
    assert cfg.clip_eps == 0.2
    assert cfg.vf_coef == 0.5
    assert cfg.ent_coef == 0.01
    assert cfg.lr == 1e-4
    assert cfg.method == "gamma_decay"
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64


def test_token_rl_config_override():
    """TokenRLConfig values can be overridden at construction."""
    cfg = TokenRLConfig(gamma=0.95, lr=5e-5, method="dense")
    assert cfg.gamma == 0.95
    assert cfg.lr == 5e-5
    assert cfg.method == "dense"
