"""
Tests for src/training/soft_actor_critic.py

Covers: SACPolicyHead, SACQNetwork, SACReplayBuffer, SACTrainer, SACConfig.
All tests use small dimensions: d_model=16, vocab_size=16, B=4.
"""

import math

import torch
import torch.nn as nn

from src.training.soft_actor_critic import (
    SACConfig,
    SACPolicyHead,
    SACQNetwork,
    SACReplayBuffer,
    SACTrainer,
)

# ---------------------------------------------------------------------------
# Shared fixtures / constants
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_SIZE = 16
B = 4
BUF_CAP = 32


def make_policy() -> SACPolicyHead:
    return SACPolicyHead(d_model=D_MODEL, vocab_size=VOCAB_SIZE, alpha_init=1.0)


def make_q_net() -> SACQNetwork:
    return SACQNetwork(d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_layers=2)


def make_buffer() -> SACReplayBuffer:
    return SACReplayBuffer(capacity=BUF_CAP, state_dim=D_MODEL)


def make_trainer() -> SACTrainer:
    policy = make_policy()
    q_net = make_q_net()
    return SACTrainer(policy=policy, q_net=q_net, lr=3e-4)


def fill_buffer(buf: SACReplayBuffer, n: int) -> None:
    """Push n random transitions into buf."""
    for _ in range(n):
        s = torch.randn(D_MODEL)
        a = torch.randint(0, VOCAB_SIZE, ()).item()
        r = float(torch.randn(1).item())
        ns = torch.randn(D_MODEL)
        d = bool(torch.randint(0, 2, ()).item())
        buf.push(s, a, r, ns, d)


def make_batch(n: int = B) -> tuple:
    """Create a minimal in-memory batch (not from buffer)."""
    states = torch.randn(n, D_MODEL)
    actions = torch.randint(0, VOCAB_SIZE, (n,))
    rewards = torch.randn(n)
    next_states = torch.randn(n, D_MODEL)
    dones = torch.zeros(n, dtype=torch.bool)
    return states, actions, rewards, next_states, dones


# ===========================================================================
# SACPolicyHead tests
# ===========================================================================


def test_policy_forward_logits_shape():
    """forward() must return logits of shape [B, vocab_size]."""
    policy = make_policy()
    hidden = torch.randn(B, D_MODEL)
    logits, _ = policy(hidden)
    assert logits.shape == (B, VOCAB_SIZE), f"Expected {(B, VOCAB_SIZE)}, got {logits.shape}"


def test_policy_forward_log_probs_shape():
    """forward() must return log_probs of shape [B, vocab_size]."""
    policy = make_policy()
    hidden = torch.randn(B, D_MODEL)
    _, log_probs = policy(hidden)
    assert log_probs.shape == (B, VOCAB_SIZE), f"Expected {(B, VOCAB_SIZE)}, got {log_probs.shape}"


def test_policy_log_probs_sum_to_zero():
    """log-softmax rows must sum to ~0 (since exp sums to 1)."""
    policy = make_policy()
    hidden = torch.randn(B, D_MODEL)
    _, log_probs = policy(hidden)
    row_sums = log_probs.exp().sum(dim=-1)  # should be ~1
    assert torch.allclose(row_sums, torch.ones(B), atol=1e-5), (
        f"Softmax rows do not sum to 1: {row_sums}"
    )


def test_policy_sample_action_valid_indices():
    """sample_action must return token indices in [0, vocab_size)."""
    policy = make_policy()
    hidden = torch.randn(B, D_MODEL)
    action, _ = policy.sample_action(hidden)
    assert action.shape == (B,), f"Expected shape ({B},), got {action.shape}"
    assert (action >= 0).all() and (action < VOCAB_SIZE).all(), f"Action out of range: {action}"


def test_policy_sample_action_log_prob_shape():
    """sample_action must return log_prob of shape [B]."""
    policy = make_policy()
    hidden = torch.randn(B, D_MODEL)
    _, log_prob = policy.sample_action(hidden)
    assert log_prob.shape == (B,), f"Expected shape ({B},), got {log_prob.shape}"


def test_policy_alpha_is_positive():
    """alpha property must always be positive (exp of parameter)."""
    policy = make_policy()
    assert policy.alpha.item() > 0, "alpha must be positive"
    # Even after manual perturbation of log_alpha
    with torch.no_grad():
        policy.log_alpha.fill_(-10.0)
    assert policy.alpha.item() > 0


def test_policy_log_alpha_is_parameter():
    """log_alpha must be a nn.Parameter so it can be optimised."""
    policy = make_policy()
    assert isinstance(policy.log_alpha, nn.Parameter), "log_alpha should be nn.Parameter"


# ===========================================================================
# SACQNetwork tests
# ===========================================================================


def test_q_net_forward_q1_shape():
    """q_net forward q1 must have shape [B, vocab_size]."""
    q_net = make_q_net()
    state = torch.randn(B, D_MODEL)
    q1, _ = q_net(state)
    assert q1.shape == (B, VOCAB_SIZE), f"Expected {(B, VOCAB_SIZE)}, got {q1.shape}"


def test_q_net_forward_q2_shape():
    """q_net forward q2 must have shape [B, vocab_size]."""
    q_net = make_q_net()
    state = torch.randn(B, D_MODEL)
    _, q2 = q_net(state)
    assert q2.shape == (B, VOCAB_SIZE), f"Expected {(B, VOCAB_SIZE)}, got {q2.shape}"


def test_q_net_q1_neq_q2():
    """Q1 and Q2 must be independent networks producing different values."""
    q_net = make_q_net()
    state = torch.randn(B, D_MODEL)
    q1, q2 = q_net(state)
    # They should differ at least somewhere (probability 1 with random init)
    assert not torch.allclose(q1, q2), "Q1 and Q2 should not be identical"


# ===========================================================================
# SACReplayBuffer tests
# ===========================================================================


def test_buffer_push_increases_size():
    """Pushing transitions must increase buffer size up to capacity."""
    buf = make_buffer()
    assert len(buf) == 0
    fill_buffer(buf, 5)
    assert len(buf) == 5


def test_buffer_sample_shapes():
    """sample() must return tensors with correct shapes."""
    buf = make_buffer()
    fill_buffer(buf, 16)
    batch_size = 8
    s, a, r, ns, d = buf.sample(batch_size)
    assert s.shape == (batch_size, D_MODEL), f"states shape wrong: {s.shape}"
    assert a.shape == (batch_size,), f"actions shape wrong: {a.shape}"
    assert r.shape == (batch_size,), f"rewards shape wrong: {r.shape}"
    assert ns.shape == (batch_size, D_MODEL), f"next_states shape wrong: {ns.shape}"
    assert d.shape == (batch_size,), f"dones shape wrong: {d.shape}"


def test_buffer_circular_overflow():
    """Overflowing past capacity must keep size at capacity."""
    buf = make_buffer()
    fill_buffer(buf, BUF_CAP + 10)
    assert len(buf) == BUF_CAP, f"Buffer size should be capped at {BUF_CAP}, got {len(buf)}"


def test_buffer_sample_action_dtype():
    """Sampled actions must be long tensors (token indices)."""
    buf = make_buffer()
    fill_buffer(buf, 10)
    _, a, _, _, _ = buf.sample(4)
    assert a.dtype == torch.long, f"Actions dtype should be long, got {a.dtype}"


def test_buffer_sample_dones_dtype():
    """Sampled dones must be bool tensors."""
    buf = make_buffer()
    fill_buffer(buf, 10)
    _, _, _, _, d = buf.sample(4)
    assert d.dtype == torch.bool, f"Dones dtype should be bool, got {d.dtype}"


# ===========================================================================
# SACTrainer tests
# ===========================================================================


def test_trainer_critic_loss_finite_scalar():
    """critic_loss must return a finite scalar tensor."""
    trainer = make_trainer()
    batch = make_batch()
    loss = trainer.critic_loss(batch)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
    assert loss.item() == loss.item(), "critic_loss is NaN"
    assert math.isfinite(loss.item()), f"critic_loss is not finite: {loss.item()}"


def test_trainer_actor_loss_finite_scalar():
    """actor_loss must return a finite scalar tensor."""
    trainer = make_trainer()
    batch = make_batch()
    loss = trainer.actor_loss(batch)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
    assert math.isfinite(loss.item()), f"actor_loss is not finite: {loss.item()}"


def test_trainer_alpha_loss_finite_scalar():
    """alpha_loss must return a finite scalar tensor."""
    trainer = make_trainer()
    batch = make_batch()
    loss = trainer.alpha_loss(batch)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"
    assert math.isfinite(loss.item()), f"alpha_loss is not finite: {loss.item()}"


def test_trainer_update_returns_expected_keys():
    """update() must return a dict with critic_loss, actor_loss, alpha_loss, alpha."""
    trainer = make_trainer()
    batch = make_batch()
    result = trainer.update(batch)
    expected_keys = {"critic_loss", "actor_loss", "alpha_loss", "alpha"}
    assert set(result.keys()) == expected_keys, (
        f"Expected keys {expected_keys}, got {set(result.keys())}"
    )


def test_trainer_update_values_are_finite():
    """All values returned by update() must be finite floats."""
    trainer = make_trainer()
    batch = make_batch()
    result = trainer.update(batch)
    for key, val in result.items():
        assert math.isfinite(val), f"update() key '{key}' is not finite: {val}"


def test_trainer_soft_update_target_moves_toward_main():
    """soft_update_target() must move target parameters closer to main Q-network."""
    trainer = make_trainer()

    # Dramatically diverge target from main by zeroing main params
    with torch.no_grad():
        for p in trainer.q_net.parameters():
            p.fill_(1.0)
        for p in trainer.target_q_net.parameters():
            p.fill_(0.0)

    # Record distance before update
    def l2_dist(net_a, net_b):
        total = 0.0
        for pa, pb in zip(net_a.parameters(), net_b.parameters()):
            total += (pa - pb).pow(2).sum().item()
        return total

    dist_before = l2_dist(trainer.q_net, trainer.target_q_net)
    trainer.soft_update_target()
    dist_after = l2_dist(trainer.q_net, trainer.target_q_net)

    assert dist_after < dist_before, (
        f"Target did not move toward main: dist_before={dist_before}, dist_after={dist_after}"
    )


def test_trainer_target_entropy_default():
    """Default target_entropy should be log(vocab_size) (maximum entropy)."""
    trainer = make_trainer()
    expected = math.log(VOCAB_SIZE)
    assert abs(trainer.target_entropy - expected) < 1e-6, (
        f"Expected target_entropy={expected}, got {trainer.target_entropy}"
    )


# ===========================================================================
# SACConfig tests
# ===========================================================================


def test_sac_config_defaults():
    """SACConfig must have the documented default values."""
    cfg = SACConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.n_layers == 2
    assert abs(cfg.lr - 3e-4) < 1e-9
    assert abs(cfg.gamma - 0.99) < 1e-9
    assert abs(cfg.tau - 0.005) < 1e-9
    assert cfg.buffer_capacity == 1000
    assert abs(cfg.alpha_init - 1.0) < 1e-9


def test_sac_config_override():
    """SACConfig fields can be overridden at construction time."""
    cfg = SACConfig(d_model=64, vocab_size=128, lr=1e-3)
    assert cfg.d_model == 64
    assert cfg.vocab_size == 128
    assert abs(cfg.lr - 1e-3) < 1e-9
