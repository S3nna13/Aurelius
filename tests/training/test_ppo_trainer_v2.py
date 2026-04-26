"""Tests for PPO trainer v2 (src/training/ppo_trainer_v2.py).

15 tests covering ValueHead, GAEComputation, PPOLoss, PPOBuffer, and PPOTrainer.
Tiny config: d_model=16, vocab=16, seq_len=8, batch=2, max_new_tokens=4, ppo_epochs=2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from aurelius.training.ppo_trainer_v2 import (
    GAEComputation,
    PPOBuffer,
    PPOLoss,
    PPOTrainer,
    ValueHead,
)

# ---------------------------------------------------------------------------
# Tiny constants
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB = 16
SEQ_LEN = 8
BATCH = 2
MAX_NEW_TOKENS = 4
PPO_EPOCHS = 2


# ---------------------------------------------------------------------------
# Minimal policy / ref model fixture
# Uses nn.Embedding + nn.Linear to produce (logits, hidden_states).
# ---------------------------------------------------------------------------


class TinyPolicy(nn.Module):
    """Embedding -> linear. Returns (logits, embeddings) as hidden states."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.proj = nn.Linear(D_MODEL, VOCAB)

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embed(input_ids)  # (B, T, D)
        logits = self.proj(hidden)  # (B, T, V)
        return logits, hidden


def make_policy() -> TinyPolicy:
    torch.manual_seed(42)
    return TinyPolicy()


def make_ref() -> TinyPolicy:
    torch.manual_seed(99)
    return TinyPolicy()


def make_value_head() -> ValueHead:
    torch.manual_seed(7)
    return ValueHead(D_MODEL)


def dummy_reward_fn(sequences: torch.Tensor) -> torch.Tensor:
    """Returns a fixed reward (1.0) per sequence."""
    return torch.ones(sequences.shape[0])


# ---------------------------------------------------------------------------
# Test 1: ValueHead output shape
# ---------------------------------------------------------------------------


def test_value_head_output_shape():
    vh = make_value_head()
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)
    out = vh(x)
    assert out.shape == (BATCH, SEQ_LEN), f"Expected ({BATCH}, {SEQ_LEN}), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2: ValueHead gradient flows
# ---------------------------------------------------------------------------


def test_value_head_grad_flows():
    vh = make_value_head()
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, requires_grad=True)
    out = vh(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# Test 3: GAEComputation advantages + values approx returns (definition)
# ---------------------------------------------------------------------------


def test_gae_returns_definition():
    gae = GAEComputation(gamma=0.99, lam=0.95)
    rewards = torch.rand(BATCH, SEQ_LEN)
    values = torch.rand(BATCH, SEQ_LEN)
    dones = torch.zeros(BATCH, SEQ_LEN)
    advantages, returns = gae.compute(rewards, values, dones)
    # returns = advantages + values by definition
    assert torch.allclose(returns, advantages + values, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: GAEComputation output shape
# ---------------------------------------------------------------------------


def test_gae_output_shape():
    gae = GAEComputation()
    rewards = torch.rand(BATCH, SEQ_LEN)
    values = torch.rand(BATCH, SEQ_LEN)
    dones = torch.zeros(BATCH, SEQ_LEN)
    advantages, returns = gae.compute(rewards, values, dones)
    assert advantages.shape == (BATCH, SEQ_LEN)
    assert returns.shape == (BATCH, SEQ_LEN)


# ---------------------------------------------------------------------------
# Test 5: GAEComputation gamma=0 -> advantage[t] = rewards[t] - values[t]
# ---------------------------------------------------------------------------


def test_gae_gamma_zero_no_discounting():
    gae = GAEComputation(gamma=0.0, lam=0.95)
    rewards = torch.rand(BATCH, SEQ_LEN)
    values = torch.rand(BATCH, SEQ_LEN)
    dones = torch.zeros(BATCH, SEQ_LEN)
    advantages, _ = gae.compute(rewards, values, dones)
    expected = rewards - values
    assert torch.allclose(advantages, expected, atol=1e-5), (
        f"With gamma=0, advantage should equal rewards - values.\n"
        f"max diff: {(advantages - expected).abs().max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 6: GAEComputation done=1 at step t -> no value propagation past that step
# ---------------------------------------------------------------------------


def test_gae_done_blocks_propagation():
    gae = GAEComputation(gamma=0.99, lam=0.95)
    T = SEQ_LEN

    # Set up: reward=1 at step 0 only; all values=0
    rewards = torch.zeros(1, T)
    rewards[0, 0] = 1.0
    values = torch.zeros(1, T)

    # done at step 0 -> episode ends immediately; later steps isolated
    dones = torch.zeros(1, T)
    dones[0, 0] = 1.0

    advantages_done, _ = gae.compute(rewards, values, dones)

    # Without done flag, the non-zero reward at step 0 would not affect steps >0
    # anyway (since it's only the reward at t=0). But done at t=0 means gae resets.
    # The key check: advantage at t>0 should not incorporate gae from t=0.
    # With dones[0]=1, the gae carried forward is multiplied by (1-done[0])=0.
    # So adv[1..T-1] should each be just delta[t] = reward[t] + 0 - value[t] = 0.
    assert torch.allclose(advantages_done[0, 1:], torch.zeros(T - 1), atol=1e-5), (
        "After done=1 at step 0, subsequent advantages should be 0 (no carry-over)."
    )


# ---------------------------------------------------------------------------
# Test 7: PPOLoss all keys present and finite
# ---------------------------------------------------------------------------


def test_ppo_loss_keys_and_finite():
    loss_fn = PPOLoss()
    B, T = BATCH, SEQ_LEN
    log_probs = torch.randn(B, T)
    old_log_probs = torch.randn(B, T)
    advantages = torch.randn(B, T)
    values = torch.randn(B, T)
    returns = torch.randn(B, T)
    ref_log_probs = torch.randn(B, T)

    result = loss_fn(log_probs, old_log_probs, advantages, values, returns, ref_log_probs)
    expected_keys = {
        "total",
        "policy_loss",
        "value_loss",
        "entropy_loss",
        "kl_loss",
        "clip_fraction",
    }
    assert expected_keys == set(result.keys())
    for k, v in result.items():
        assert torch.isfinite(v), f"Loss component '{k}' is not finite: {v}"


# ---------------------------------------------------------------------------
# Test 8: PPOLoss clip_fraction in [0, 1]
# ---------------------------------------------------------------------------


def test_ppo_loss_clip_fraction_range():
    loss_fn = PPOLoss(clip_eps=0.2)
    B, T = BATCH, SEQ_LEN
    log_probs = torch.randn(B, T)
    old_log_probs = torch.randn(B, T)
    advantages = torch.randn(B, T)
    values = torch.randn(B, T)
    returns = torch.randn(B, T)
    ref_log_probs = torch.randn(B, T)

    result = loss_fn(log_probs, old_log_probs, advantages, values, returns, ref_log_probs)
    cf = result["clip_fraction"].item()
    assert 0.0 <= cf <= 1.0, f"clip_fraction={cf} outside [0,1]"


# ---------------------------------------------------------------------------
# Test 9: PPOLoss when log_probs == old_log_probs -> ratio=1, clip_fraction=0
# ---------------------------------------------------------------------------


def test_ppo_loss_no_clip_when_ratio_one():
    loss_fn = PPOLoss(clip_eps=0.2)
    B, T = BATCH, SEQ_LEN
    log_probs = torch.full((B, T), -1.5)
    old_log_probs = log_probs.clone()
    advantages = torch.randn(B, T)
    values = torch.randn(B, T)
    returns = torch.randn(B, T)
    ref_log_probs = torch.randn(B, T)

    result = loss_fn(log_probs, old_log_probs, advantages, values, returns, ref_log_probs)
    assert result["clip_fraction"].item() == 0.0, (
        f"Expected clip_fraction=0 when ratio=1, got {result['clip_fraction'].item()}"
    )


# ---------------------------------------------------------------------------
# Test 10: PPOBuffer add + get_batch correct shapes, clear empties
# ---------------------------------------------------------------------------


def test_ppo_buffer_add_get_clear():
    buf = PPOBuffer(capacity=3)
    B, T = BATCH, SEQ_LEN
    for _ in range(2):
        buf.add(
            input_ids=torch.randint(0, VOCAB, (B, T)),
            log_probs=torch.randn(B, T),
            values=torch.randn(B, T),
            rewards=torch.randn(B, T),
            dones=torch.zeros(B, T),
        )

    batch = buf.get_batch()
    assert batch["input_ids"].shape == (2 * B, T)
    assert batch["log_probs"].shape == (2 * B, T)

    buf.clear()
    assert not buf.is_full()
    assert len(buf._data) == 0


# ---------------------------------------------------------------------------
# Test 11: PPOBuffer is_full() True after capacity episodes
# ---------------------------------------------------------------------------


def test_ppo_buffer_is_full():
    capacity = 3
    buf = PPOBuffer(capacity=capacity)
    B, T = BATCH, SEQ_LEN
    assert not buf.is_full()
    for _ in range(capacity):
        buf.add(
            input_ids=torch.randint(0, VOCAB, (B, T)),
            log_probs=torch.randn(B, T),
            values=torch.randn(B, T),
            rewards=torch.randn(B, T),
            dones=torch.zeros(B, T),
        )
    assert buf.is_full()


# ---------------------------------------------------------------------------
# Test 12: PPOTrainer ref model frozen
# ---------------------------------------------------------------------------


def test_ppo_trainer_ref_model_frozen():
    policy = make_policy()
    ref = make_ref()
    vh = make_value_head()
    trainer = PPOTrainer(
        policy_model=policy,
        ref_model=ref,
        value_head=vh,
        reward_fn=dummy_reward_fn,
        optimizer_policy=torch.optim.Adam(policy.parameters(), lr=1e-4),
        optimizer_value=torch.optim.Adam(vh.parameters(), lr=1e-4),
        ppo_epochs=PPO_EPOCHS,
    )
    for param in trainer.ref_model.parameters():
        assert not param.requires_grad, "ref_model parameter should be frozen"


# ---------------------------------------------------------------------------
# Test 13: PPOTrainer.rollout returns correct keys and shapes
# ---------------------------------------------------------------------------


def test_ppo_trainer_rollout_keys_and_shapes():
    policy = make_policy()
    ref = make_ref()
    vh = make_value_head()
    trainer = PPOTrainer(
        policy_model=policy,
        ref_model=ref,
        value_head=vh,
        reward_fn=dummy_reward_fn,
        optimizer_policy=torch.optim.Adam(policy.parameters(), lr=1e-4),
        optimizer_value=torch.optim.Adam(vh.parameters(), lr=1e-4),
        ppo_epochs=PPO_EPOCHS,
    )
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    result = trainer.rollout(input_ids, max_new_tokens=MAX_NEW_TOKENS)

    assert set(result.keys()) == {"sequences", "log_probs", "values", "rewards"}
    assert result["sequences"].shape == (BATCH, SEQ_LEN + MAX_NEW_TOKENS)
    assert result["log_probs"].shape == (BATCH, MAX_NEW_TOKENS)
    assert result["values"].shape == (BATCH, MAX_NEW_TOKENS)
    assert result["rewards"].shape == (BATCH,)


# ---------------------------------------------------------------------------
# Test 14: PPOTrainer.rollout rewards come from reward_fn applied to sequences
# ---------------------------------------------------------------------------


def test_ppo_trainer_rollout_rewards_from_reward_fn():
    called_with: list = []

    def tracking_reward_fn(sequences: torch.Tensor) -> torch.Tensor:
        called_with.append(sequences.clone())
        return torch.ones(sequences.shape[0]) * 3.14

    policy = make_policy()
    ref = make_ref()
    vh = make_value_head()
    trainer = PPOTrainer(
        policy_model=policy,
        ref_model=ref,
        value_head=vh,
        reward_fn=tracking_reward_fn,
        optimizer_policy=torch.optim.Adam(policy.parameters(), lr=1e-4),
        optimizer_value=torch.optim.Adam(vh.parameters(), lr=1e-4),
        ppo_epochs=PPO_EPOCHS,
    )
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    result = trainer.rollout(input_ids, max_new_tokens=MAX_NEW_TOKENS)

    assert len(called_with) == 1, "reward_fn should be called exactly once per rollout"
    assert called_with[0].shape == (BATCH, SEQ_LEN + MAX_NEW_TOKENS)
    assert torch.allclose(result["rewards"], torch.full((BATCH,), 3.14))


# ---------------------------------------------------------------------------
# Test 15: PPOTrainer.train_epoch returns all keys, loss is finite
# ---------------------------------------------------------------------------


def test_ppo_trainer_train_epoch_keys_and_finite():
    policy = make_policy()
    ref = make_ref()
    vh = make_value_head()
    trainer = PPOTrainer(
        policy_model=policy,
        ref_model=ref,
        value_head=vh,
        reward_fn=dummy_reward_fn,
        optimizer_policy=torch.optim.Adam(policy.parameters(), lr=1e-4),
        optimizer_value=torch.optim.Adam(vh.parameters(), lr=1e-4),
        ppo_epochs=PPO_EPOCHS,
    )
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    result = trainer.train_epoch(input_ids)

    expected_keys = {"avg_total_loss", "policy_loss", "value_loss", "kl_loss", "mean_reward"}
    assert expected_keys == set(result.keys())
    for k, v in result.items():
        assert isinstance(v, float), f"Expected float for key '{k}', got {type(v)}"
        assert not (v != v), f"Key '{k}' is NaN"  # NaN check


# ---------------------------------------------------------------------------
# Test 16: PPOTrainer policy params updated after train_epoch
# ---------------------------------------------------------------------------


def test_ppo_trainer_policy_params_updated():
    policy = make_policy()
    ref = make_ref()
    vh = make_value_head()

    # Snapshot initial params
    initial_params = {n: p.clone().detach() for n, p in policy.named_parameters()}

    trainer = PPOTrainer(
        policy_model=policy,
        ref_model=ref,
        value_head=vh,
        reward_fn=dummy_reward_fn,
        optimizer_policy=torch.optim.Adam(policy.parameters(), lr=1e-2),
        optimizer_value=torch.optim.Adam(vh.parameters(), lr=1e-2),
        ppo_epochs=PPO_EPOCHS,
    )
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    trainer.train_epoch(input_ids)

    changed = False
    for n, p in policy.named_parameters():
        if not torch.allclose(p.detach(), initial_params[n]):
            changed = True
            break
    assert changed, "Policy parameters should have been updated after train_epoch"


# ---------------------------------------------------------------------------
# Test 17 (counts as test 15 in the spec): Full 2-epoch rollout+train cycle
# ---------------------------------------------------------------------------


def test_full_two_epoch_cycle():
    policy = make_policy()
    ref = make_ref()
    vh = make_value_head()
    trainer = PPOTrainer(
        policy_model=policy,
        ref_model=ref,
        value_head=vh,
        reward_fn=dummy_reward_fn,
        optimizer_policy=torch.optim.Adam(policy.parameters(), lr=1e-4),
        optimizer_value=torch.optim.Adam(vh.parameters(), lr=1e-4),
        ppo_epochs=PPO_EPOCHS,
    )
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    for epoch in range(2):
        metrics = trainer.train_epoch(input_ids)
        assert isinstance(metrics, dict), f"Epoch {epoch}: train_epoch should return a dict"
        assert "avg_total_loss" in metrics
