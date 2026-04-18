"""
Tests for src/inference/world_model_planning.py

Configuration used across all tests:
    d_model=16, d_latent=8, vocab_size=16, n_actions=4, seq_len=6, B=2, H=3
"""

import math
import torch
import pytest

from src.inference.world_model_planning import (
    LatentEncoder,
    LatentTransitionModel,
    RewardPredictor,
    LatentDecoder,
    WorldModel,
    WorldModelTrainer,
    PlanningAgent,
    WorldModelConfig,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

D_MODEL = 16
D_LATENT = 8
VOCAB_SIZE = 16
N_ACTIONS = 4
SEQ_LEN = 6
B = 2
H = 3


def make_world_model() -> WorldModel:
    return WorldModel(
        d_model=D_MODEL,
        d_latent=D_LATENT,
        vocab_size=VOCAB_SIZE,
        n_actions=N_ACTIONS,
        seq_len=SEQ_LEN,
    )


def make_input_ids(seq_len: int = SEQ_LEN) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (B, seq_len))


def make_latent() -> torch.Tensor:
    return torch.randn(B, D_LATENT)


def make_actions() -> torch.Tensor:
    return torch.randint(0, N_ACTIONS, (B,))


# ---------------------------------------------------------------------------
# LatentEncoder
# ---------------------------------------------------------------------------

def test_latent_encoder_output_shape():
    encoder = LatentEncoder(D_MODEL, D_LATENT, VOCAB_SIZE)
    ids = make_input_ids()
    z = encoder(ids)
    assert z.shape == (B, D_LATENT), f"Expected ({B}, {D_LATENT}), got {z.shape}"


def test_latent_encoder_no_nan():
    encoder = LatentEncoder(D_MODEL, D_LATENT, VOCAB_SIZE)
    ids = make_input_ids()
    z = encoder(ids)
    assert not torch.isnan(z).any(), "LatentEncoder output contains NaN"


# ---------------------------------------------------------------------------
# LatentTransitionModel
# ---------------------------------------------------------------------------

def test_transition_model_output_shape():
    model = LatentTransitionModel(D_LATENT, N_ACTIONS)
    z = make_latent()
    a = make_actions()
    z_next = model(z, a)
    assert z_next.shape == (B, D_LATENT), f"Expected ({B}, {D_LATENT}), got {z_next.shape}"


def test_transition_model_different_actions_produce_different_z_next():
    """Two distinct actions from the same state should yield different z_next."""
    model = LatentTransitionModel(D_LATENT, N_ACTIONS)
    model.eval()
    z = make_latent()
    a0 = torch.zeros(B, dtype=torch.long)
    a1 = torch.ones(B, dtype=torch.long)
    with torch.no_grad():
        z0 = model(z, a0)
        z1 = model(z, a1)
    assert not torch.allclose(z0, z1), "Different actions should yield different z_next"


def test_transition_model_no_nan():
    model = LatentTransitionModel(D_LATENT, N_ACTIONS)
    z = make_latent()
    a = make_actions()
    z_next = model(z, a)
    assert not torch.isnan(z_next).any()


# ---------------------------------------------------------------------------
# RewardPredictor
# ---------------------------------------------------------------------------

def test_reward_predictor_output_shape():
    rp = RewardPredictor(D_LATENT)
    z = make_latent()
    reward = rp(z)
    assert reward.shape == (B,), f"Expected ({B},), got {reward.shape}"


def test_reward_predictor_scalar_per_batch():
    rp = RewardPredictor(D_LATENT)
    z = make_latent()
    reward = rp(z)
    assert reward.ndim == 1


# ---------------------------------------------------------------------------
# LatentDecoder
# ---------------------------------------------------------------------------

def test_latent_decoder_output_shape():
    decoder = LatentDecoder(D_LATENT, D_MODEL, VOCAB_SIZE, SEQ_LEN)
    z = make_latent()
    logits = decoder(z)
    assert logits.shape == (B, SEQ_LEN, VOCAB_SIZE), (
        f"Expected ({B}, {SEQ_LEN}, {VOCAB_SIZE}), got {logits.shape}"
    )


def test_latent_decoder_no_nan():
    decoder = LatentDecoder(D_LATENT, D_MODEL, VOCAB_SIZE, SEQ_LEN)
    z = make_latent()
    logits = decoder(z)
    assert not torch.isnan(logits).any()


# ---------------------------------------------------------------------------
# WorldModel
# ---------------------------------------------------------------------------

def test_world_model_encode_shape():
    wm = make_world_model()
    ids = make_input_ids()
    z = wm.encode(ids)
    assert z.shape == (B, D_LATENT)


def test_world_model_predict_next_shapes():
    wm = make_world_model()
    z = make_latent()
    a = make_actions()
    z_next, reward = wm.predict_next(z, a)
    assert z_next.shape == (B, D_LATENT), f"z_next shape wrong: {z_next.shape}"
    assert reward.shape == (B,), f"reward shape wrong: {reward.shape}"


def test_world_model_decode_shape():
    wm = make_world_model()
    z = make_latent()
    logits = wm.decode(z)
    assert logits.shape == (B, SEQ_LEN, VOCAB_SIZE)


def test_world_model_imagination_rollout_trajectory_shape():
    wm = make_world_model()
    z0 = make_latent()
    actions = torch.randint(0, N_ACTIONS, (B, H))
    z_traj, rewards = wm.imagination_rollout(z0, actions)
    assert z_traj.shape == (B, H + 1, D_LATENT), (
        f"Expected ({B}, {H+1}, {D_LATENT}), got {z_traj.shape}"
    )


def test_world_model_imagination_rollout_rewards_shape():
    wm = make_world_model()
    z0 = make_latent()
    actions = torch.randint(0, N_ACTIONS, (B, H))
    z_traj, rewards = wm.imagination_rollout(z0, actions)
    assert rewards.shape == (B, H), f"Expected ({B}, {H}), got {rewards.shape}"


def test_world_model_imagination_rollout_z0_preserved():
    """The first element of the trajectory should equal z0."""
    wm = make_world_model()
    z0 = make_latent()
    actions = torch.randint(0, N_ACTIONS, (B, H))
    z_traj, _ = wm.imagination_rollout(z0, actions)
    assert torch.allclose(z_traj[:, 0, :], z0), "z_traj[:, 0, :] should equal z0"


# ---------------------------------------------------------------------------
# WorldModelTrainer
# ---------------------------------------------------------------------------

def test_trainer_reconstruction_loss_finite():
    wm = make_world_model()
    trainer = WorldModelTrainer(wm, lr=1e-3)
    ids = make_input_ids()
    loss = trainer.reconstruction_loss(ids)
    assert loss.ndim == 0, "reconstruction_loss should be a scalar"
    assert math.isfinite(loss.item()), f"reconstruction_loss not finite: {loss.item()}"


def test_trainer_transition_loss_finite():
    wm = make_world_model()
    trainer = WorldModelTrainer(wm, lr=1e-3)
    z = make_latent()
    a = make_actions()
    z_next_true = make_latent()
    loss = trainer.transition_loss(z, a, z_next_true)
    assert loss.ndim == 0
    assert math.isfinite(loss.item()), f"transition_loss not finite: {loss.item()}"


def test_trainer_train_step_returns_loss_dict():
    wm = make_world_model()
    trainer = WorldModelTrainer(wm, lr=1e-3)
    ids_t = make_input_ids()
    ids_next = make_input_ids()
    actions = make_actions()
    rewards = torch.randn(B)
    result = trainer.train_step(ids_t, ids_next, actions, rewards)
    assert isinstance(result, dict), "train_step should return a dict"
    for key in ("reconstruction_loss", "transition_loss", "reward_loss", "total_loss"):
        assert key in result, f"Missing key '{key}' in train_step output"


def test_trainer_train_step_losses_finite():
    wm = make_world_model()
    trainer = WorldModelTrainer(wm, lr=1e-3)
    ids_t = make_input_ids()
    ids_next = make_input_ids()
    actions = make_actions()
    rewards = torch.randn(B)
    result = trainer.train_step(ids_t, ids_next, actions, rewards)
    for key, val in result.items():
        assert math.isfinite(val), f"Loss '{key}' is not finite: {val}"


# ---------------------------------------------------------------------------
# PlanningAgent
# ---------------------------------------------------------------------------

def test_planning_agent_random_shooting_shape():
    wm = make_world_model()
    agent = PlanningAgent(wm, horizon=H, n_rollouts=4)
    z0 = make_latent()
    best = agent.random_shooting(z0)
    assert best.shape == (B, H), f"Expected ({B}, {H}), got {best.shape}"


def test_planning_agent_random_shooting_valid_actions():
    wm = make_world_model()
    agent = PlanningAgent(wm, horizon=H, n_rollouts=4)
    z0 = make_latent()
    best = agent.random_shooting(z0)
    assert (best >= 0).all() and (best < N_ACTIONS).all(), (
        "random_shooting returned out-of-range action indices"
    )


def test_planning_agent_greedy_rollout_shape():
    wm = make_world_model()
    agent = PlanningAgent(wm, horizon=H, n_rollouts=4)
    z0 = make_latent()
    actions = agent.greedy_rollout(z0)
    assert actions.shape == (B, H), f"Expected ({B}, {H}), got {actions.shape}"


def test_planning_agent_greedy_rollout_valid_actions():
    wm = make_world_model()
    agent = PlanningAgent(wm, horizon=H, n_rollouts=4)
    z0 = make_latent()
    actions = agent.greedy_rollout(z0)
    assert (actions >= 0).all() and (actions < N_ACTIONS).all(), (
        "greedy_rollout returned out-of-range action indices"
    )


# ---------------------------------------------------------------------------
# WorldModelConfig defaults
# ---------------------------------------------------------------------------

def test_world_model_config_defaults():
    cfg = WorldModelConfig()
    assert cfg.d_model == 32
    assert cfg.d_latent == 16
    assert cfg.vocab_size == 64
    assert cfg.n_actions == 8
    assert cfg.seq_len == 8
    assert cfg.horizon == 4
    assert cfg.n_rollouts == 8
    assert math.isclose(cfg.lr, 1e-4, rel_tol=1e-6)


def test_world_model_config_override():
    cfg = WorldModelConfig(d_model=64, d_latent=32, lr=5e-4)
    assert cfg.d_model == 64
    assert cfg.d_latent == 32
    assert math.isclose(cfg.lr, 5e-4, rel_tol=1e-6)
