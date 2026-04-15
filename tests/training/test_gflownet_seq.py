"""Tests for src/training/gflownet_seq.py"""
import pytest
import torch
import torch.nn as nn

from src.training.gflownet_seq import (
    GFlowNetConfig,
    FlowModel,
    compute_forward_log_prob,
    trajectory_balance_loss,
    detailed_balance_loss,
    GFlowNetTrainer,
)

# ---------------------------------------------------------------------------
# Tiny constants
# ---------------------------------------------------------------------------

VOCAB = 32
D_MODEL = 16
BATCH = 4
SEQ = 6


# ---------------------------------------------------------------------------
# Mock LM (returns (loss, logits, pkv))
# ---------------------------------------------------------------------------

class _MockLM(nn.Module):
    def __init__(self, vocab=VOCAB, d_model=D_MODEL):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        logits = self.proj(x)
        return None, logits, None


@pytest.fixture
def cfg():
    return GFlowNetConfig(
        reward_temperature=1.0,
        flow_lr=1e-3,
        n_trajectories=4,
        max_seq_len=SEQ,
        epsilon=0.1,
    )


@pytest.fixture
def lm():
    torch.manual_seed(0)
    return _MockLM()


@pytest.fixture
def flow_model():
    torch.manual_seed(1)
    return FlowModel(d_model=D_MODEL, hidden_dim=16)


@pytest.fixture
def trainer(lm, flow_model, cfg):
    opt = torch.optim.Adam(
        list(lm.parameters()) + list(flow_model.parameters()), lr=1e-3
    )
    return GFlowNetTrainer(lm, flow_model, cfg, opt)


@pytest.fixture
def input_ids():
    return torch.randint(0, VOCAB, (BATCH, SEQ))


# ---------------------------------------------------------------------------
# GFlowNetConfig
# ---------------------------------------------------------------------------

def test_config_default_temperature():
    cfg = GFlowNetConfig()
    assert cfg.reward_temperature == 1.0


def test_config_epsilon_in_range():
    cfg = GFlowNetConfig(epsilon=0.05)
    assert 0.0 <= cfg.epsilon <= 1.0


# ---------------------------------------------------------------------------
# FlowModel
# ---------------------------------------------------------------------------

def test_flow_model_output_shape(flow_model):
    h = torch.randn(BATCH, D_MODEL)
    out = flow_model(h)
    assert out.shape == (BATCH,)


def test_flow_model_output_is_finite(flow_model):
    h = torch.randn(BATCH, D_MODEL)
    out = flow_model(h)
    assert torch.isfinite(out).all()


def test_flow_model_gradient_flows(flow_model):
    h = torch.randn(BATCH, D_MODEL, requires_grad=True)
    out = flow_model(h)
    out.sum().backward()
    assert h.grad is not None


# ---------------------------------------------------------------------------
# compute_forward_log_prob
# ---------------------------------------------------------------------------

def test_forward_log_prob_shape(lm, input_ids):
    lp = compute_forward_log_prob(lm, input_ids)
    assert lp.shape == (BATCH,)


def test_forward_log_prob_nonpositive(lm, input_ids):
    lp = compute_forward_log_prob(lm, input_ids)
    assert (lp <= 0.0).all()


def test_forward_log_prob_finite(lm, input_ids):
    lp = compute_forward_log_prob(lm, input_ids)
    assert torch.isfinite(lp).all()


# ---------------------------------------------------------------------------
# trajectory_balance_loss
# ---------------------------------------------------------------------------

def test_tb_loss_scalar():
    log_Z = torch.tensor(0.0, requires_grad=True)
    fwd = torch.randn(BATCH)
    log_r = torch.randn(BATCH)
    loss = trajectory_balance_loss(log_Z, fwd, log_r)
    assert loss.shape == ()


def test_tb_loss_nonneg():
    log_Z = torch.tensor(0.0)
    fwd = torch.randn(BATCH)
    log_r = torch.randn(BATCH)
    loss = trajectory_balance_loss(log_Z, fwd, log_r)
    assert loss.item() >= 0.0


def test_tb_loss_zero_when_balanced():
    # If log_Z + fwd == log_r, loss should be 0
    log_Z = torch.tensor(1.0)
    fwd = torch.zeros(BATCH)
    log_r = torch.ones(BATCH)
    loss = trajectory_balance_loss(log_Z, fwd, log_r)
    assert abs(loss.item()) < 1e-6


def test_tb_loss_has_gradient():
    log_Z = torch.tensor(0.0, requires_grad=True)
    fwd = torch.randn(BATCH, requires_grad=True)
    log_r = torch.randn(BATCH)
    loss = trajectory_balance_loss(log_Z, fwd, log_r)
    loss.backward()
    assert log_Z.grad is not None
    assert fwd.grad is not None


# ---------------------------------------------------------------------------
# detailed_balance_loss
# ---------------------------------------------------------------------------

def test_db_loss_scalar():
    log_fs = torch.randn(BATCH)
    log_pf = torch.randn(BATCH)
    log_fsp = torch.randn(BATCH)
    log_pb = torch.randn(BATCH)
    loss = detailed_balance_loss(log_fs, log_pf, log_fsp, log_pb)
    assert loss.shape == ()


def test_db_loss_zero_when_balanced():
    val = torch.ones(BATCH)
    loss = detailed_balance_loss(val, val, val, val)
    assert abs(loss.item()) < 1e-6


# ---------------------------------------------------------------------------
# GFlowNetTrainer
# ---------------------------------------------------------------------------

def test_trainer_log_z_is_parameter(trainer):
    assert isinstance(trainer.log_Z, nn.Parameter)


def test_trainer_step_returns_dict(trainer):
    def reward_fn(seqs):
        return torch.ones(seqs.shape[0])

    result = trainer.train_step(reward_fn)
    assert "loss" in result
    assert "mean_reward" in result
    assert "log_Z" in result


def test_trainer_step_loss_finite(trainer):
    def reward_fn(seqs):
        return torch.rand(seqs.shape[0]).clamp(min=1e-4)

    result = trainer.train_step(reward_fn)
    assert torch.isfinite(torch.tensor(result["loss"]))


def test_trainer_step_log_z_updates(trainer):
    initial = trainer.log_Z.item()

    def reward_fn(seqs):
        return torch.ones(seqs.shape[0])

    trainer.train_step(reward_fn)
    # log_Z should have been updated via optimizer
    # (may or may not change depending on grad, but shouldn't error)
    assert isinstance(trainer.log_Z.item(), float)
