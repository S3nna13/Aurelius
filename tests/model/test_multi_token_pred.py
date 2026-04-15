"""Tests for src/model/multi_token_pred.py"""
import pytest
import torch
import torch.nn as nn

from src.model.multi_token_pred import (
    MTPConfig,
    MTPHead,
    mtp_loss,
    MTPModel,
    MTPTrainer,
)

# ---------------------------------------------------------------------------
# Tiny constants
# ---------------------------------------------------------------------------

D_MODEL = 32
VOCAB = 64
SEQ = 10
BATCH = 2
N_FUTURE = 3


# ---------------------------------------------------------------------------
# Mock backbone
# ---------------------------------------------------------------------------

class _PassthroughLayer(nn.Module):
    def forward(self, x, freqs_cis, mask=None, past_kv=None):
        return x, None


class _MockBackbone(nn.Module):
    def __init__(self, d_model=D_MODEL, vocab=VOCAB, max_seq=64):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([_PassthroughLayer(), _PassthroughLayer()])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.register_buffer("freqs_cis", torch.ones(max_seq, d_model // 2))

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x, _ = layer(x, self.freqs_cis[: input_ids.shape[1]])
        x = self.norm(x)
        return None, self.lm_head(x), None


@pytest.fixture
def cfg():
    return MTPConfig(n_future_tokens=N_FUTURE, d_model=D_MODEL, vocab_size=VOCAB)


@pytest.fixture
def backbone():
    torch.manual_seed(0)
    return _MockBackbone()


@pytest.fixture
def mtp_model(backbone, cfg):
    return MTPModel(backbone, cfg)


@pytest.fixture
def input_ids():
    return torch.randint(0, VOCAB, (BATCH, SEQ))


# ---------------------------------------------------------------------------
# MTPConfig
# ---------------------------------------------------------------------------

def test_config_default_loss_weights():
    cfg = MTPConfig(n_future_tokens=4, d_model=D_MODEL, vocab_size=VOCAB)
    assert len(cfg.loss_weights) == 4
    assert abs(sum(cfg.loss_weights) - 1.0) < 1e-6


def test_config_custom_loss_weights():
    weights = [0.5, 0.3, 0.2]
    cfg = MTPConfig(n_future_tokens=3, d_model=D_MODEL, vocab_size=VOCAB, loss_weights=weights)
    assert cfg.loss_weights == weights


def test_config_wrong_weights_length_raises():
    with pytest.raises(ValueError):
        MTPConfig(n_future_tokens=3, d_model=D_MODEL, vocab_size=VOCAB, loss_weights=[0.5, 0.5])


def test_config_n_future_tokens_one():
    cfg = MTPConfig(n_future_tokens=1, d_model=D_MODEL, vocab_size=VOCAB)
    assert cfg.loss_weights == [1.0]


# ---------------------------------------------------------------------------
# MTPHead
# ---------------------------------------------------------------------------

def test_mtp_head_output_count(cfg):
    head = MTPHead(cfg)
    hidden = torch.randn(BATCH, SEQ, D_MODEL)
    out = head(hidden)
    assert len(out) == N_FUTURE


def test_mtp_head_output_shape(cfg):
    head = MTPHead(cfg)
    hidden = torch.randn(BATCH, SEQ, D_MODEL)
    out = head(hidden)
    for logits in out:
        assert logits.shape == (BATCH, SEQ, VOCAB)


def test_mtp_head_distinct_heads(cfg):
    head = MTPHead(cfg)
    hidden = torch.randn(BATCH, SEQ, D_MODEL)
    out = head(hidden)
    # Different heads should produce different outputs
    assert not torch.allclose(out[0], out[1])


def test_mtp_head_n_modules(cfg):
    head = MTPHead(cfg)
    assert len(head.heads) == N_FUTURE


# ---------------------------------------------------------------------------
# mtp_loss
# ---------------------------------------------------------------------------

def test_mtp_loss_is_scalar(cfg, input_ids):
    head = MTPHead(cfg)
    hidden = torch.randn(BATCH, SEQ, D_MODEL)
    logits_list = head(hidden)
    loss = mtp_loss(logits_list, input_ids, cfg.loss_weights)
    assert loss.shape == ()


def test_mtp_loss_is_positive(cfg, input_ids):
    head = MTPHead(cfg)
    hidden = torch.randn(BATCH, SEQ, D_MODEL)
    logits_list = head(hidden)
    loss = mtp_loss(logits_list, input_ids, cfg.loss_weights)
    assert loss.item() > 0.0


def test_mtp_loss_has_gradient(cfg, input_ids):
    head = MTPHead(cfg)
    hidden = torch.randn(BATCH, SEQ, D_MODEL, requires_grad=True)
    logits_list = head(hidden)
    loss = mtp_loss(logits_list, input_ids, cfg.loss_weights)
    loss.backward()
    assert hidden.grad is not None


def test_mtp_loss_single_head(input_ids):
    cfg1 = MTPConfig(n_future_tokens=1, d_model=D_MODEL, vocab_size=VOCAB)
    head = MTPHead(cfg1)
    hidden = torch.randn(BATCH, SEQ, D_MODEL)
    logits_list = head(hidden)
    loss = mtp_loss(logits_list, input_ids, cfg1.loss_weights)
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# MTPModel
# ---------------------------------------------------------------------------

def test_mtp_model_forward_primary_shape(mtp_model, input_ids):
    primary, _ = mtp_model(input_ids)
    assert primary.shape == (BATCH, SEQ, VOCAB)


def test_mtp_model_forward_aux_count(mtp_model, input_ids):
    _, aux = mtp_model(input_ids)
    assert len(aux) == N_FUTURE


def test_mtp_model_forward_aux_shapes(mtp_model, input_ids):
    _, aux = mtp_model(input_ids)
    for logits in aux:
        assert logits.shape == (BATCH, SEQ, VOCAB)


# ---------------------------------------------------------------------------
# MTPTrainer
# ---------------------------------------------------------------------------

def test_mtp_trainer_step_returns_dict(mtp_model, cfg, input_ids):
    opt = torch.optim.Adam(mtp_model.parameters(), lr=1e-3)
    trainer = MTPTrainer(mtp_model, cfg, opt)
    result = trainer.train_step(input_ids)
    assert isinstance(result, dict)
    assert "total_loss" in result
    assert "per_head_losses" in result


def test_mtp_trainer_step_per_head_length(mtp_model, cfg, input_ids):
    opt = torch.optim.Adam(mtp_model.parameters(), lr=1e-3)
    trainer = MTPTrainer(mtp_model, cfg, opt)
    result = trainer.train_step(input_ids)
    assert len(result["per_head_losses"]) == N_FUTURE


def test_mtp_trainer_total_loss_positive(mtp_model, cfg, input_ids):
    opt = torch.optim.Adam(mtp_model.parameters(), lr=1e-3)
    trainer = MTPTrainer(mtp_model, cfg, opt)
    result = trainer.train_step(input_ids)
    assert result["total_loss"] > 0.0
