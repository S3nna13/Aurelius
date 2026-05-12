from unittest.mock import MagicMock

import torch
import torch.nn as nn

from src.alignment.praxis.config import PRAXISConfig
from src.alignment.praxis.trainer import PRAXISTrainer


class FakeRouter(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)


class FakeFFN(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.router = FakeRouter(d_model, n_experts)


class FakeBlock(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.ffn = FakeFFN(d_model, n_experts)


def make_tiny_model(d_model=16, vocab_size=64, n_layers=4):
    """Minimal mock matching AureliusTransformer's interface."""
    model = MagicMock()
    model.layers = nn.ModuleList([FakeBlock(d_model, 4) for _ in range(n_layers)])
    B, T = 2, 8
    model.return_value = (
        torch.tensor(1.0),
        torch.randn(B, T, vocab_size),
        None,
        torch.randn(B, T, d_model),
    )
    model.train = MagicMock()
    return model


def test_trainer_train_step_returns_metrics():
    D, V = 16, 64
    cfg = PRAXISConfig(
        d_model=D,
        n_principles=2,
        mc_dropout_n=2,
        steer_layers=[0, 1],
        safety_experts=[0, 1],
        n_group=2,
        max_new_tokens=4,
        warp_interval=9999,
    )
    model = make_tiny_model(d_model=D, vocab_size=V)
    ref_sd = {}
    sft_sd = {}
    trainer = PRAXISTrainer(model, cfg, ref_state_dict=ref_sd, sft_state_dict=sft_sd)

    batch = {
        "input_ids": torch.randint(0, V, (2, 8)),
        "labels": torch.randint(0, V, (2, 8)),
        "attention_mask": torch.ones(2, 8, dtype=torch.bool),
    }
    metrics = trainer.train_step(batch, step=1)
    assert "total_loss" in metrics or "dapo_loss" in metrics, f"metrics: {metrics}"
