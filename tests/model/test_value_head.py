"""Tests for ValueHead module used by RLHF PPO training."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.model.value_head import ValueHead


class _DummyBackbone(nn.Module):
    def __init__(self, d_model: int = 32, vocab: int = 100):
        super().__init__()
        self.d_model = d_model
        self.config = type("Cfg", (), {"d_model": d_model})()
        self.embed = nn.Embedding(vocab, d_model)
        self.linear = nn.Linear(d_model, vocab)

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embed(input_ids)
        logits = self.linear(hidden)
        return logits, hidden


def test_value_head_forward_shapes():
    backbone = _DummyBackbone(d_model=32, vocab=50)
    vh = ValueHead(backbone, hidden_dim=32)
    input_ids = torch.randint(0, 50, (2, 10))

    logits, hidden, values = vh(input_ids)

    assert logits.shape == (2, 10, 50)
    assert hidden.shape == (2, 10, 32)
    assert values.shape == (2, 10)


def test_value_head_infers_hidden_dim():
    backbone = _DummyBackbone(d_model=64, vocab=50)
    vh = ValueHead(backbone)
    input_ids = torch.randint(0, 50, (1, 5))

    logits, hidden, values = vh(input_ids)
    assert values.shape == (1, 5)


def test_value_head_gradient_flows():
    backbone = _DummyBackbone(d_model=16, vocab=20)
    vh = ValueHead(backbone, hidden_dim=16)
    input_ids = torch.randint(0, 20, (1, 4))

    logits, _, values = vh(input_ids)
    loss = values.sum()
    loss.backward()

    assert vh.value_head.weight.grad is not None
