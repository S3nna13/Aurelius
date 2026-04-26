"""Integration: auxiliary loss registry + config."""

from __future__ import annotations

import torch

import src.training as training


def test_auxiliary_loss_registry():
    assert hasattr(training, "AUXILIARY_LOSS_REGISTRY")
    assert (
        training.AUXILIARY_LOSS_REGISTRY.get("tool_call_supervision")
        is training.ToolCallSupervisionLoss
    )


def test_config_default_off():
    from src.model.config import AureliusConfig

    assert AureliusConfig().training_tool_call_supervision_enabled is False


def test_smoke_forward():
    crit = training.ToolCallSupervisionLoss()
    logits = torch.randn(1, 3, 12, requires_grad=True)
    labels = torch.tensor([[1, -100, 2]])
    mask = torch.tensor([[True, False, True]])
    loss = crit(logits, labels, mask)
    loss.backward()
    assert torch.isfinite(loss)
