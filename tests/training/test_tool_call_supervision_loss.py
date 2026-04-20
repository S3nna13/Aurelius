"""Tests for tool_call_supervision_loss."""

from __future__ import annotations

import pytest
import torch

from src.training.tool_call_supervision_loss import ToolCallSupervisionLoss


def _tiny_batch():
    torch.manual_seed(0)
    b, t, v = 2, 8, 32
    logits = torch.randn(b, t, v, requires_grad=True)
    labels = torch.full((b, t), -100)
    labels[:, 3:5] = torch.tensor([5, 7])
    mask = torch.zeros(b, t, dtype=torch.bool)
    mask[:, 3:5] = True
    return logits, labels, mask


def test_forward_matches_manual_ce():
    logits, labels, mask = _tiny_batch()
    crit = ToolCallSupervisionLoss()
    loss = crit(logits, labels, mask)
    assert loss.ndim == 0
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    active = mask.reshape(-1) & (flat_labels != -100)
    manual = torch.nn.functional.cross_entropy(
        flat_logits[active],
        flat_labels[active],
        reduction="mean",
    )
    assert torch.allclose(loss, manual)


def test_backward_finite():
    logits, labels, mask = _tiny_batch()
    crit = ToolCallSupervisionLoss()
    loss = crit(logits, labels, mask)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_zero_active_raises():
    logits = torch.randn(1, 4, 8)
    labels = torch.full((1, 4), -100)
    mask = torch.ones(1, 4, dtype=torch.bool)
    crit = ToolCallSupervisionLoss()
    with pytest.raises(RuntimeError, match="zero active"):
        crit(logits, labels, mask)


def test_all_masked_positions_raises():
    logits = torch.randn(1, 4, 8)
    labels = torch.tensor([[1, 2, 3, 4]])
    mask = torch.zeros(1, 4, dtype=torch.bool)
    crit = ToolCallSupervisionLoss()
    with pytest.raises(RuntimeError):
        crit(logits, labels, mask)


def test_shape_errors():
    crit = ToolCallSupervisionLoss()
    with pytest.raises(ValueError):
        crit(torch.randn(4, 8), torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, 4, dtype=torch.bool))
    with pytest.raises(ValueError):
        crit(torch.randn(2, 4, 8), torch.zeros(2, 5, dtype=torch.long), torch.zeros(2, 4, dtype=torch.bool))


def test_bool_mask_required():
    crit = ToolCallSupervisionLoss()
    with pytest.raises(TypeError):
        crit(
            torch.randn(1, 2, 3),
            torch.zeros(1, 2, dtype=torch.long),
            torch.zeros(1, 2, dtype=torch.long),
        )


def test_seq_len_one():
    torch.manual_seed(1)
    logits = torch.randn(1, 1, 16, requires_grad=True)
    labels = torch.tensor([[3]])
    mask = torch.tensor([[True]])
    loss = ToolCallSupervisionLoss()(logits, labels, mask)
    loss.backward()
    assert torch.isfinite(loss)


def test_determinism():
    torch.manual_seed(42)
    logits = torch.randn(1, 5, 10)
    labels = torch.tensor([[0, -100, -100, 2, 3]])
    mask = torch.tensor([[False, False, False, True, True]])
    a = ToolCallSupervisionLoss()(logits, labels, mask)
    torch.manual_seed(42)
    logits2 = torch.randn(1, 5, 10)
    labels2 = torch.tensor([[0, -100, -100, 2, 3]])
    mask2 = torch.tensor([[False, False, False, True, True]])
    b = ToolCallSupervisionLoss()(logits2, labels2, mask2)
    assert torch.allclose(a, b)
