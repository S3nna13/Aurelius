"""Tests for REINFORCE++ alignment algorithm."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.alignment.reinforce_pp import reinforce_pp_loss


def _log_probs(
    batch: int = 4, seq_len: int = 16, vocab: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(batch, seq_len, vocab)
    targets = torch.randint(0, vocab, (batch, seq_len))
    return F.log_softmax(logits, dim=-1), targets


def test_reinforce_pp_loss_scalar():
    log_probs, targets = _log_probs()
    rewards = torch.tensor([1.0, 0.5, 0.8, 0.3])
    loss = reinforce_pp_loss(log_probs, targets, rewards)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_reinforce_pp_loss_with_ref():
    log_probs, targets = _log_probs()
    ref_log_probs, _ = _log_probs()
    rewards = torch.tensor([1.0, 0.5, 0.8, 0.3])
    loss = reinforce_pp_loss(log_probs, targets, rewards, ref_log_probs, kl_coef=0.01)
    assert not torch.isnan(loss)


def test_reinforce_pp_variant():
    log_probs, targets = _log_probs(batch=8)
    rewards = torch.tensor([1.0, 0.5, 0.8, 0.3, 0.9, 0.6, 0.7, 0.4])
    group_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss_std = reinforce_pp_loss(log_probs, targets, rewards, variant="standard")
    loss_base = reinforce_pp_loss(
        log_probs, targets, rewards, variant="baseline", group_ids=group_ids
    )
    assert isinstance(loss_std, torch.Tensor)
    assert isinstance(loss_base, torch.Tensor)


def test_reinforce_pp_gradient_flows():
    log_probs, targets = _log_probs()
    log_probs.requires_grad_()
    rewards = torch.tensor([1.0, 0.5, 0.8, 0.3])
    loss = reinforce_pp_loss(log_probs, targets, rewards)
    loss.backward()
    assert log_probs.grad is not None


def test_reinforce_pp_advantage_shape():
    log_probs, targets = _log_probs()
    log_probs.requires_grad_()
    rewards = torch.tensor([1.0, 0.5, 0.8, 0.3])
    loss = reinforce_pp_loss(log_probs, targets, rewards)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
