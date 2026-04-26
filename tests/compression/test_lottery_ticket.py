"""Tests for lottery_ticket — Lottery Ticket Hypothesis pruning."""

from __future__ import annotations

import torch

from src.compression.lottery_ticket import LotteryTicketPruner


class TestLotteryTicketPruner:
    def test_prune_removes_smallest_magnitude_weights(self):
        module = torch.nn.Linear(10, 10)
        before = module.weight.data.clone()
        pruner = LotteryTicketPruner(prune_fraction=0.5)
        pruner.prune(module)
        pruned = module.weight.data
        nonzero_before = before.numel()
        nonzero_after = pruned.nonzero().size(0)
        assert nonzero_after < nonzero_before

    def test_prune_fraction_zero_no_change(self):
        module = torch.nn.Linear(10, 10)
        before = module.weight.data.clone()
        pruner = LotteryTicketPruner(prune_fraction=0.0)
        pruner.prune(module)
        assert torch.equal(module.weight.data, before)

    def test_prune_fraction_one_zeros_all(self):
        module = torch.nn.Linear(10, 10)
        pruner = LotteryTicketPruner(prune_fraction=1.0)
        pruner.prune(module)
        assert module.weight.data.nonzero().size(0) == 0

    def test_rewind_restores_weights(self):
        module = torch.nn.Linear(10, 10)
        initial = module.weight.data.clone()
        pruner = LotteryTicketPruner(prune_fraction=0.3)
        pruner.prune(module)
        pruner.rewind(module)
        assert torch.equal(module.weight.data, initial)

    def test_mask_persistence(self):
        module = torch.nn.Linear(10, 10)
        pruner = LotteryTicketPruner(prune_fraction=0.3)
        pruner.prune(module)
        mask = pruner.mask
        assert mask is not None
        assert mask.shape == module.weight.shape
        assert mask.dtype == torch.bool

    def test_train_rewind_prune_cycle(self):
        module = torch.nn.Linear(5, 3)
        initial = module.weight.data.clone()
        pruner = LotteryTicketPruner(prune_fraction=0.3)
        # First prune + train simulation
        pruner.prune(module)
        with torch.no_grad():
            module.weight.add_(torch.randn_like(module.weight) * 0.01)
        pruner.rewind(module)
        assert torch.equal(module.weight.data, initial)
        # Second prune with same mask
        pruner.prune(module)
        assert (module.weight.data == 0).sum() > 0
