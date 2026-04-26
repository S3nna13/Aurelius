"""Tests for structured_pruning — layer/head/filter level pruning."""

from __future__ import annotations

import torch

from src.compression.structured_pruning import PruningStrategy, StructuredPruner


class TestStructuredPruner:
    def test_prune_heads_removes_attention_heads(self):
        weight = torch.randn(8, 64)
        pruner = StructuredPruner(strategy=PruningStrategy.HEADS, amount=0.5)
        pruned = pruner.prune(weight, dim=0, groups=8, group_size=64)
        assert pruned.shape == weight.shape
        # Check entire groups are zeroed
        group_norms = pruned.view(8, 64).norm(dim=1)
        assert (group_norms == 0).sum() == 4

    def test_prune_neurons_removes_ffn_neurons(self):
        weight = torch.randn(128, 64)
        pruner = StructuredPruner(strategy=PruningStrategy.NEURONS, amount=0.25)
        pruned = pruner.prune(weight, dim=0)
        zero_rows = (pruned == 0).all(dim=1).sum()
        expected = int(128 * 0.25)
        assert zero_rows == expected

    def test_prune_channels_removes_filters(self):
        weight = torch.randn(16, 32, 3, 3)
        pruner = StructuredPruner(strategy=PruningStrategy.CHANNELS, amount=0.5)
        pruned = pruner.prune(weight, dim=0)
        zero_filters = (pruned == 0).all(dim=(1, 2, 3)).sum()
        assert zero_filters == 8

    def test_amount_zero_no_change(self):
        weight = torch.randn(10, 20)
        pruner = StructuredPruner(strategy=PruningStrategy.NEURONS, amount=0.0)
        pruned = pruner.prune(weight, dim=0)
        assert torch.equal(pruned, weight)

    def test_amount_one_zeros_all(self):
        weight = torch.randn(10, 20)
        pruner = StructuredPruner(strategy=PruningStrategy.NEURONS, amount=1.0)
        pruned = pruner.prune(weight, dim=0)
        assert (pruned == 0).all()

    def test_l1_norm_strategy(self):
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        pruner = StructuredPruner(strategy=PruningStrategy.NEURONS, amount=0.5)
        pruned = pruner.prune(weight, dim=0)
        # Row 0 (L1=3) should be pruned before row 1 (L1=7)
        assert (pruned[0] == 0).all()
        assert (pruned[1] != 0).any()

    def test_heads_strategy_non_divisible_groups(self):
        weight = torch.randn(7, 64)
        pruner = StructuredPruner(strategy=PruningStrategy.HEADS, amount=0.5)
        pruned = pruner.prune(weight, dim=0, groups=7, group_size=64)
        zero_norms = (pruned.view(7, 64).norm(dim=1) == 0).sum()
        expected = int(7 * 0.5)
        assert zero_norms == expected

    def test_invalid_strategy_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown pruning strategy"):
            StructuredPruner(strategy="invalid", amount=0.3)
