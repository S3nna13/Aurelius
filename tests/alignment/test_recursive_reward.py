"""Tests for recursive_reward.py — hierarchical reward decomposition."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.alignment.recursive_reward import (
    RecursiveRMConfig,
    RewardNode,
    RecursiveRewardTree,
    TaskDecomposer,
    build_recursive_rm,
    evaluate_with_uncertainty,
)

# ---------------------------------------------------------------------------
# Shared mock
# ---------------------------------------------------------------------------

BATCH = 4
SEQ_LEN = 16


class MockRewardModel(nn.Module):
    def __init__(self, output_val: float = 1.0):
        super().__init__()
        self.output_val = output_val
        self.linear = nn.Linear(4, 1)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.full((input_ids.shape[0],), self.output_val)


def make_input(batch: int = BATCH, seq: int = SEQ_LEN) -> torch.Tensor:
    return torch.randint(0, 100, (batch, seq))


# ---------------------------------------------------------------------------
# Test 1: RecursiveRMConfig defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = RecursiveRMConfig()
    assert cfg.max_depth == 3
    assert cfg.n_subtasks == 3
    assert cfg.aggregation == "mean"
    assert cfg.use_weighted is False


# ---------------------------------------------------------------------------
# Test 2: RewardNode.evaluate returns (batch,) tensor
# ---------------------------------------------------------------------------

def test_reward_node_evaluate_shape():
    model = MockRewardModel(output_val=2.5)
    node = RewardNode(reward_model=model)
    input_ids = make_input()
    reward = node.evaluate(input_ids)
    assert reward.shape == (BATCH,), f"Expected ({BATCH},), got {reward.shape}"
    assert torch.allclose(reward, torch.full((BATCH,), 2.5))


# ---------------------------------------------------------------------------
# Test 3: RewardNode.get_depth returns correct depth
# ---------------------------------------------------------------------------

def test_reward_node_get_depth():
    # Single leaf node — depth 0
    leaf = RewardNode(reward_model=MockRewardModel(), depth=0)
    assert leaf.get_depth() == 0

    # A node with two leaf children — depth 1
    child_a = RewardNode(reward_model=MockRewardModel(), depth=1)
    child_b = RewardNode(reward_model=MockRewardModel(), depth=1)
    parent = RewardNode(
        reward_model=MockRewardModel(), depth=0, children=[child_a, child_b]
    )
    assert parent.get_depth() == 1

    # A node with a child that itself has a child — depth 2
    grandchild = RewardNode(reward_model=MockRewardModel(), depth=2)
    child = RewardNode(
        reward_model=MockRewardModel(), depth=1, children=[grandchild]
    )
    root = RewardNode(reward_model=MockRewardModel(), depth=0, children=[child])
    assert root.get_depth() == 2


# ---------------------------------------------------------------------------
# Test 4: RecursiveRewardTree.add_node returns valid id
# ---------------------------------------------------------------------------

def test_add_node_returns_valid_id():
    tree = RecursiveRewardTree()
    node = RewardNode(reward_model=MockRewardModel())
    nid = tree.add_node(node)
    assert isinstance(nid, int)
    assert nid >= 0
    assert nid in tree._nodes


# ---------------------------------------------------------------------------
# Test 5: Tree with 2 leaf nodes evaluates correctly (mean of both)
# ---------------------------------------------------------------------------

def test_tree_two_leaves_mean():
    tree = RecursiveRewardTree(aggregation="mean")
    # Add two independent root leaves (no shared parent)
    n0 = tree.add_node(RewardNode(MockRewardModel(output_val=2.0)))
    n1 = tree.add_node(RewardNode(MockRewardModel(output_val=4.0)))
    input_ids = make_input()
    result = tree.evaluate(input_ids)
    assert result.shape == (BATCH,)
    expected = torch.full((BATCH,), 3.0)
    assert torch.allclose(result, expected), f"Expected 3.0, got {result}"


# ---------------------------------------------------------------------------
# Test 6: aggregate 'mean' returns mean of rewards
# ---------------------------------------------------------------------------

def test_aggregate_mean():
    tree = RecursiveRewardTree()
    r1 = torch.tensor([1.0, 2.0, 3.0])
    r2 = torch.tensor([3.0, 4.0, 5.0])
    result = tree.aggregate([r1, r2], "mean")
    expected = torch.tensor([2.0, 3.0, 4.0])
    assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# Test 7: aggregate 'min' returns minimum reward
# ---------------------------------------------------------------------------

def test_aggregate_min():
    tree = RecursiveRewardTree()
    r1 = torch.tensor([1.0, 5.0, 3.0])
    r2 = torch.tensor([4.0, 2.0, 6.0])
    result = tree.aggregate([r1, r2], "min")
    expected = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# Test 8: aggregate 'product' returns product of rewards
# ---------------------------------------------------------------------------

def test_aggregate_product():
    tree = RecursiveRewardTree()
    r1 = torch.tensor([2.0, 3.0])
    r2 = torch.tensor([4.0, 5.0])
    result = tree.aggregate([r1, r2], "product")
    expected = torch.tensor([8.0, 15.0])
    assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# Test 9: TaskDecomposer.decompose returns n_subtasks chunks
# ---------------------------------------------------------------------------

def test_task_decomposer_decompose():
    n_subtasks = 3
    decomposer = TaskDecomposer(decompose_fn=lambda x: x, n_subtasks=n_subtasks)
    input_ids = make_input(batch=2, seq=12)
    chunks = decomposer.decompose(input_ids, max_subtask_len=64)
    assert len(chunks) == n_subtasks
    for chunk in chunks:
        assert chunk.shape[0] == 2, "Batch dimension should be preserved"
        assert chunk.ndim == 2


# ---------------------------------------------------------------------------
# Test 10: compose_rewards with equal weights == mean
# ---------------------------------------------------------------------------

def test_compose_rewards_equal_weights():
    decomposer = TaskDecomposer(decompose_fn=lambda x: x, n_subtasks=3)
    r0 = torch.tensor([1.0, 2.0])
    r1 = torch.tensor([3.0, 4.0])
    r2 = torch.tensor([5.0, 6.0])

    # No explicit weights → uniform
    result_uniform = decomposer.compose_rewards([r0, r1, r2])
    expected = torch.tensor([3.0, 4.0])
    assert torch.allclose(result_uniform, expected)

    # Explicit equal weights
    weights = torch.ones(3) / 3
    result_explicit = decomposer.compose_rewards([r0, r1, r2], weights=weights)
    assert torch.allclose(result_explicit, expected)


# ---------------------------------------------------------------------------
# Test 11: build_recursive_rm creates tree with correct leaf count
# ---------------------------------------------------------------------------

def test_build_recursive_rm_leaf_count():
    branching = 2
    depth = 3
    # A complete tree has branching^depth leaves
    expected_leaves = branching ** depth

    tree = build_recursive_rm(
        reward_model_factory=lambda: MockRewardModel(),
        depth=depth,
        branching=branching,
    )
    leaf_ids = tree._leaf_ids()
    assert len(leaf_ids) == expected_leaves, (
        f"Expected {expected_leaves} leaves, got {len(leaf_ids)}"
    )


# ---------------------------------------------------------------------------
# Test 12: evaluate_with_uncertainty returns (mean, std) with correct shapes
# ---------------------------------------------------------------------------

def test_evaluate_with_uncertainty_shapes():
    # Use a model with dropout so samples actually vary
    class DropoutRewardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(p=0.5)
            self.proj = nn.Linear(SEQ_LEN, 1)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = input_ids.float()
            x = self.drop(x)
            return self.proj(x).squeeze(-1)

    tree = build_recursive_rm(
        reward_model_factory=DropoutRewardModel,
        depth=1,
        branching=2,
    )
    input_ids = make_input()
    mean_r, std_r = evaluate_with_uncertainty(tree, input_ids, n_samples=10)

    assert mean_r.shape == (BATCH,), f"mean shape wrong: {mean_r.shape}"
    assert std_r.shape == (BATCH,), f"std shape wrong: {std_r.shape}"
    # std should be non-negative
    assert (std_r >= 0).all()
