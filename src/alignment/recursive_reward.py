"""Recursive Reward Modeling (RRM) — hierarchical reward decomposition for scalable oversight.

Based on Leike et al. (2018) and Irving et al.: decompose hard-to-evaluate tasks into subtasks
that are easier for humans to evaluate, then compose reward signals bottom-up through a tree.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RecursiveRMConfig:
    """Configuration for recursive reward modeling."""

    max_depth: int = 3
    n_subtasks: int = 3
    aggregation: str = "mean"
    use_weighted: bool = False


# ---------------------------------------------------------------------------
# RewardNode
# ---------------------------------------------------------------------------


class RewardNode:
    """A single node in the recursive reward tree.

    Leaf nodes hold a reward model that scores inputs directly.
    Internal nodes aggregate scores from their children.
    """

    def __init__(
        self,
        reward_model: nn.Module,
        depth: int = 0,
        children: list[RewardNode] | None = None,
    ) -> None:
        self.reward_model = reward_model
        self.depth = depth
        self.children: list[RewardNode] = children if children is not None else []

    def evaluate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Evaluate this node's reward model on input_ids.

        Args:
            input_ids: (batch, seq_len) integer token ids.

        Returns:
            Scalar reward tensor of shape (batch,).
        """
        self.reward_model.eval()
        with torch.no_grad():
            reward = self.reward_model(input_ids)
        return reward

    def get_depth(self) -> int:
        """Return the depth of the subtree rooted at this node (0 = leaf)."""
        if not self.children:
            return 0
        return 1 + max(child.get_depth() for child in self.children)


# ---------------------------------------------------------------------------
# RecursiveRewardTree
# ---------------------------------------------------------------------------


class RecursiveRewardTree:
    """A tree of RewardNodes evaluated bottom-up with configurable aggregation.

    Nodes are stored in a flat dict keyed by integer node_id.  The parent
    relationship is tracked so that bottom-up aggregation can be performed
    without recursion.
    """

    def __init__(self, max_depth: int = 3, aggregation: str = "mean") -> None:
        if aggregation not in ("mean", "min", "product", "weighted"):
            raise ValueError(f"Unknown aggregation '{aggregation}'")
        self.max_depth = max_depth
        self.aggregation = aggregation

        self._nodes: dict[int, RewardNode] = {}
        self._parent: dict[int, int | None] = {}
        self._children: dict[int, list[int]] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Tree construction
    # ------------------------------------------------------------------

    def add_node(self, node: RewardNode, parent_id: int | None = None) -> int:
        """Add a node to the tree and return its assigned node_id.

        Args:
            node:      The RewardNode to insert.
            parent_id: ID of the parent node, or None for the root.

        Returns:
            The integer node_id assigned to this node.
        """
        node_id = self._next_id
        self._next_id += 1

        self._nodes[node_id] = node
        self._parent[node_id] = parent_id
        self._children[node_id] = []

        if parent_id is not None:
            if parent_id not in self._nodes:
                raise ValueError(f"parent_id {parent_id} not found in tree")
            self._children[parent_id].append(node_id)

        return node_id

    # ------------------------------------------------------------------
    # Leaf detection
    # ------------------------------------------------------------------

    def _leaf_ids(self) -> list[int]:
        return [nid for nid, children in self._children.items() if not children]

    # ------------------------------------------------------------------
    # Bottom-up evaluation
    # ------------------------------------------------------------------

    def evaluate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Evaluate the tree bottom-up and return an aggregated scalar reward.

        Args:
            input_ids: (batch, seq_len) token ids.

        Returns:
            Aggregated reward of shape (batch,).
        """
        if not self._nodes:
            raise RuntimeError("Tree is empty — add at least one node before evaluating")

        cache: dict[int, torch.Tensor] = {}

        def _score(nid: int) -> torch.Tensor:
            if nid in cache:
                return cache[nid]
            children = self._children[nid]
            if not children:
                result = self._nodes[nid].evaluate(input_ids)
            else:
                child_rewards = [_score(c) for c in children]
                result = self.aggregate(child_rewards, self.aggregation)
            cache[nid] = result
            return result

        # Find root (node with no parent)
        roots = [nid for nid, p in self._parent.items() if p is None]
        if len(roots) == 1:
            return _score(roots[0])

        # Multiple roots: aggregate all root rewards
        root_rewards = [_score(r) for r in roots]
        return self.aggregate(root_rewards, self.aggregation)

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self, rewards: list[torch.Tensor], method: str) -> torch.Tensor:
        """Aggregate a list of (batch,) reward tensors into a single (batch,) tensor.

        Args:
            rewards: List of tensors, each of shape (batch,).
            method:  One of 'mean', 'min', 'product', 'weighted'.

        Returns:
            Aggregated reward of shape (batch,).
        """
        if not rewards:
            raise ValueError("Cannot aggregate empty rewards list")

        stacked = torch.stack(rewards, dim=0)  # (n, batch)

        if method == "mean":
            return stacked.mean(dim=0)
        elif method == "min":
            return stacked.min(dim=0).values
        elif method == "product":
            result = rewards[0].clone()
            for r in rewards[1:]:
                result = result * r
            return result
        elif method == "weighted":
            # Equal weights by default when called from aggregate()
            n = stacked.shape[0]
            weights = torch.ones(n, device=stacked.device) / n
            return (stacked * weights.unsqueeze(1)).sum(dim=0)
        else:
            raise ValueError(f"Unknown aggregation method '{method}'")


# ---------------------------------------------------------------------------
# TaskDecomposer
# ---------------------------------------------------------------------------


class TaskDecomposer:
    """Decomposes an input sequence into subtasks and composes their rewards.

    Simple chunking strategy: split the sequence dimension evenly into
    n_subtasks chunks, padding the last chunk if necessary.
    """

    def __init__(self, decompose_fn: Callable, n_subtasks: int = 3) -> None:
        self.decompose_fn = decompose_fn
        self.n_subtasks = n_subtasks

    def decompose(self, input_ids: torch.Tensor, max_subtask_len: int = 64) -> list[torch.Tensor]:
        """Split input_ids into n_subtasks chunks along the sequence dimension.

        Args:
            input_ids:       (batch, seq_len) token ids.
            max_subtask_len: Maximum tokens per subtask chunk.

        Returns:
            List of n_subtasks tensors, each of shape (batch, chunk_len).
        """
        batch, seq_len = input_ids.shape

        # Compute chunk size, then cap at max_subtask_len
        chunk_size = max(1, (seq_len + self.n_subtasks - 1) // self.n_subtasks)
        chunk_size = min(chunk_size, max_subtask_len)

        chunks: list[torch.Tensor] = []
        for i in range(self.n_subtasks):
            start = i * chunk_size
            end = min(start + chunk_size, seq_len)
            if start >= seq_len:
                # Pad with zeros if the sequence is shorter than expected
                chunk = torch.zeros(
                    batch, chunk_size, dtype=input_ids.dtype, device=input_ids.device
                )
            else:
                chunk = input_ids[:, start:end]
                # Pad to uniform chunk_size if needed
                if chunk.shape[1] < chunk_size:
                    pad = torch.zeros(
                        batch,
                        chunk_size - chunk.shape[1],
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                    chunk = torch.cat([chunk, pad], dim=1)
            chunks.append(chunk)

        return chunks

    def compose_rewards(
        self,
        subtask_rewards: list[torch.Tensor],
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute a weighted average of subtask rewards.

        Args:
            subtask_rewards: List of (batch,) reward tensors.
            weights:         Optional 1-D weight tensor of length n_subtasks.
                             If None, uses uniform weights.

        Returns:
            Composed reward of shape (batch,).
        """
        n = len(subtask_rewards)
        stacked = torch.stack(subtask_rewards, dim=0)  # (n, batch)

        if weights is None:
            weights = torch.ones(n, device=stacked.device) / n
        else:
            weights = weights.to(stacked.device)
            weights = weights / weights.sum()  # normalise

        return (stacked * weights.unsqueeze(1)).sum(dim=0)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def build_recursive_rm(
    reward_model_factory: Callable[[], nn.Module],
    depth: int,
    branching: int,
) -> RecursiveRewardTree:
    """Build a complete tree of the given depth and branching factor.

    All leaf nodes are created by calling reward_model_factory().

    Args:
        reward_model_factory: Zero-arg callable returning a fresh nn.Module.
        depth:                Tree depth (0 = single root leaf).
        branching:            Number of children per internal node.

    Returns:
        A fully constructed RecursiveRewardTree.
    """
    tree = RecursiveRewardTree(max_depth=depth)

    def _build(parent_id: int | None, current_depth: int) -> None:
        node = RewardNode(reward_model=reward_model_factory(), depth=current_depth)
        nid = tree.add_node(node, parent_id=parent_id)
        if current_depth < depth:
            for _ in range(branching):
                _build(parent_id=nid, current_depth=current_depth + 1)

    _build(parent_id=None, current_depth=0)
    return tree


def evaluate_with_uncertainty(
    tree: RecursiveRewardTree,
    input_ids: torch.Tensor,
    n_samples: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate reward uncertainty via MC-Dropout.

    Puts all reward models into training mode (enabling dropout), runs
    n_samples forward passes, then returns the sample mean and std.

    Args:
        tree:       A RecursiveRewardTree whose reward models may contain Dropout.
        input_ids:  (batch, seq_len) token ids.
        n_samples:  Number of stochastic forward passes.

    Returns:
        (mean_reward, std_reward): both tensors of shape (batch,).
    """
    # Enable dropout (training mode) for all models in the tree
    for node in tree._nodes.values():
        node.reward_model.train()

    samples: list[torch.Tensor] = []
    for _ in range(n_samples):
        reward = _forward_train_mode(tree, input_ids)
        samples.append(reward)

    # Restore eval mode
    for node in tree._nodes.values():
        node.reward_model.eval()

    stacked = torch.stack(samples, dim=0)  # (n_samples, batch)
    mean_reward = stacked.mean(dim=0)
    std_reward = stacked.std(dim=0)
    return mean_reward, std_reward


def _forward_train_mode(tree: RecursiveRewardTree, input_ids: torch.Tensor) -> torch.Tensor:
    """Like RecursiveRewardTree.evaluate but keeps models in their current mode."""
    if not tree._nodes:
        raise RuntimeError("Tree is empty")

    cache: dict[int, torch.Tensor] = {}

    def _score(nid: int) -> torch.Tensor:
        if nid in cache:
            return cache[nid]
        children = tree._children[nid]
        if not children:
            result = tree._nodes[nid].reward_model(input_ids)
        else:
            child_rewards = [_score(c) for c in children]
            result = tree.aggregate(child_rewards, tree.aggregation)
        cache[nid] = result
        return result

    roots = [nid for nid, p in tree._parent.items() if p is None]
    if len(roots) == 1:
        return _score(roots[0])
    return tree.aggregate([_score(r) for r in roots], tree.aggregation)
