"""Draft-tree helpers for speculative decoding search.

Implements equal-growth tree structures (inspired by Yggdrasil, NeurIPS 2025)
for tree-based speculative decoding. Unlike standard k-ary trees which grow
exponentially (1 + k + k² + k³), equal-growth trees have ~equal nodes per level,
keeping the tree within a fixed budget for static graph compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DraftNode:
    """A single node in a draft tree.

    Attributes:
        token_id: The token at this node.
        score: Log-probability or confidence score from the draft model.
        parent: Index of the parent node in the flat node list (None for root).
    """

    token_id: int
    score: float
    parent: int | None = None


def equal_growth_parent_indices(
    total_budget: int,
    nodes_per_level: int = 4,
) -> list[int | None]:
    """Build an equal-growth tree where each level has ``nodes_per_level`` nodes.

    Yggdrasil (NeurIPS 2025) uses this structure for static-graph-compatible
    speculative decoding. Unlike full k-ary trees which grow exponentially,
    the equal-growth tree has approximately the same fan-out at every level,
    capped by a total node budget.

    Args:
        total_budget: Maximum total nodes including root. The tree stops growing
            when this budget would be exceeded.
        nodes_per_level: How many new nodes to add per level (Yggdrasil's
            "equal-growth" parameter). Default 4.

    Returns:
        Parent index for each node in flat list. ``None`` for root (index 0).

    Example:
        >>> equal_growth_parent_indices(13, nodes_per_level=4)
        [None, 0, 1, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7]
        # Level 0: root [0]
        # Level 1: nodes [1]          (child of 0)
        # Level 2: nodes [2, 3]       (children of 1)
        # Level 3: nodes [4, 5, 6, 7] (children of 2, 3)
        # Level 4: nodes [8, 9, 10, 11, 12] (children of 4, 5, 6, 7) - truncated at budget
    """
    if total_budget < 1:
        raise ValueError("total_budget must be >= 1")
    if nodes_per_level < 1:
        raise ValueError("nodes_per_level must be >= 1")

    parents: list[int | None] = [None]  # root
    frontier = [0]
    next_index = 1

    while len(parents) < total_budget:
        new_frontier = []
        # Distribute the nodes_per_level across the current frontier
        for i, parent in enumerate(frontier):
            if len(parents) >= total_budget:
                break
            # How many children for this parent?
            remaining_budget = total_budget - len(parents)
            remaining_frontier = len(frontier) - i
            children_here = max(1, remaining_budget // remaining_frontier)
            children_here = min(children_here, nodes_per_level)
            children_here = min(children_here, remaining_budget)

            for _ in range(children_here):
                parents.append(parent)
                new_frontier.append(next_index)
                next_index += 1
                if len(parents) >= total_budget:
                    break

        if not new_frontier:
            break
        frontier = new_frontier

    return parents


def parent_indices(branching_factor: int, depth: int) -> list[int | None]:
    """Return parent indices for a full k-ary tree up to depth.

    Provided for backward compatibility. Prefer ``equal_growth_parent_indices``
    for new code.

    Args:
        branching_factor: Number of children per node.
        depth: How many layers below the root.

    Returns:
        Parent index per node (``None`` for root).
    """
    if branching_factor <= 0 or depth < 0:
        raise ValueError("branching_factor must be positive and depth non-negative")
    parents: list[int | None] = [None]
    frontier = [0]
    next_index = 1
    for _ in range(depth):
        new_frontier = []
        for parent in frontier:
            for _ in range(branching_factor):
                parents.append(parent)
                new_frontier.append(next_index)
                next_index += 1
        frontier = new_frontier
    return parents


def root_to_leaf_paths(nodes: list[DraftNode]) -> list[list[int]]:
    """Enumerate token-id paths from root to all leaves.

    Args:
        nodes: Flat list of DraftNodes (root at index 0).

    Returns:
        List of token-id sequences, one per leaf node.
    """
    children: dict[int, list[int]] = {}
    for index, node in enumerate(nodes):
        if node.parent is not None:
            children.setdefault(node.parent, []).append(index)

    paths: list[list[int]] = []

    def visit(index: int, prefix: list[int]) -> None:
        node = nodes[index]
        new_prefix = prefix + [node.token_id]
        if index not in children:
            paths.append(new_prefix)
            return
        for child in children[index]:
            visit(child, new_prefix)

    if nodes:
        visit(0, [])
    return paths


def best_leaf_path(nodes: list[DraftNode]) -> list[int]:
    """Return the highest-scoring root-to-leaf path by summed node scores.

    Args:
        nodes: Flat list of DraftNodes (root at index 0).

    Returns:
        Token-id sequence along the best path.
    """
    if not nodes:
        return []

    # Build child index list for efficient traversal
    children: list[list[int]] = [[] for _ in range(len(nodes))]
    for idx, node in enumerate(nodes):
        if node.parent is not None:
            children[node.parent].append(idx)

    best_score = float("-inf")
    best_path: list[int] = []

    def traverse(node_idx: int, path: list[int], acc_score: float) -> None:
        nonlocal best_score, best_path
        node = nodes[node_idx]
        new_path = path + [node.token_id]
        new_score = acc_score + node.score
        child_list = children[node_idx]
        if not child_list:
            # Leaf — check if this is the best path so far
            if new_score > best_score:
                best_score = new_score
                best_path = new_path
        else:
            for child in child_list:
                traverse(child, new_path, new_score)

    traverse(0, [], 0.0)
    return best_path


def tree_causal_mask(
    parents: list[int | None],
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Build a causal attention mask for tree-structured draft verification.

    In tree-based speculative decoding, the target model verifies all draft
    paths in one forward pass. The standard causal mask (token i attends to
    tokens 0..i) is incorrect because tree nodes at the same depth may not
    all be causally related.

    Yggdrasil insight: the mask should allow token i to attend to all tokens
    on the path from root to i (its ancestors), plus any sibling tokens that
    share the same ancestor path (they were generated in parallel).

    Args:
        parents: Parent index per node (None for root).
        device: Target device for the mask tensor.

    Returns:
        Boolean mask of shape (N, N) where mask[i, j] is True if token j is
        allowed to attend to token i.
    """
    N = len(parents)
    # Compute ancestors for each node
    ancestors: list[set[int]] = [set() for _ in range(N)]
    for i in range(N):
        p = parents[i]
        while p is not None:
            ancestors[i].add(p)
            p = parents[p]

    mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(N):
        # Each token attends to itself, its ancestors, and any node
        # that shares its full ancestor chain (tree siblings)
        for j in range(N):
            if j == i or j in ancestors[i]:
                mask[i, j] = True
            elif i in ancestors[j]:
                # Ancestor attends to descendant
                mask[i, j] = True
            elif parents[i] is not None and parents[j] is not None and parents[i] == parents[j]:
                # Siblings attend to each other
                mask[i, j] = True

    return mask


def tokens_from_draft(
    nodes: list[DraftNode],
    top_k: int = 5,
) -> tuple[list[int], list[int | None]]:
    """Flatten a draft tree into a token sequence + parent indices for verification.

    Args:
        nodes: Draft tree nodes in breadth-first order (root first).
        top_k: Max children per node.

    Returns:
        Tuple of (token_ids, parent_indices) suitable for building an
        attention mask and verification input.
    """
    token_ids = [n.token_id for n in nodes]
    parents = [n.parent for n in nodes]
    return token_ids, parents
