"""Draft-tree helpers for speculative decoding search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DraftNode:
    token_id: int
    score: float
    parent: int | None = None


def parent_indices(branching_factor: int, depth: int) -> list[int | None]:
    """Return parent indices for a full k-ary tree up to depth."""
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
    """Enumerate token-id paths from root to all leaves."""
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

    visit(0, [])
    return paths


def best_leaf_path(nodes: list[DraftNode]) -> list[int]:
    """Return the highest-scoring root-to-leaf path by summed node scores."""
    best_score = float("-inf")
    best_path: list[int] = []
    for path in root_to_leaf_paths(nodes):
        score = 0.0
        current_parent = None
        for token in path:
            for node in nodes:
                if node.token_id == token and node.parent == current_parent:
                    score += node.score
                    current_parent = nodes.index(node)
                    break
        if score > best_score:
            best_score = score
            best_path = path
    return best_path
