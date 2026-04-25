"""Tree-of-Thought planner: beam search over thought branches (Yao et al. 2305.10601)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field


@dataclass
class ThoughtNode:
    parent_id: str | None
    thought: str
    score: float = 0.0
    depth: int = 0
    children: list["ThoughtNode"] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self) -> None:
        # id is auto-generated via default_factory; nothing extra needed
        pass


class ToTPlanner:
    def __init__(self, beam_width: int = 3, max_depth: int = 5) -> None:
        self.beam_width = beam_width
        self.max_depth = max_depth

    def add_root(self, thought: str, score: float = 0.0) -> ThoughtNode:
        return ThoughtNode(parent_id=None, thought=thought, score=score, depth=0)

    def expand(
        self,
        parent: ThoughtNode,
        thoughts: list[str],
        scores: list[float],
    ) -> list[ThoughtNode]:
        new_nodes: list[ThoughtNode] = []
        for thought, score in zip(thoughts, scores):
            node = ThoughtNode(
                parent_id=parent.id,
                thought=thought,
                score=score,
                depth=parent.depth + 1,
            )
            parent.children.append(node)
            new_nodes.append(node)
        return new_nodes

    def beam_select(
        self, nodes: list[ThoughtNode], k: int | None = None
    ) -> list[ThoughtNode]:
        if k is None:
            k = self.beam_width
        return sorted(nodes, key=lambda n: n.score, reverse=True)[:k]

    def best_path(self, root: ThoughtNode) -> list[ThoughtNode]:
        path = [root]
        current = root
        while current.children:
            current = max(current.children, key=lambda n: n.score)
            path.append(current)
        return path

    def all_leaves(self, root: ThoughtNode) -> list[ThoughtNode]:
        if not root.children:
            return [root]
        leaves: list[ThoughtNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if not node.children:
                leaves.append(node)
            else:
                stack.extend(node.children)
        return leaves

    def tree_size(self, root: ThoughtNode) -> int:
        count = 1
        for child in root.children:
            count += self.tree_size(child)
        return count


TOT_PLANNER = ToTPlanner()
