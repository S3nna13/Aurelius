"""MCTS reasoner for step-level reasoning.

Inspired by DeepSeek-R1 (arXiv:2501.12599) and STILL-3 (Xu et al. 2411.11984);
clean-room reimplementation. License: MIT.
"""
from __future__ import annotations
import math
import uuid
from dataclasses import dataclass, field

_MAX_STATE_LEN = 32768

@dataclass
class MCTSNode:
    state: str
    parent_id: str | None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    prior: float = 1.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    def __post_init__(self) -> None:
        self.state = self.state[:_MAX_STATE_LEN]

    @property
    def value(self) -> float:
        return self.total_value / self.visits if self.visits > 0 else 0.0

    def ucb1(self, c: float = 1.414, parent_visits: int = 1) -> float:
        return self.value + c * math.sqrt(math.log(max(parent_visits, 1)) / (self.visits + 1))


class MCTSReasoner:
    def __init__(self, c_puct: float = 1.414, max_depth: int = 8,
                 max_simulations: int = 64, value_floor: float = -1.0,
                 value_ceiling: float = 1.0) -> None:
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.max_simulations = max_simulations
        self.value_floor = value_floor
        self.value_ceiling = value_ceiling

    def create_root(self, initial_state: str) -> MCTSNode:
        return MCTSNode(state=initial_state, parent_id=None)

    def expand(self, node: MCTSNode, children_states: list[str],
               priors: list[float] | None = None) -> list[MCTSNode]:
        if priors is not None and len(priors) != len(children_states):
            raise ValueError("priors length must match children_states length")
        new_nodes = []
        for i, state in enumerate(children_states):
            p = priors[i] if priors is not None else 1.0
            if not (0.0 < p <= 1.0):
                raise ValueError(f"prior must be in (0, 1], got {p}")
            child = MCTSNode(state=state, parent_id=node.id, prior=p)
            node.children.append(child)
            new_nodes.append(child)
        return new_nodes

    def backup(self, node: MCTSNode, value: float) -> None:
        if not (self.value_floor <= value <= self.value_ceiling):
            raise ValueError(f"value {value} out of range [{self.value_floor}, {self.value_ceiling}]")
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += value
            # walk up: find parent in tree (caller must pass the actual node chain)
            break  # single-node backup; full tree backup requires passing the path

    def backup_path(self, path: list[MCTSNode], value: float) -> None:
        """Backpropagate along a path from leaf to root."""
        if not (self.value_floor <= value <= self.value_ceiling):
            raise ValueError(f"value {value} out of range [{self.value_floor}, {self.value_ceiling}]")
        for node in reversed(path):
            node.visits += 1
            node.total_value += value

    def select_child(self, node: MCTSNode) -> MCTSNode:
        if not node.children:
            raise ValueError("node has no children")
        return max(node.children, key=lambda c: c.ucb1(self.c_puct, node.visits))

    def best_child(self, node: MCTSNode) -> MCTSNode:
        if not node.children:
            raise ValueError("node has no children")
        return max(node.children, key=lambda c: c.visits)

    def best_path(self, root: MCTSNode) -> list[MCTSNode]:
        path = [root]
        current = root
        while current.children:
            current = self.best_child(current)
            path.append(current)
        return path

    def rollout_path(self, root: MCTSNode, budget: int | None = None) -> list[MCTSNode]:
        if budget is None:
            budget = self.max_simulations
        if budget <= 0:
            raise ValueError(f"budget must be > 0, got {budget}")
        path = [root]
        current = root
        steps = 0
        while steps < budget and len(path) < self.max_depth:
            if not current.children:
                break
            current = self.select_child(current)
            path.append(current)
            steps += 1
        return path


MCTS_REASONER = MCTSReasoner()
