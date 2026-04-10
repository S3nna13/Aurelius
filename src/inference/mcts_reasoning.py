"""MCTS-based step-by-step reasoning for AureliusTransformer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class MCTSConfig:
    """Configuration for MCTS reasoning."""

    n_simulations: int = 50
    c_puct: float = 1.4
    max_depth: int = 10
    temperature: float = 1.0
    value_discount: float = 0.95


@dataclass
class MCTSNode:
    """A node in the MCTS search tree."""

    state: list[int]                          # token ids
    parent: "MCTSNode | None"
    children: list["MCTSNode"] = field(default_factory=list)
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 1.0
    action: int | None = None                  # token that led here

    def value(self) -> float:
        """Average value over visits."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        """UCB1 score balancing exploitation and exploration."""
        return (
            self.value()
            + c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        )

    def is_leaf(self) -> bool:
        """True when node has no children yet."""
        return len(self.children) == 0


def evaluate_state(model: torch.nn.Module, state: list[int]) -> float:
    """Evaluate how 'good' the current reasoning state is.

    Runs a model forward pass and computes the mean log-prob of the state
    tokens as a proxy value, then squashes to [-1, 1] via tanh.
    """
    if len(state) == 0:
        return 0.0

    device = next(model.parameters()).device
    input_ids = torch.tensor([state], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(input_ids)
        logits = output[1]  # (1, S, V)

    # Shift: predict each token from previous context
    # logits[:, :-1, :] predicts tokens state[1:]
    if logits.shape[1] < 2:
        # Single token — use max log-prob of next token as proxy
        log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
        proxy = log_probs.max().item()
    else:
        log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)  # (S-1, V)
        target_ids = torch.tensor(state[1:], dtype=torch.long, device=device)  # (S-1,)
        token_log_probs = log_probs[range(len(state) - 1), target_ids]  # (S-1,)
        proxy = token_log_probs.mean().item()

    return math.tanh(proxy)


def expand_node(
    model: torch.nn.Module, node: MCTSNode, top_k: int = 5
) -> None:
    """Expand a leaf node by creating children for the top-k next tokens.

    Priors are set to the softmax probability of each token.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor([node.state], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(input_ids)
        logits = output[1]  # (1, S, V)

    # Next-token distribution at last position
    last_logits = logits[0, -1, :]  # (V,)
    probs = F.softmax(last_logits, dim=-1)

    k = min(top_k, probs.shape[0])
    topk_probs, topk_indices = torch.topk(probs, k)

    for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
        child = MCTSNode(
            state=node.state + [idx],
            parent=node,
            prior=prob,
            action=idx,
        )
        node.children.append(child)


def backpropagate(
    node: MCTSNode, value: float, discount: float
) -> None:
    """Walk up the tree updating visit_count and value_sum with discounting."""
    current = node
    current_value = value
    while current is not None:
        current.visit_count += 1
        current.value_sum += current_value
        current_value *= discount
        current = current.parent


class MCTSReasoner:
    """Runs MCTS over token sequences to find the best reasoning continuation."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: MCTSConfig,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode

    def search(self, prompt: str) -> tuple[str, float]:
        """Run MCTS from prompt. Returns (best_continuation_text, best_value)."""
        prompt_ids = self.tokenizer_encode(prompt)
        root = MCTSNode(state=prompt_ids, parent=None)

        for _ in range(self.config.n_simulations):
            # Selection
            leaf = self._select(root)

            # Expansion — only if not at max depth
            depth = 0
            cur = leaf
            while cur.parent is not None:
                depth += 1
                cur = cur.parent

            if depth < self.config.max_depth:
                expand_node(self.model, leaf, top_k=5)
                if leaf.children:
                    leaf = leaf.children[0]

            # Simulation / evaluation
            value = self._simulate(leaf)

            # Backpropagation
            backpropagate(leaf, value, self.config.value_discount)

        best_path = self.get_best_path(root)
        continuation_ids = best_path[len(prompt_ids):]
        continuation_text = self.tokenizer_decode(continuation_ids)
        best_value = root.value()

        return continuation_text, best_value

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Traverse from root to a leaf using UCB scores."""
        current = node
        while not current.is_leaf():
            best_child = max(
                current.children,
                key=lambda c: c.ucb_score(current.visit_count, self.config.c_puct),
            )
            current = best_child
        return current

    def _simulate(self, node: MCTSNode) -> float:
        """Evaluate a leaf node using the model."""
        return evaluate_state(self.model, node.state)

    def get_best_path(self, root: MCTSNode) -> list[int]:
        """Return the token sequence of the highest-value path from root."""
        path = root.state[:]
        current = root
        while not current.is_leaf():
            best_child = max(current.children, key=lambda c: c.value())
            current = best_child
            # Append only the new action token
            if current.action is not None:
                path.append(current.action)
        return path
