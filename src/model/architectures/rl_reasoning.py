"""RL/Reasoning: DQN, AlphaGo/Zero, AlphaZero, PPO, DeepSeek-R1.

Papers: Mnih 2015, Silver 2016, Silver 2017, Schulman 2017, DeepSeek 2025.
"""

from __future__ import annotations

import math
import random
from typing import Any

from .foundational import MLP
from .registry import register


class DQN:
    """Deep Q-Network (Mnih et al. 2015)."""

    def __init__(self, state_dim: int = 4, action_dim: int = 2, gamma: float = 0.99) -> None:
        self.q_network = MLP([state_dim, 128, action_dim])
        self.target_network = MLP([state_dim, 128, action_dim])
        self.gamma = gamma

    def select_action(self, state: list[float], epsilon: float = 0.1) -> int:
        if random.random() < epsilon:
            return random.randint(0, len(self.q_network.forward(state)) - 1)
        q_values = self.q_network.forward(state)
        return max(range(len(q_values)), key=lambda i: q_values[i])

    def train_step(
        self, state: list[float], action: int, reward: float, next_state: list[float], done: bool
    ) -> float:
        q = self.q_network.forward(state)
        q_next = self.target_network.forward(next_state)
        target = reward + (0.0 if done else self.gamma * max(q_next))
        td_error = target - q[action]
        return td_error**2


register("rl.dqn", DQN)


class MCTSNode:
    """Monte Carlo Tree Search node (used in AlphaGo/AlphaZero)."""

    def __init__(self, state: Any = None, parent: Any = None, prior: float = 1.0) -> None:
        self.state = state
        self.parent = parent
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, MCTSNode] = {}

    def value(self) -> float:
        return self.value_sum / max(self.visit_count, 1)

    def ucb_score(self, c_puct: float = 1.4) -> float:
        return self.value() + c_puct * self.prior * math.sqrt(
            max(self.parent.visit_count, 1)
        ) / max(self.visit_count, 1)


class AlphaZero:
    """AlphaZero (Silver et al. 2017). MCTS + neural network."""

    def __init__(self, network: Any = None, n_simulations: int = 100) -> None:
        self.network = network or MLP([9 * 9, 128, 9 * 9 + 1])  # simplified for 9x9 Go
        self.n_simulations = n_simulations

    def mcts_search(self, root_state: Any) -> dict[int, float]:
        root = MCTSNode(state=root_state)
        for _ in range(self.n_simulations):
            node = root
            # Select
            while node.children:
                node = max(node.children.values(), key=lambda n: n.ucb_score())
            # Expand & Evaluate
            policy_logits = self.network.forward(
                node.state if hasattr(node.state, "__iter__") else [0.0] * 81
            )
            probs = [math.exp(p - max(policy_logits)) for p in policy_logits]
            total = sum(probs)
            probs = [p / total for p in probs]
            value = probs[-1]  # value head
            for action, prob in enumerate(probs[:-1]):
                if prob > 0.01:
                    node.children[action] = MCTSNode(state=None, parent=node, prior=prob)
            # Back up
            while node:
                node.visit_count += 1
                node.value_sum += value
                node = node.parent
        return {a: c.visit_count for a, c in root.children.items()}


register("rl.alphazero", AlphaZero)


class PPO:
    """Proximal Policy Optimization (Schulman et al. 2017)."""

    def __init__(self, state_dim: int = 4, action_dim: int = 2, clip_epsilon: float = 0.2) -> None:
        self.actor = MLP([state_dim, 64, action_dim])
        self.critic = MLP([state_dim, 64, 1])
        self.clip_epsilon = clip_epsilon

    def get_action_probs(self, state: list[float]) -> list[float]:
        logits = self.actor.forward(state)
        exp_l = [math.exp(v) for v in logits]
        total = sum(exp_l)
        return [e / total for e in exp_l]

    def compute_loss(
        self,
        states: list[list[float]],
        actions: list[int],
        old_probs: list[float],
        rewards: list[float],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> float:
        returns: list[float] = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + gamma * (0.0 if t == len(rewards) - 1 else self.critic.forward(states[t + 1])[0])
                - self.critic.forward(states[t])[0]
            )
            gae = delta + gamma * lam * gae
            returns.insert(0, gae)
        total_loss = 0.0
        for s, a, old_p, ret in zip(states, actions, old_probs, returns, strict=True):
            probs = self.get_action_probs(s)
            ratio = probs[a] / (old_p + 1e-8)
            clipped = max(min(ratio, 1.0 + self.clip_epsilon), 1.0 - self.clip_epsilon)
            total_loss -= min(ratio * ret, clipped * ret)
        return total_loss / max(len(states), 1)


register("rl.ppo", PPO)


class DeepSeekR1:
    """DeepSeek-R1: RL-driven reasoning (DeepSeek 2025)."""

    def __init__(self) -> None:
        self.chain_of_thought: list[str] = []
        self.verification_scores: list[float] = []

    def reason(self, problem: str, n_attempts: int = 3) -> tuple[str, float]:
        best_answer, best_score = "", -float("inf")
        for _ in range(n_attempts):
            thought = self._generate_cot(problem)
            answer = self._extract_answer(thought)
            score = self._verify(answer, problem)
            self.chain_of_thought.append(thought)
            self.verification_scores.append(score)
            if score > best_score:
                best_answer, best_score = answer, score
        return best_answer, best_score

    def _generate_cot(self, problem: str) -> str:
        return f"Step-by-step reasoning for: {problem}"

    def _extract_answer(self, thought: str) -> str:
        return thought.split(":")[-1].strip() if ":" in thought else thought

    def _verify(self, answer: str, problem: str) -> float:
        return random.uniform(0.0, 1.0)


register("rl.deepseek_r1", DeepSeekR1)
