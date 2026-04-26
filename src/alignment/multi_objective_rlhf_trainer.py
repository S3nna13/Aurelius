from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ObjectiveWeight:
    name: str
    weight: float


@dataclass
class MOConfig:
    objectives: list[ObjectiveWeight]
    epsilon: float = 0.2
    entropy_coef: float = 0.01
    normalize_rewards: bool = True


@dataclass
class MOTrainResult:
    total_loss: float
    per_objective_losses: dict[str, float]
    weighted_reward: float


class MultiObjectiveRLHFTrainer:
    """Multi-objective RLHF via linear scalarization of reward heads."""

    def __init__(self, policy: nn.Module, config: MOConfig) -> None:
        self.policy = policy
        self.config = config

    def scalarize_rewards(self, reward_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        total: torch.Tensor | None = None
        for obj in self.config.objectives:
            r = reward_dict[obj.name]
            weighted = obj.weight * r
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("No objectives configured")
        return total

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        return rewards - values

    def ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        ratio = torch.exp(log_probs - old_log_probs)
        eps = self.config.epsilon
        clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        surrogate = torch.min(ratio * advantages, clipped * advantages)
        return -surrogate.mean()

    def train_step(
        self,
        reward_dict: dict[str, torch.Tensor],
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        values: torch.Tensor,
    ) -> MOTrainResult:
        if self.config.normalize_rewards:
            normed: dict[str, torch.Tensor] = {}
            for k, v in reward_dict.items():
                std = v.std()
                normed[k] = (v - v.mean()) / (std + 1e-8)
        else:
            normed = reward_dict

        scalar_rewards = self.scalarize_rewards(normed)
        advantages = self.compute_advantages(scalar_rewards, values)
        policy_loss = self.ppo_loss(log_probs, old_log_probs, advantages)

        entropy_bonus = -(log_probs * torch.exp(log_probs)).mean()
        total_loss = policy_loss - self.config.entropy_coef * entropy_bonus

        per_obj: dict[str, float] = {}
        for obj in self.config.objectives:
            r = normed[obj.name]
            adv = self.compute_advantages(r, values)
            per_obj[obj.name] = self.ppo_loss(log_probs, old_log_probs, adv).item()

        return MOTrainResult(
            total_loss=total_loss.item(),
            per_objective_losses=per_obj,
            weighted_reward=scalar_rewards.mean().item(),
        )

    def pareto_check(self, results: list[MOTrainResult]) -> list[bool]:
        n = len(results)
        is_pareto: list[bool] = [True] * n

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                losses_i = results[i].per_objective_losses
                losses_j = results[j].per_objective_losses
                keys = list(losses_i.keys())
                # j dominates i if j is better (lower loss) on ALL objectives
                if all(losses_j[k] <= losses_i[k] for k in keys) and any(
                    losses_j[k] < losses_i[k] for k in keys
                ):
                    is_pareto[i] = False
                    break

        return is_pareto
