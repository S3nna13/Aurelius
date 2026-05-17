"""AEM: Adaptive Entropy Modulation for Multi-Turn Agentic Reinforcement Learning.

Supervision-free credit assignment that modulates entropy dynamics to improve
exploration-exploitation trade-off. Response-level uncertainty proxy rescales advantages
based on positive/negative sample balance.

Paper: arXiv:2605.00425
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


class AdaptiveEntropyModulator:
    """Response-level uncertainty proxy for advantage rescaling."""

    def __init__(
        self,
        entropy_coef_init: float = 1.0,
        min_entropy_coef: float = 0.01,
        max_entropy_coef: float = 1.0,
    ) -> None:
        self.min_entropy_coef = min_entropy_coef
        self.max_entropy_coef = max_entropy_coef
        self.entropy_coef = max(
            self.min_entropy_coef,
            min(self.max_entropy_coef, entropy_coef_init),
        )
        self._step_count = 0

    def compute_response_entropy(self, logits: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(-1)
        return entropy

    def compute_entropy_drift(
        self,
        response_advantage: Tensor,
        response_surprisal: Tensor,
    ) -> float:
        drift = (response_advantage * response_surprisal).mean().item()
        return drift

    def update_coef(self, drift: float, lr: float = 0.01) -> None:
        self._step_count += 1
        updated = self.entropy_coef - lr * drift
        self.entropy_coef = max(
            self.min_entropy_coef,
            min(self.max_entropy_coef, updated),
        )

    def rescale_advantages(
        self,
        advantages: Tensor,
        response_advantage: Tensor,
        response_surprisal: Tensor,
    ) -> Tensor:
        balance = (response_advantage - response_advantage.mean()) / (
            response_advantage.std(unbiased=False).clamp(min=1e-8)
        )
        uncertainty_proxy = 1.0 / (response_surprisal.exp() + 1e-8)
        uncertainty_proxy = uncertainty_proxy * (1.0 + 0.1 * balance.tanh())
        rescaled = advantages * uncertainty_proxy
        return rescaled

    def forward(
        self,
        logits: Tensor,
        advantages: Tensor,
        policy_log_probs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict]:
        response_entropy = self.compute_response_entropy(logits)
        response_adv = response_entropy.detach()
        if policy_log_probs is not None:
            response_surprisal = -policy_log_probs
        else:
            response_surprisal = response_entropy

        rescaled_adv = self.rescale_advantages(advantages, response_adv, response_surprisal)

        entropy_loss = -self.entropy_coef * response_entropy.mean()
        metrics = {
            "response_entropy": response_entropy.mean().item(),
            "entropy_coef": self.entropy_coef,
            "uncertainty_proxy": (1.0 / (response_surprisal.exp() + 1e-8)).mean().item(),
        }
        return rescaled_adv, entropy_loss, metrics

    def __call__(
        self,
        logits: Tensor,
        advantages: Tensor,
        policy_log_probs: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict]:
        return self.forward(logits, advantages, policy_log_probs)


class AEMTrainer:
    def __init__(self, model, optimizer, config=None):
        self.model = model
        self.optimizer = optimizer
        self.config = config or {}
        self.modulator = AdaptiveEntropyModulator()

    def compute_loss(self, input_ids: Tensor, advantages: Tensor) -> tuple[Tensor, dict]:
        logits = self.model(input_ids).logits
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs[:, :-1]
        target_ids = input_ids[:, 1:]
        selected_log_probs = token_log_probs.gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        ).squeeze(-1)

        token_advantages = advantages
        if token_advantages.dim() == 1:
            token_advantages = token_advantages.unsqueeze(-1).expand_as(selected_log_probs)
        elif token_advantages.shape[1] >= selected_log_probs.shape[1]:
            token_advantages = token_advantages[:, : selected_log_probs.shape[1]]
        else:
            pad_width = selected_log_probs.shape[1] - token_advantages.shape[1]
            token_advantages = F.pad(token_advantages, (0, pad_width))

        rescaled_adv, entropy_loss, metrics = self.modulator(
            logits[:, :-1], token_advantages, selected_log_probs
        )

        policy_loss = -(selected_log_probs * rescaled_adv.detach()).mean()
        total_loss = policy_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        drift = self.modulator.compute_entropy_drift(
            rescaled_adv.detach(),
            -selected_log_probs.detach(),
        )
        self.modulator.update_coef(drift)

        metrics["policy_loss"] = policy_loss.item()
        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics


__all__ = ["AEMTrainer", "AdaptiveEntropyModulator"]
