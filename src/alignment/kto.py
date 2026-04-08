"""Aurelius -- Kahneman-Tversky Optimization (KTO).

KTO (Ethayarajh et al. 2024) is a preference alignment method grounded in
prospect theory. Unlike DPO which requires paired (chosen, rejected) examples,
KTO only needs individual (prompt, response, label) triples where label indicates
whether the response is desirable (1) or undesirable (0).

Key idea: Humans are more averse to losses than motivated by gains. KTO models
this asymmetry using a prospect-theory-inspired value function:
    - For desirable responses: maximize sigmoid(reward - z_ref)
    - For undesirable responses: maximize sigmoid(-(reward - z_ref))

Where:
    - reward = beta * (log pi(y|x) - log pi_ref(y|x))   [implicit reward]
    - z_ref  = beta * KL(pi || pi_ref)                   [reference point]

References:
    - KTO: Model Alignment as Prospect Theoretic Optimization (Ethayarajh et al. 2024)
      https://arxiv.org/abs/2402.01306
"""

from __future__ import annotations

import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# KTO Loss
# ---------------------------------------------------------------------------

class KTOLoss(nn.Module):
    """KTO loss for binary-labeled preference data.

    Unlike DPO, does not require paired (chosen, rejected) examples.
    Each example is independently labeled as desirable (y=1) or undesirable (y=0).

    L_KTO = E_desirable[-v(reward_d, z_ref)] + E_undesirable[-v(reward_u, z_ref)]

    Where:
    - reward = beta * (log pi(y|x) - log pi_ref(y|x))  [implicit reward]
    - z_ref = beta * KL(pi || pi_ref)  [reference point, estimated from data]
    - v(r, z) = sigmoid(r - z)  for desirable outputs
    - v(r, z) = sigmoid(-(r - z))  for undesirable outputs (loss aversion)

    The loss is cast as NLL: -log sigma(v), so it is always positive.

    Args:
        beta: temperature coefficient (default 0.1)
        desirable_weight: weight for desirable examples (default 1.0)
        undesirable_weight: weight for undesirable examples (default 1.0)
    """

    def __init__(
        self,
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def compute_rewards(
        self,
        policy_logps: torch.Tensor,
        ref_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Implicit reward: beta * (log pi - log pi_ref). Returns (B,)."""
        return self.beta * (policy_logps - ref_logps)

    def estimate_kl(
        self,
        policy_logps: torch.Tensor,
        ref_logps: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence estimate: mean(policy_logps - ref_logps).

        This is the Monte Carlo estimate of KL(pi || pi_ref).
        Returns scalar.
        """
        return (policy_logps - ref_logps).mean()

    def forward(
        self,
        policy_logps: torch.Tensor,
        ref_logps: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute KTO loss.

        Steps:
        1. Split into desirable and undesirable subsets
        2. Compute rewards for all
        3. Estimate z_ref = beta * KL from all examples
        4. For desirable: loss_d = -log sigmoid(reward_d - z_ref)
        5. For undesirable: loss_u = -log sigmoid(-(reward_u - z_ref))
        6. total_loss = dw * mean(loss_d) + uw * mean(loss_u)

        Returns: (loss, metrics_dict)
        metrics_dict = {
            'kl': float,
            'z_ref': float,
            'desirable_reward': float,
            'undesirable_reward': float,
            'reward_margin': float,
        }

        Handle edge case: if no desirable OR no undesirable examples in batch,
        only compute loss for the present type.
        """
        desirable_mask = labels.bool()
        undesirable_mask = ~desirable_mask

        rewards = self.compute_rewards(policy_logps, ref_logps)

        kl = self.estimate_kl(policy_logps, ref_logps)
        kl_nonneg = kl.clamp(min=0.0)
        z_ref = self.beta * kl_nonneg

        has_desirable = desirable_mask.any()
        has_undesirable = undesirable_mask.any()

        rewards_d = rewards[desirable_mask] if has_desirable else rewards.new_tensor([])
        rewards_u = rewards[undesirable_mask] if has_undesirable else rewards.new_tensor([])

        mean_reward_d = rewards_d.mean() if has_desirable else rewards.new_tensor(0.0)
        mean_reward_u = rewards_u.mean() if has_undesirable else rewards.new_tensor(0.0)

        loss = rewards.new_tensor(0.0)

        if has_desirable:
            loss_d = -F.logsigmoid(rewards_d - z_ref)
            loss = loss + self.desirable_weight * loss_d.mean()

        if has_undesirable:
            loss_u = -F.logsigmoid(-(rewards_u - z_ref))
            loss = loss + self.undesirable_weight * loss_u.mean()

        metrics = {
            "kl": kl.item(),
            "z_ref": z_ref.item(),
            "desirable_reward": mean_reward_d.item(),
            "undesirable_reward": mean_reward_u.item(),
            "reward_margin": (mean_reward_d - mean_reward_u).item(),
        }

        return loss, metrics


# ---------------------------------------------------------------------------
# KTO Trainer
# ---------------------------------------------------------------------------

class KTOTrainer:
    """Trains a model using KTO loss."""

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        beta: float = 0.1,
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.kto_loss = KTOLoss(beta=beta)

        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def compute_logps(
        self,
        model: nn.Module,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean log probability over response tokens. Returns scalar."""
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        P = prompt_ids.shape[1]
        R = response_ids.shape[1]

        _, logits, _ = model(input_ids)

        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)

        token_lp = log_probs.gather(
            2, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        response_lp = token_lp[:, P - 1: P + R - 1]

        return response_lp.mean()

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        label: int,
    ) -> dict:
        """Single KTO training step. Returns metrics dict."""
        self.policy_model.train()
        self.ref_model.eval()

        policy_logp = self.compute_logps(self.policy_model, prompt_ids, response_ids)

        with torch.no_grad():
            ref_logp = self.compute_logps(self.ref_model, prompt_ids, response_ids)

        policy_logps = policy_logp.unsqueeze(0)
        ref_logps = ref_logp.unsqueeze(0)
        labels_t = torch.tensor([label], dtype=torch.long)

        loss, metrics = self.kto_loss(policy_logps, ref_logps, labels_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        metrics["loss"] = loss.item()
        return metrics


# ---------------------------------------------------------------------------
# KTO Dataset
# ---------------------------------------------------------------------------

class KTODataset:
    """Dataset for KTO training. Each example is (prompt, response, label).

    No pairing required. Each example is independently labeled as
    desirable (label=1) or undesirable (label=0).
    """

    def __init__(self) -> None:
        self.examples: list[dict] = []

    def add(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        label: int,
    ) -> None:
        """Add one example. label: 1=desirable, 0=undesirable."""
        self.examples.append({
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "label": label,
        })

    def get_batch(self, batch_size: int) -> list[dict]:
        """Random sample of batch_size examples."""
        return random.sample(self.examples, min(batch_size, len(self.examples)))

    def label_distribution(self) -> dict:
        """Return {'n_desirable': int, 'n_undesirable': int, 'ratio': float}."""
        n_desirable = sum(1 for ex in self.examples if ex["label"] == 1)
        n_undesirable = sum(1 for ex in self.examples if ex["label"] == 0)
        total = len(self.examples)
        ratio = n_desirable / total if total > 0 else 0.0
        return {
            "n_desirable": n_desirable,
            "n_undesirable": n_undesirable,
            "ratio": ratio,
        }

    def __len__(self) -> int:
        return len(self.examples)
