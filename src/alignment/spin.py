"""Aurelius -- SPIN: Self-Play Fine-Tuning (Chen et al. 2024).

The model iteratively improves by playing against a frozen copy of its previous
self. At iteration t, the current model pi_t learns to distinguish human-generated
responses from responses it generated at iteration t-1. No preference labels are
required -- only human demonstrations.

Reference:
    Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models
    Chen et al., 2024. https://arxiv.org/abs/2401.01335

Loss:
    L_SPIN = -E[log sigma(beta * (log p_theta(y_real|x) - log p_theta_prev(y_real|x))
                          - beta * (log p_theta(y_gen|x) - log p_theta_prev(y_gen|x)))]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# SPIN Loss
# ---------------------------------------------------------------------------


class SPINLoss(nn.Module):
    """SPIN loss function (DPO-style, using previous iteration as reference).

    Args:
        beta: Temperature / KL coefficient. Default: 0.1.
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        pi_real: torch.Tensor,
        pi_gen: torch.Tensor,
        ref_real: torch.Tensor,
        ref_gen: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute SPIN loss.

        Args:
            pi_real:  Log probs of real (human) completions under current policy.  Shape (B,).
            pi_gen:   Log probs of generated completions under current policy.     Shape (B,).
            ref_real: Log probs of real completions under reference (prev) model.  Shape (B,).
            ref_gen:  Log probs of generated completions under reference model.    Shape (B,).

        Returns:
            Tuple of (loss_scalar, metrics_dict) where metrics_dict contains:
                "accuracy":    fraction of samples where reward_real > reward_gen.
                "reward_real": mean beta * (pi_real - ref_real).
                "reward_gen":  mean beta * (pi_gen - ref_gen).
                "margin":      mean (reward_real - reward_gen).
        """
        reward_real = self.beta * (pi_real - ref_real)  # (B,)
        reward_gen = self.beta * (pi_gen - ref_gen)  # (B,)

        loss = -F.logsigmoid(reward_real - reward_gen).mean()

        accuracy = (reward_real > reward_gen).float().mean().item()

        metrics: dict = {
            "accuracy": accuracy,
            "reward_real": reward_real.mean().item(),
            "reward_gen": reward_gen.mean().item(),
            "margin": (reward_real - reward_gen).mean().item(),
        }

        return loss, metrics


# ---------------------------------------------------------------------------
# SPIN Data Collector
# ---------------------------------------------------------------------------


class SPINDataCollector:
    """Builds training pairs from real and generated log-probability sequences.

    Args:
        beta: Temperature coefficient (stored for reference). Default: 0.1.
    """

    def __init__(self, beta: float = 0.1) -> None:
        self.beta = beta

    def sequence_log_prob(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Reduce a sequence of per-token log probs to a scalar via mean.

        Args:
            log_probs: Per-token log probabilities, shape (T,) or (B, T).

        Returns:
            Mean log probability, scalar (0-d) or shape (B,).
        """
        return log_probs.mean(dim=-1)

    def build_pairs(
        self,
        real_log_probs: list[torch.Tensor],
        gen_log_probs: list[torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Zip real and generated log-prob sequences into (real, gen) pairs.

        Args:
            real_log_probs: List of per-token log-prob tensors for real completions.
            gen_log_probs:  List of per-token log-prob tensors for generated completions.

        Returns:
            List of (real_scalar, gen_scalar) tuples, one per sample.
        """
        return [
            (self.sequence_log_prob(r), self.sequence_log_prob(g))
            for r, g in zip(real_log_probs, gen_log_probs)
        ]


# ---------------------------------------------------------------------------
# SPIN Trainer
# ---------------------------------------------------------------------------


class SPINTrainer:
    """Orchestrates SPIN training: one policy model vs. a frozen reference.

    Args:
        policy_model: The model being trained (current iteration).
        ref_model:    Frozen copy of the previous iteration's weights.
        optimizer:    Optimizer for policy_model parameters.
        loss_fn:      SPINLoss instance.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: SPINLoss,
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.freeze_ref()

    def freeze_ref(self) -> None:
        """Set all reference model parameters to requires_grad=False."""
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def compute_sequence_log_prob(
        self,
        model: nn.Module,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute per-sequence log probability from a causal LM.

        Forward pass: model(input_ids) -> logits of shape (B, T, V).
        Per-token log probs are computed via log_softmax over vocabulary.
        Positions where labels == -100 are masked out (not counted).
        The sum of valid token log probs is returned for each sequence.

        Args:
            model:     A model with forward(input_ids: LongTensor) -> Tensor(B, T, V).
            input_ids: Token IDs.  Shape (B, T).
            labels:    Target IDs. Shape (B, T). Positions with -100 are ignored.

        Returns:
            Per-sequence log probability.  Shape (B,).
        """
        logits = model(input_ids)  # (B, T, V)
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

        # Clamp labels to valid range for gather; masked positions excluded below.
        labels_clamped = labels.clamp(min=0)  # (B, T)
        token_log_probs = log_probs.gather(dim=2, index=labels_clamped.unsqueeze(-1)).squeeze(
            -1
        )  # (B, T)

        mask = (labels != -100).float()  # (B, T)
        sequence_log_probs = (token_log_probs * mask).sum(dim=-1)  # (B,)
        return sequence_log_probs

    def spin_step(
        self,
        real_ids: torch.LongTensor,
        gen_ids: torch.LongTensor,
        labels: torch.LongTensor,
    ) -> tuple[torch.Tensor, dict]:
        """Run one SPIN training step.

        1. Compute policy log probs for real and generated sequences.
        2. Compute reference log probs (no_grad) for both.
        3. Compute SPIN loss.
        4. Backward pass + optimizer step.

        Args:
            real_ids: Token IDs for real (human) completions.  Shape (B, T).
            gen_ids:  Token IDs for generated completions.     Shape (B, T).
            labels:   Label IDs (positions with -100 are masked). Shape (B, T).

        Returns:
            Tuple of (loss_tensor, metrics_dict).
        """
        # Policy log probs (with grad)
        self.policy_model.train()
        pi_real = self.compute_sequence_log_prob(self.policy_model, real_ids, labels)
        pi_gen = self.compute_sequence_log_prob(self.policy_model, gen_ids, labels)

        # Reference log probs (frozen, no grad)
        with torch.no_grad():
            ref_real = self.compute_sequence_log_prob(self.ref_model, real_ids, labels)
            ref_gen = self.compute_sequence_log_prob(self.ref_model, gen_ids, labels)

        loss, metrics = self.loss_fn(pi_real, pi_gen, ref_real, ref_gen)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, metrics
