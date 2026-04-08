"""AdaLoRA: Adaptive rank allocation for parameter-efficient fine-tuning.

Decomposes LoRA update as W_delta = P * diag(Λ) * Q (SVD form).
Learns to zero out small singular values (magnitude pruning) to
concentrate rank budget on the most important directions.

Reference: Zhang et al. 2023, arXiv:2303.10512
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdaLoRAConfig:
    init_rank: int = 12            # starting rank (will be pruned down)
    target_rank: int = 4           # target rank after pruning
    alpha: float = 32.0            # LoRA scaling factor
    beta1: float = 0.85            # EMA coefficient for importance scores
    beta2: float = 0.85            # EMA coefficient for sensitivity scores
    pruning_warmup_steps: int = 10  # steps before starting to prune
    total_steps: int = 100          # total training steps (for scheduling)
    reg_lambda: float = 0.1         # orthogonality regularization weight


class AdaLoRALinear(nn.Module):
    """Linear layer with SVD-parameterized adaptive LoRA.

    W_delta = P * diag(lambda_vec) * Q

    Where P (d_out, r) and Q (r, d_in) are learned orthogonal matrices,
    and lambda_vec (r,) are learned singular values that can be zeroed.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        cfg: AdaLoRA configuration
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cfg: AdaLoRAConfig,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cfg = cfg
        self.rank = cfg.init_rank
        self.scale = cfg.alpha / cfg.init_rank

        # Frozen base weight
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features), requires_grad=False
        )

        # SVD-parameterized LoRA: W_delta = P @ diag(lambda_vec) @ Q
        self.P = nn.Parameter(torch.randn(out_features, cfg.init_rank) * 0.01)
        self.lambda_vec = nn.Parameter(torch.ones(cfg.init_rank) * 0.01)
        self.Q = nn.Parameter(torch.randn(cfg.init_rank, in_features) * 0.01)

        # Importance scores (EMA, not learned)
        self.register_buffer("importance", torch.ones(cfg.init_rank))

        # Mask for zeroed singular values (1=active, 0=pruned)
        self.register_buffer("sv_mask", torch.ones(cfg.init_rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply base weight + masked LoRA update."""
        base_out = F.linear(x, self.weight)
        # Masked singular values: shape (r,)
        masked_sv = self.lambda_vec * self.sv_mask
        # P * diag(masked_sv): scale each column of P by corresponding sv
        # Equivalent to P @ diag(masked_sv) which gives (out, r)
        # Then (P @ diag(masked_sv)) @ Q gives (out, in)
        # F.linear(F.linear(x, Q), P_scaled) = P_scaled @ Q @ x^T
        P_scaled = self.P * (self.scale * masked_sv).unsqueeze(0)  # (out, r)
        lora_out = F.linear(F.linear(x, self.Q), P_scaled)
        return base_out + lora_out

    def update_importance(self, step: int) -> None:
        """Update importance scores using gradient magnitude (EMA).

        importance[i] = beta1 * importance[i] + (1-beta1) * |lambda_vec.grad[i]|
        """
        if self.lambda_vec.grad is not None:
            grad_mag = self.lambda_vec.grad.detach().abs()
            self.importance = (
                self.cfg.beta1 * self.importance
                + (1 - self.cfg.beta1) * grad_mag
            )

    def prune_to_budget(self, target_rank: int) -> None:
        """Zero out the least important singular values.

        Keeps the top-target_rank singular values by importance score.
        Updates sv_mask accordingly.
        """
        if target_rank >= self.rank:
            return
        # Zero mask for bottom (rank - target_rank) singular values
        _, indices = torch.topk(self.importance, target_rank)
        mask = torch.zeros_like(self.sv_mask)
        mask[indices] = 1.0
        self.sv_mask.copy_(mask)

    def orthogonality_loss(self) -> torch.Tensor:
        """Regularization to keep P and Q approximately orthogonal.

        L_orth = ||P.T @ P - I||_F^2 + ||Q @ Q.T - I||_F^2
        """
        r = self.rank
        PtP = self.P.T @ self.P   # (r, r)
        QQt = self.Q @ self.Q.T   # (r, r)
        I = torch.eye(r, device=self.P.device, dtype=self.P.dtype)
        return (PtP - I).pow(2).sum() + (QQt - I).pow(2).sum()


class AdaLoRATrainer:
    """Training wrapper that manages rank pruning schedule.

    Calls update_importance() and prune_to_budget() at the right steps.
    """

    def __init__(
        self,
        model: nn.Module,
        adalora_layers: list[AdaLoRALinear],
        cfg: AdaLoRAConfig,
        lr: float = 1e-4,
    ) -> None:
        self.model = model
        self.adalora_layers = adalora_layers
        self.cfg = cfg
        # Collect only trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr)

    def get_current_target_rank(self, step: int) -> int:
        """Linear rank schedule from init_rank to target_rank.

        Before warmup: init_rank. After total_steps: target_rank.
        Linear interpolation in between.
        """
        cfg = self.cfg
        if step <= cfg.pruning_warmup_steps:
            return cfg.init_rank
        if step >= cfg.total_steps:
            return cfg.target_rank
        # Linear interpolation between warmup and total_steps
        progress = (step - cfg.pruning_warmup_steps) / max(
            1, cfg.total_steps - cfg.pruning_warmup_steps
        )
        rank = cfg.init_rank + progress * (cfg.target_rank - cfg.init_rank)
        return max(cfg.target_rank, round(rank))

    def train_step(
        self,
        loss: torch.Tensor,
        step: int,
    ) -> dict[str, float]:
        """Backward + importance update + scheduled pruning + optimizer step.

        Returns metrics: {"loss": float, "active_rank": float, "orth_loss": float}
        """
        # Add orthogonality regularization to the loss
        orth_loss = sum(
            layer.orthogonality_loss() for layer in self.adalora_layers
        )
        total_loss = loss + self.cfg.reg_lambda * orth_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        # Update importance scores before optimizer step (gradients still valid)
        for layer in self.adalora_layers:
            layer.update_importance(step)

        self.optimizer.step()

        # Prune to current target rank
        target_rank = self.get_current_target_rank(step)
        for layer in self.adalora_layers:
            layer.prune_to_budget(target_rank)

        # Compute average active rank across all adalora layers
        active_ranks = [layer.sv_mask.sum().item() for layer in self.adalora_layers]
        avg_active_rank = sum(active_ranks) / max(1, len(active_ranks))

        return {
            "loss": loss.item(),
            "active_rank": avg_active_rank,
            "orth_loss": orth_loss.item() if isinstance(orth_loss, torch.Tensor) else float(orth_loss),
        }
