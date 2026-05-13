from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.alignment.praxis.config import PRAXISConfig


class ExpertSafetyAffinity:
    """Routes unsafe tokens toward safety experts via direct router gate calls.

    Calls layer.ffn.router.gate(h) directly on each MoE layer's TopKRouter gate
    (an nn.Linear) to get router logits, then applies cross-entropy toward a
    uniform target distribution over the designated safety_experts indices.
    """

    def __init__(self, moe_layers: nn.ModuleList, config: PRAXISConfig) -> None:
        self.moe_layers = moe_layers
        self.config = config

    def compute(self, hidden: Tensor, const_scores: Tensor) -> Tensor:
        """Compute ESA routing loss.

        Args:
            hidden:       (B, T, D) — final hidden states from transformer.
            const_scores: (B,) — constitutional safety score per sequence [0, 1].

        Returns:
            Scalar ESA loss (non-negative).
        """
        unsafe_mask = const_scores < self.config.tau_safety  # (B,) bool
        if not unsafe_mask.any():
            return hidden.new_zeros(())

        total_loss = hidden.new_zeros(())
        count = 0

        for layer in self.moe_layers:
            gate: nn.Linear = layer.ffn.router.gate
            router_logits = gate(hidden)  # (B, T, n_experts)
            n_experts = router_logits.shape[-1]

            unsafe_logits = router_logits[unsafe_mask]  # (n_unsafe, T, n_experts)
            unsafe_logits = unsafe_logits.reshape(-1, n_experts)  # (n_unsafe*T, n_experts)

            target = torch.zeros(n_experts, device=hidden.device)
            for idx in self.config.safety_experts:
                if idx < n_experts:
                    target[idx] = 1.0
            n_valid = target.sum().clamp(min=1.0)
            target = target / n_valid
            target = target.unsqueeze(0).expand(unsafe_logits.shape[0], -1)  # (N, n_experts)

            log_probs = F.log_softmax(unsafe_logits, dim=-1)
            esa_loss = -(target * log_probs).sum(dim=-1).mean()
            total_loss = total_loss + esa_loss
            count += 1

        if count == 0:
            return hidden.new_zeros(())

        return self.config.alpha_esa * (total_loss / count)
