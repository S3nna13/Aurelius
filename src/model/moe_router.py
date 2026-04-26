from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as F


class RouterType(StrEnum):
    TOP_K = "top_k"
    EXPERT_CHOICE = "expert_choice"
    HASH = "hash"


@dataclass
class RouterConfig:
    n_experts: int = 8
    top_k: int = 2
    router_type: RouterType = RouterType.TOP_K
    aux_loss_coef: float = 0.01
    jitter_noise: float = 0.0


@dataclass
class RouterOutput:
    expert_indices: torch.Tensor
    router_probs: torch.Tensor
    aux_loss: torch.Tensor
    load_distribution: torch.Tensor


class MoERouter(nn.Module):
    """Mixture of Experts token router."""

    def __init__(self, config: RouterConfig, d_model: int = 512) -> None:
        super().__init__()
        self.config = config
        self.gate = nn.Linear(d_model, config.n_experts, bias=False)

    def _top_k_route(self, hidden: torch.Tensor) -> RouterOutput:
        logits = self.gate(hidden)
        if self.training and self.config.jitter_noise > 0.0:
            noise = torch.empty_like(logits).uniform_(
                1.0 - self.config.jitter_noise,
                1.0 + self.config.jitter_noise,
            )
            logits = logits * noise

        probs = F.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, self.config.top_k, dim=-1)

        n_tokens = hidden.shape[0]
        expert_counts = torch.zeros(self.config.n_experts, device=hidden.device, dtype=hidden.dtype)
        for k in range(self.config.top_k):
            expert_counts.scatter_add_(
                0,
                top_indices[:, k],
                torch.ones(n_tokens, device=hidden.device, dtype=hidden.dtype),
            )
        load = expert_counts / (n_tokens * self.config.top_k)

        mean_load = load.mean()
        variance = ((load - mean_load) ** 2).mean()
        cv_sq = variance / (mean_load**2 + 1e-9)
        aux_loss = self.config.aux_loss_coef * cv_sq

        return RouterOutput(
            expert_indices=top_indices,
            router_probs=top_probs,
            aux_loss=aux_loss,
            load_distribution=load,
        )

    def _expert_choice_route(self, hidden: torch.Tensor) -> RouterOutput:
        n_tokens = hidden.shape[0]
        logits = self.gate(hidden)
        probs = F.softmax(logits, dim=-1)

        capacity = max(1, int(n_tokens * self.config.top_k / self.config.n_experts))
        expert_probs_t = probs.t()
        chosen_probs, chosen_indices = torch.topk(expert_probs_t, capacity, dim=-1)

        expert_indices = torch.full(
            (n_tokens, self.config.top_k),
            -1,
            dtype=torch.long,
            device=hidden.device,
        )
        router_probs = torch.zeros(n_tokens, self.config.top_k, device=hidden.device)

        assignment_count = torch.zeros(n_tokens, dtype=torch.long, device=hidden.device)
        for e in range(self.config.n_experts):
            for pos in range(capacity):
                tok = chosen_indices[e, pos].item()
                slot = assignment_count[tok].item()
                if slot < self.config.top_k:
                    expert_indices[tok, slot] = e
                    router_probs[tok, slot] = chosen_probs[e, pos]
                    assignment_count[tok] += 1

        mask = expert_indices[:, 0] >= 0
        expert_indices = torch.where(
            mask.unsqueeze(-1).expand_as(expert_indices),
            expert_indices,
            torch.zeros_like(expert_indices),
        )

        expert_counts = torch.zeros(self.config.n_experts, device=hidden.device, dtype=hidden.dtype)
        for e in range(self.config.n_experts):
            expert_counts[e] = (chosen_indices[e] >= 0).sum().float()
        load = expert_counts / max(n_tokens * self.config.top_k, 1)

        mean_load = load.mean()
        variance = ((load - mean_load) ** 2).mean()
        cv_sq = variance / (mean_load**2 + 1e-9)
        aux_loss = self.config.aux_loss_coef * cv_sq

        return RouterOutput(
            expert_indices=expert_indices,
            router_probs=router_probs,
            aux_loss=aux_loss,
            load_distribution=load,
        )

    def _hash_route(self, hidden: torch.Tensor) -> RouterOutput:
        n_tokens = hidden.shape[0]
        token_ids = torch.arange(n_tokens, device=hidden.device)
        assigned = token_ids % self.config.n_experts

        expert_indices = assigned.unsqueeze(-1).expand(-1, self.config.top_k).clone()
        router_probs = (
            torch.ones(n_tokens, self.config.top_k, device=hidden.device) / self.config.top_k
        )

        expert_counts = torch.zeros(self.config.n_experts, device=hidden.device, dtype=hidden.dtype)
        expert_counts.scatter_add(
            0, assigned, torch.ones(n_tokens, device=hidden.device, dtype=hidden.dtype)
        )
        load = expert_counts / n_tokens

        aux_loss = torch.tensor(0.0, device=hidden.device)

        return RouterOutput(
            expert_indices=expert_indices,
            router_probs=router_probs,
            aux_loss=aux_loss,
            load_distribution=load,
        )

    def forward(self, hidden_states: torch.Tensor) -> RouterOutput:
        if hidden_states.dim() == 3:
            batch, seq_len, d_model = hidden_states.shape
            hidden = hidden_states.reshape(batch * seq_len, d_model)
        else:
            hidden = hidden_states

        if self.config.router_type == RouterType.TOP_K:
            return self._top_k_route(hidden)
        if self.config.router_type == RouterType.EXPERT_CHOICE:
            return self._expert_choice_route(hidden)
        return self._hash_route(hidden)
