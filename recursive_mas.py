"""Recursive multi-agent loop with cross-agent latent state transfer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, List, Tuple, Callable
import logging
logger = logging.getLogger(__name__)


class RecursiveLink(nn.Module):
    """Cross-agent latent state transfer with learned gating.

    Maps (b, d_model) latent state to (b, d_model) via:
      h_norm = LayerNorm(proj(h))
      delta = MLP(h_norm)
      gate = sigmoid(Linear(concat(h_norm, delta_proj)))
      h_out = gate * h_norm + (1-gate) * delta
    """

    def __init__(self, d_model: int, d_latent: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_latent)
        self.norm = nn.LayerNorm(d_latent)
        self.delta_mlp = nn.Sequential(
            nn.Linear(d_latent, d_latent * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_latent * 4, d_latent),
        )
        self.gate = nn.Linear(d_latent * 2, d_latent)
        self.output_proj = nn.Linear(d_latent, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        for module in [self.input_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for name, param in self.delta_mlp.named_parameters():
            nn.init.xavier_uniform_(param, gain=0.5) if 'weight' in name else nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        h_prev: torch.Tensor,
        h_new: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_proj = self.input_proj(h_prev)
        h_norm = self.norm(h_proj)
        delta = self.delta_mlp(h_norm)

        if h_new is not None:
            h_new_proj = self.input_proj(h_new)
            gate_input = torch.cat([h_norm, h_new_proj], dim=-1)
        else:
            gate_input = torch.cat([h_norm, h_norm], dim=-1)

        g = torch.sigmoid(self.gate(gate_input))
        h_latent = g * h_norm + (1 - g) * delta
        return self.output_proj(h_latent)


class RecursiveAgentWrapper(nn.Module):
    """Orchestrates multiple agents in a recursive collaboration loop.

    Each recursion round applies all agents in sequence. The RecursiveLink
    modules transfer latent state between agents across rounds. The latent
    state accumulates information over the full recursion trace.
    """

    def __init__(
        self,
        d_model: int,
        d_latent: int,
        n_agents: int,
        agent_factory: Callable[[], nn.Module],
        n_rounds: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.n_agents = n_agents
        self.n_rounds = n_rounds

        self.agents = nn.ModuleList([agent_factory() for _ in range(n_agents)])
        self.links = nn.ModuleList([
            RecursiveLink(d_model, d_latent)
            for _ in range(n_agents * n_rounds)
        ])
        self.round_embed = nn.Embedding(n_rounds, d_model)
        self.agent_embed = nn.Embedding(n_agents, d_model)
        self.h_proj = nn.Linear(d_model * 2, d_model)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
        )
        self.value_head = nn.Linear(d_model, 1)
        self.norm_out = nn.LayerNorm(d_model)

    def _get_link_idx(self, agent_idx: int, round_idx: int) -> int:
        return agent_idx * self.n_rounds + round_idx

    def forward(
        self,
        h: torch.Tensor,
        tool_descs: Optional[torch.Tensor] = None,
        return_all_rounds: bool = False,
    ) -> Dict[str, Any]:
        b, t, d = h.shape
        device = h.device
        latent = h[:, -1:]  # (b, 1, d)
        all_round_outputs = [] if return_all_rounds else None

        for r in range(self.n_rounds):
            round_confidences = []
            for a in range(self.n_agents):
                link = self.links[self._get_link_idx(a, r)]
                agent = self.agents[a]

                r_emb = self.round_embed(torch.tensor(r, device=device)).view(1, 1, d)
                a_emb = self.agent_embed(torch.tensor(a, device=device)).view(1, 1, d)
                latent_in = self.norm_out(latent + r_emb.expand(b, -1, -1) + a_emb.expand(b, -1, -1))

                h_aug = torch.cat([h, latent_in.expand(-1, t, -1)], dim=-1)
                h_in = self.h_proj(h_aug)

                agent_out = agent(h_in, tool_descs, full_cycle=True) if hasattr(agent, 'forward') and 'full_cycle' in agent.forward.__code__.co_varnames else agent(h_in)
                h_agent = agent_out if isinstance(agent_out, torch.Tensor) else agent_out.get('hidden', agent_out)

                h_prev_2d = latent.squeeze(1)
                h_new_2d = h_agent.mean(dim=1) if h_agent.dim() == 3 else h_agent
                latent = link(h_prev_2d, h_new_2d).unsqueeze(1)

                conf = torch.sigmoid(self.confidence_head(latent))
                round_confidences.append(conf)

                if return_all_rounds:
                    all_round_outputs.append({
                        'round': r, 'agent': a,
                        'latent': latent.detach().clone(),
                        'confidence': conf,
                    })

        value = self.value_head(latent.mean(dim=1)).squeeze(-1)
        confidence = torch.stack(round_confidences).mean(dim=0)

        result = {
            'latent': latent,
            'value': value,
            'confidence': confidence.squeeze(-1),
            'n_rounds': self.n_rounds,
            'n_agents': self.n_agents,
        }
        if return_all_rounds:
            result['rounds'] = all_round_outputs
        return result


class InnerOuterOptimizer:
    """Inner-outer loop optimization for recursive multi-agent systems.

    Inner loop: agent-local supervised objectives.
    Outer loop: credit assignment across recursion rounds via RecursiveLink grads.
    """

    def __init__(
        self,
        mas: RecursiveAgentWrapper,
        inner_lr: float = 1e-4,
        outer_lr: float = 1e-5,
        tau: float = 0.1,
    ):
        self.mas = mas
        self.tau = tau

        inner_params = []
        for agent in mas.agents:
            inner_params.extend(list(agent.parameters()))
        self.inner_optim = torch.optim.AdamW(inner_params, lr=inner_lr, weight_decay=0.01)

        outer_params = list(mas.links.parameters()) + list(mas.confidence_head.parameters())
        outer_params += list(mas.round_embed.parameters()) + list(mas.agent_embed.parameters())
        outer_params += list(mas.value_head.parameters()) + list(mas.norm_out.parameters())
        outer_params += list(mas.h_proj.parameters())
        self.outer_optim = torch.optim.AdamW(outer_params, lr=outer_lr, weight_decay=0.01)

    def inner_step(
        self,
        h: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        tool_descs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self.mas.train()
        self.inner_optim.zero_grad()
        out = self.mas(h, tool_descs, return_all_rounds=True)
        total_loss = 0.0
        metrics = {}

        if 'value_labels' in targets:
            value_loss = F.mse_loss(out['value'], targets['value_labels'])
            total_loss = total_loss + 0.5 * value_loss
            metrics['value_loss'] = value_loss.item()

        conf = out['confidence'].mean()
        total_loss = total_loss - self.tau * conf
        metrics['confidence'] = conf.item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mas.parameters(), 1.0)
        self.inner_optim.step()
        metrics['inner_loss'] = total_loss.item()
        return metrics

    def outer_step(
        self,
        h: torch.Tensor,
        rewards: torch.Tensor,
        tool_descs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        self.mas.train()
        self.outer_optim.zero_grad()
        out = self.mas(h, tool_descs, return_all_rounds=True)

        round_values = []
        for r_info in out['rounds']:
            v = self.mas.value_head(r_info['latent'].mean(dim=1))
            round_values.append(v)
        value_pred = torch.stack(round_values, dim=-1).mean(dim=-1)

        if rewards.dim() == 0:
            rewards = rewards.unsqueeze(0)
        advantage = rewards - value_pred
        outer_loss = (advantage ** 2).mean()

        outer_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mas.links.parameters(), 0.5)
        self.outer_optim.step()

        return {
            'outer_loss': outer_loss.item(),
            'avg_advantage': advantage.mean().item(),
        }

    def train_step(
        self,
        h: torch.Tensor,
        targets: Dict[str, torch.Tensor],
        task_embed: torch.Tensor,
        rewards: torch.Tensor,
        tool_descs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        inner_metrics = self.inner_step(h, targets, tool_descs)
        outer_metrics = self.outer_step(h, rewards, tool_descs)
        return {**inner_metrics, **outer_metrics}


class RecursiveMASConfig:
    """Configuration container for RecursiveMAS parameters."""
    def __init__(
        self,
        d_model: int = 768,
        d_latent: int = 256,
        n_agents: int = 4,
        n_rounds: int = 4,
        inner_lr: float = 1e-4,
        outer_lr: float = 1e-5,
        tau: float = 0.1,
    ):
        self.d_model = d_model
        self.d_latent = d_latent
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.tau = tau
