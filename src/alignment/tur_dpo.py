"""TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization.

Topology-aware DPO that rewards reasoning derivation quality, not just answer.
Elicits lightweight reasoning topologies and combines semantic faithfulness,
utility, and topology quality into a calibrated uncertainty signal.

Paper: arXiv:2605.00224 (ICML 2026) — Abdullah et al.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReasoningTopologyExtractor(nn.Module):
    """Extract lightweight reasoning topologies from chain-of-thought."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.topology_quality_head = nn.Linear(d_model, 3, bias=False)

    def extract_topology(self, hidden_states: Tensor) -> dict[str, Tensor]:
        """Compute per-step topology scores.

        Returns:
            dict with keys: 'faithfulness', 'utility', 'topology_quality'
            each shape (B, S)
        """
        logits = self.topology_quality_head(hidden_states)
        faithfulness, utility, topq = logits.chunk(3, dim=-1)
        return {
            "faithfulness": faithfulness.squeeze(-1),
            "utility": utility.squeeze(-1),
            "topology_quality": topq.squeeze(-1),
        }


class UncalibratedUncertaintyReward(nn.Module):
    """Small learnable reward decomposed over semantic + utility + topology."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(3))

    def forward(self, hidden_states: Tensor, topology: dict[str, Tensor]) -> Tensor:
        """Compute uncertainty-weighted reward from decomposed signals."""
        del hidden_states
        w = F.softmax(self.weight, dim=0)
        r_sem = topology["faithfulness"] * w[0]
        r_util = topology["utility"] * w[1]
        r_topq = topology["topology_quality"] * w[2]
        return r_sem + r_util + r_topq


def tur_dpo_loss(
    chosen_log_probs: Tensor,
    rejected_log_probs: Tensor,
    chosen_uncertainty: Tensor,
    rejected_uncertainty: Tensor,
    ref_chosen_log_probs: Tensor | None = None,
    ref_rejected_log_probs: Tensor | None = None,
    beta: float = 0.1,
) -> tuple[Tensor, dict]:
    """TUR-DPO loss with uncertainty-weighted preference.

    Args:
        chosen_log_probs: (B, S) log probs for chosen response
        rejected_log_probs: (B, S) log probs for rejected response
        chosen_uncertainty: (B,) uncertainty for chosen
        rejected_uncertainty: (B,) uncertainty for rejected
        beta: DPO temperature
    Returns:
        (loss, metrics)
    """
    pi_logratio = chosen_log_probs.sum(-1) - rejected_log_probs.sum(-1)
    if ref_chosen_log_probs is not None and ref_rejected_log_probs is not None:
        ref_logratio = (ref_chosen_log_probs.sum(-1) - ref_rejected_log_probs.sum(-1)).detach()
    else:
        ref_logratio = torch.zeros_like(pi_logratio)

    uncertainty_weight = torch.sigmoid(
        torch.stack([chosen_uncertainty, rejected_uncertainty], dim=1)
    ).mean(dim=1)

    delta = pi_logratio - ref_logratio
    r_clamped = delta.clamp(-20, 20)
    ratio = torch.exp(r_clamped)

    loss = -torch.log(torch.sigmoid(delta / beta) + 1e-8)
    uncertainty_penalty = (uncertainty_weight * loss).mean()

    metrics = {
        "tur_dpo_loss": uncertainty_penalty.item(),
        "mean_uncertainty": uncertainty_weight.mean().item(),
        "mean_ratio": ratio.mean().item(),
    }
    return uncertainty_penalty, metrics


class TURDPOTrainer:
    """TUR-DPO training loop."""

    def __init__(self, model: nn.Module, ref_model: nn.Module, config: dict | None = None) -> None:
        self.model = model
        self.ref_model = ref_model
        self.cfg = config or {}
        d_model = self.cfg.get("d_model")
        if d_model is None:
            model_config = getattr(model, "config", None)
            d_model = getattr(model_config, "hidden_size", None) or getattr(
                model_config,
                "d_model",
                None,
            )
        self._topology_d_model = int(d_model) if d_model is not None else None
        self.topology_extractor = (
            ReasoningTopologyExtractor(self._topology_d_model)
            if self._topology_d_model is not None
            else None
        )
        self.uncertainty_reward = (
            UncalibratedUncertaintyReward(self._topology_d_model)
            if self._topology_d_model is not None
            else None
        )

    @staticmethod
    def _token_log_probs(model: nn.Module, token_ids: Tensor) -> tuple[Tensor, object]:
        outputs = model(token_ids)
        all_log_probs = F.log_softmax(outputs.logits[:, :-1, :], dim=-1)
        token_log_probs = all_log_probs.gather(
            dim=-1,
            index=token_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)
        return token_log_probs, outputs

    def _uncertainty_scores(self, token_ids: Tensor, outputs: object) -> Tensor:
        batch_size = token_ids.size(0)
        fallback = torch.ones(batch_size, device=token_ids.device)
        if self.topology_extractor is None or self.uncertainty_reward is None:
            return fallback

        hidden_states = getattr(outputs, "hidden_states", None)
        if isinstance(hidden_states, (list, tuple)) and hidden_states:
            hidden = hidden_states[-1]
        elif isinstance(hidden_states, Tensor):
            hidden = hidden_states[-1] if hidden_states.dim() == 4 else hidden_states
        else:
            hidden = getattr(outputs, "last_hidden_state", None)

        if not isinstance(hidden, Tensor) or hidden.shape[-1] != self._topology_d_model:
            return fallback

        topology = self.topology_extractor.extract_topology(hidden)
        scores = self.uncertainty_reward(hidden, topology).mean(dim=-1)
        return scores.to(device=token_ids.device)

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        chosen_ids = batch["chosen_ids"]
        rejected_ids = batch["rejected_ids"]

        chosen_log_probs, chosen_outputs = self._token_log_probs(self.model, chosen_ids)
        rejected_log_probs, rejected_outputs = self._token_log_probs(self.model, rejected_ids)

        chosen_uncertainty = self._uncertainty_scores(chosen_ids, chosen_outputs)
        rejected_uncertainty = self._uncertainty_scores(rejected_ids, rejected_outputs)
        with torch.no_grad():
            ref_chosen_log_probs, _ = self._token_log_probs(self.ref_model, chosen_ids)
            ref_rejected_log_probs, _ = self._token_log_probs(self.ref_model, rejected_ids)

        loss, metrics = tur_dpo_loss(
            chosen_log_probs,
            rejected_log_probs,
            chosen_uncertainty,
            rejected_uncertainty,
            ref_chosen_log_probs,
            ref_rejected_log_probs,
            beta=self.cfg.get("beta", 0.1),
        )
        return loss, metrics


__all__ = [
    "TURDPOTrainer",
    "tur_dpo_loss",
    "ReasoningTopologyExtractor",
    "UncalibratedUncertaintyReward",
]
