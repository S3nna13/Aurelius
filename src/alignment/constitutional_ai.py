"""Constitutional AI: self-critique and revision loop (Bai et al., 2022, arXiv:2212.08073).

Implements the RL-CAI stage: constitutional principles guide preference scoring
and DPO-style training loss.

Also retains the original SL-CAI critique/revision loop for backward compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ===========================================================================
# Constitutional principle + constitution
# ===========================================================================

@dataclass
class ConstitutionalPrinciple:
    """A single constitutional principle.

    Attributes:
        name: Short identifier for the principle.
        critique_prompt: Template string describing what to evaluate.
        weight: Relative importance when aggregating scores.
    """

    name: str
    critique_prompt: str
    weight: float = 1.0


class Constitution:
    """Ordered collection of ConstitutionalPrinciple objects."""

    def __init__(self, principles: List[ConstitutionalPrinciple]) -> None:
        if not principles:
            raise ValueError("Constitution requires at least one principle.")
        self._principles: List[ConstitutionalPrinciple] = list(principles)
        self._by_name: Dict[str, ConstitutionalPrinciple] = {
            p.name: p for p in self._principles
        }

    def get_principle(self, name: str) -> ConstitutionalPrinciple:
        """Return the principle with the given name."""
        return self._by_name[name]

    def total_weight(self) -> float:
        """Sum of all principle weights."""
        return sum(p.weight for p in self._principles)

    def weighted_score(self, scores: Dict[str, float]) -> float:
        """Compute weighted average of per-principle scores."""
        total = 0.0
        for p in self._principles:
            total += p.weight * scores.get(p.name, 0.0)
        tw = self.total_weight()
        return total / tw if tw > 0.0 else 0.0

    @classmethod
    def from_dicts(cls, dicts: List[Dict]) -> "Constitution":
        """Build a Constitution from a list of dicts."""
        principles = [
            ConstitutionalPrinciple(
                name=d["name"],
                critique_prompt=d["critique_prompt"],
                weight=float(d.get("weight", 1.0)),
            )
            for d in dicts
        ]
        return cls(principles)

    def __len__(self) -> int:
        return len(self._principles)

    def __iter__(self):
        return iter(self._principles)


# ===========================================================================
# Constitutional scorer
# ===========================================================================

class ConstitutionalScorer:
    """Score responses against a Constitution using log-probability tensors."""

    def __init__(self, constitution: Constitution) -> None:
        self.constitution = constitution

    def score_response(
        self,
        response_log_probs: List[Tensor],
        principle_weights: Optional[List[float]] = None,
    ) -> Tensor:
        """Compute a per-principle score for a single response.

        Args:
            response_log_probs: List of N 1-D tensors, one per principle.
            principle_weights: Optional list of N floats (unused, for API symmetry).

        Returns:
            Shape (N,) tensor where scores[i] = mean(response_log_probs[i]).
        """
        scores = torch.stack([lp.mean() for lp in response_log_probs])
        return scores

    def aggregate_score(
        self,
        principle_scores: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Weighted mean of per-principle scores -> scalar."""
        if weights is None:
            return principle_scores.mean()
        weights = weights.to(dtype=principle_scores.dtype, device=principle_scores.device)
        return (principle_scores * weights).sum() / weights.sum().clamp(min=1e-8)

    def rank_responses(self, response_scores: Tensor) -> torch.LongTensor:
        """Return response indices sorted from best (highest score) to worst."""
        return torch.argsort(response_scores, descending=True)


# ===========================================================================
# CAI loss (DPO-style preference loss)
# ===========================================================================

class CAILoss(nn.Module):
    """Constitutional AI preference loss (DPO formulation).

    loss = -log_sigmoid(beta * ((chosen - ref_chosen) - (rejected - ref_rejected)))
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        chosen_log_probs: Tensor,
        rejected_log_probs: Tensor,
        ref_chosen: Tensor,
        ref_rejected: Tensor,
    ) -> Tuple[Tensor, Dict]:
        """Compute the CAI/DPO loss.

        Returns:
            (loss_scalar, {"loss": ..., "accuracy": ..., "margin": ...})
        """
        chosen_ratio = chosen_log_probs - ref_chosen.detach()
        rejected_ratio = rejected_log_probs - ref_rejected.detach()
        logit = self.beta * (chosen_ratio - rejected_ratio)
        loss = -F.logsigmoid(logit).mean()

        with torch.no_grad():
            accuracy = (logit > 0).float().mean()
            margin = logit.mean()

        metrics: Dict = {
            "loss": loss.detach().item(),
            "accuracy": accuracy.item(),
            "margin": margin.item(),
        }
        return loss, metrics


# ===========================================================================
# CAI trainer
# ===========================================================================

class CAITrainer:
    """Wraps policy and frozen reference model for CAI/DPO training."""

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        constitution: Constitution,
        loss_fn: CAILoss,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.constitution = constitution
        self.loss_fn = loss_fn
        self.freeze_ref()

    def freeze_ref(self) -> None:
        """Freeze all parameters of the reference model."""
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

    def train_step(
        self,
        chosen_lp: Tensor,
        rejected_lp: Tensor,
        ref_chosen: Tensor,
        ref_rejected: Tensor,
    ) -> Dict:
        """Perform a single CAI training step.

        Returns:
            Metrics dict with keys "loss", "accuracy", "margin".
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss, metrics = self.loss_fn(chosen_lp, rejected_lp, ref_chosen, ref_rejected)
        loss.backward()
        self.optimizer.step()
        return metrics


# ===========================================================================
# Legacy SL-CAI critique/revision loop (retained for backward compatibility)
# ===========================================================================

@dataclass
class ConstitutionalAIConfig:
    """Configuration for RLAIF Constitutional AI training loop."""

    n_principles: int = 4
    n_critique_rounds: int = 2
    sft_loss_coeff: float = 1.0
    kl_coeff: float = 0.1
    max_seq_len: int = 128


@dataclass
class Principle:
    """A constitutional principle for critiquing outputs (RLAIF variant)."""

    name: str
    description: str
    critique_prompt: str
    revision_prompt: str


def default_principles() -> list[Principle]:
    """Return a small set of default principles."""
    return [
        Principle("helpful", "Be helpful", "Is the response helpful?", "Make it more helpful."),
        Principle("harmless", "Avoid harm", "Does the response avoid harm?", "Remove harmful content."),
        Principle("honest", "Be honest", "Is the response honest?", "Correct any inaccuracies."),
        Principle("concise", "Be concise", "Is the response concise?", "Make it shorter."),
    ]


def score_principle_compliance(
    logits: Tensor,
    token_ids: Tensor,
    principle_idx: int,
    n_principles: int,
) -> Tensor:
    """Compute compliance score for a principle given logits and tokens."""
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
        token_ids = token_ids.unsqueeze(0)

    B, T, V = logits.shape

    if T < 2:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = token_ids[:, 1:]
    token_lp = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    scale = 1.0 - principle_idx / max(n_principles, 1) * 0.1
    return token_lp.mean() * scale


def compute_critique_loss(
    policy_logits: Tensor,
    ref_logits: Tensor,
    target_ids: Tensor,
    principle_scores: Tensor,
    config: "ConstitutionalAIConfig",
) -> tuple[Tensor, dict]:
    """Compute Constitutional AI training loss (legacy SL-CAI variant)."""
    B, T, V = policy_logits.shape

    if T >= 2:
        shift_logits = policy_logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        sft_loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
        )
    else:
        sft_loss = torch.tensor(0.0, dtype=policy_logits.dtype, device=policy_logits.device)

    policy_log_probs = F.log_softmax(policy_logits, dim=-1)
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
    kl_loss = F.kl_div(
        ref_log_probs,
        policy_log_probs.detach().exp(),
        reduction="batchmean",
        log_target=False,
    ).clamp(min=0.0)

    principle_reward = principle_scores.mean()
    total_loss = (
        config.sft_loss_coeff * sft_loss
        + config.kl_coeff * kl_loss
        - principle_reward
    )

    metrics: dict = {
        "sft_loss": sft_loss.detach().item(),
        "kl_loss": kl_loss.detach().item(),
        "principle_reward": principle_reward.detach().item(),
        "total_loss": total_loss.detach().item(),
    }
    return total_loss, metrics


class SelfCritiqueBuffer:
    """Stores (prompt, response, revised_response, principle_scores) tuples."""

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._prompts: list[Tensor] = []
        self._responses: list[Tensor] = []
        self._revised: list[Tensor] = []
        self._scores: list[Tensor] = []

    def add(self, prompt_ids, response_ids, revised_ids, scores) -> None:
        if len(self._prompts) >= self.max_size:
            self._prompts.pop(0)
            self._responses.pop(0)
            self._revised.pop(0)
            self._scores.pop(0)
        self._prompts.append(prompt_ids.detach().cpu())
        self._responses.append(response_ids.detach().cpu())
        self._revised.append(revised_ids.detach().cpu())
        self._scores.append(scores.detach().cpu())

    def sample(self, batch_size: int):
        if len(self._prompts) < batch_size:
            return None
        indices = torch.randperm(len(self._prompts))[:batch_size].tolist()
        prompts = torch.stack([self._prompts[i] for i in indices])
        responses = torch.stack([self._responses[i] for i in indices])
        revised = torch.stack([self._revised[i] for i in indices])
        scores = torch.stack([self._scores[i] for i in indices])
        return prompts, responses, revised, scores

    def __len__(self) -> int:
        return len(self._prompts)


class ConstitutionalAITrainer:
    """Trains a model using Constitutional AI self-critique (legacy SL-CAI variant)."""

    def __init__(self, policy, ref_model, config, optimizer, principles=None) -> None:
        self.policy = policy
        self.ref_model = ref_model
        self.config = config
        self.optimizer = optimizer
        self.principles = principles if principles is not None else default_principles()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

    def critique_and_revise(self, prompt_ids, response_ids):
        self.policy.eval()
        with torch.no_grad():
            _, logits, _ = self.policy(response_ids)
        n_principles = self.config.n_principles
        B = response_ids.shape[0]
        scores = torch.zeros(B, n_principles, dtype=logits.dtype, device=logits.device)
        for i in range(n_principles):
            score = score_principle_compliance(logits, response_ids, i, n_principles)
            scores[:, i] = score
        revised_ids = response_ids.clone()
        self.policy.train()
        return revised_ids, scores

    def train_step(self, prompt_ids, response_ids) -> dict:
        revised_ids, principle_scores = self.critique_and_revise(prompt_ids, response_ids)
        self.policy.train()
        _, policy_logits, _ = self.policy(revised_ids)
        with torch.no_grad():
            _, ref_logits, _ = self.ref_model(revised_ids)
        loss, metrics = compute_critique_loss(
            policy_logits, ref_logits, revised_ids, principle_scores, self.config,
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return metrics
