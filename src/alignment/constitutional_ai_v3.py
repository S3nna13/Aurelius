"""Constitutional AI v3: iterative critique-revision pipeline (Bai et al., 2022).

Implements the trainable/differentiable components of CAI:
  - ConstitutionalPrinciple dataclass
  - CritiqueHead: predicts per-principle harmlessness scores from hidden states
  - RevisionScorer: scores revision quality via cosine similarity
  - CAITrainer: trains the critique head with constitutional preference data
  - ConstitutionalFilter: inference-time filter / flagging of low-harmlessness outputs

Reference: Bai et al., "Constitutional AI: Harmlessness from AI Feedback" (arXiv:2212.08073).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# ConstitutionalPrinciple
# ---------------------------------------------------------------------------


@dataclass
class ConstitutionalPrinciple:
    """A single constitutional principle used for critique and revision.

    Attributes:
        name: Short identifier for the principle.
        critique_prompt: Prompt text describing what aspect to evaluate.
        revision_prompt: Prompt text guiding the revision toward this principle.
        weight: Relative importance when aggregating scores (default 1.0).
    """

    name: str
    critique_prompt: str
    revision_prompt: str
    weight: float = 1.0

    def to_dict(self) -> dict:
        """Return a dict with all fields."""
        return {
            "name": self.name,
            "critique_prompt": self.critique_prompt,
            "revision_prompt": self.revision_prompt,
            "weight": self.weight,
        }


# ---------------------------------------------------------------------------
# CritiqueHead
# ---------------------------------------------------------------------------


class CritiqueHead(nn.Module):
    """Produces per-principle harmlessness scores from hidden states.

    Applies a two-layer MLP to the last-position hidden state to produce
    per-principle scores in (0, 1).

    Args:
        d_model: Hidden state dimension.
        n_principles: Number of constitutional principles to score.
    """

    def __init__(self, d_model: int, n_principles: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_principles = n_principles
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_principles),
            nn.Sigmoid(),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Compute per-principle harmlessness scores.

        Args:
            hidden_states: (B, T, D) hidden states from a transformer.

        Returns:
            (B, n_principles) scores in (0, 1).
        """
        last = hidden_states[:, -1, :]  # (B, D)
        return self.net(last)  # (B, n_principles)

    def aggregate(self, scores: Tensor, weights: Tensor) -> Tensor:
        """Compute weighted mean score across principles.

        Args:
            scores: (B, n_principles) per-principle scores.
            weights: (n_principles,) principle weights (need not sum to 1).

        Returns:
            (B,) weighted mean score.
        """
        weights = weights.to(dtype=scores.dtype, device=scores.device)
        weight_sum = weights.sum().clamp(min=1e-8)
        return (scores * weights.unsqueeze(0)).sum(dim=-1) / weight_sum


# ---------------------------------------------------------------------------
# RevisionScorer
# ---------------------------------------------------------------------------


class RevisionScorer(nn.Module):
    """Scores the quality of a revision relative to the original response.

    Uses mean-pooled hidden state representations and cosine similarity to
    measure how well the revision aligns with (i.e., is related to) the
    original.

    Args:
        d_model: Hidden state dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def encode(self, hidden_states: Tensor) -> Tensor:
        """Mean-pool hidden states to a single vector per batch item.

        Args:
            hidden_states: (B, T, D) hidden state sequence.

        Returns:
            (B, D) mean-pooled representation.
        """
        return hidden_states.mean(dim=1)

    def score_revision(self, original_hidden: Tensor, revised_hidden: Tensor) -> Tensor:
        """Compute cosine similarity between revised and original encodings.

        Returns cosine_sim(revised, original) as a measure of alignment --
        how related the revision is to the original.

        Args:
            original_hidden: (B, T, D) hidden states for original responses.
            revised_hidden:  (B, T, D) hidden states for revised responses.

        Returns:
            (B,) cosine similarity in [-1, 1].
        """
        orig_enc = self.encode(original_hidden)  # (B, D)
        revised_enc = self.encode(revised_hidden)  # (B, D)
        return F.cosine_similarity(revised_enc, orig_enc, dim=-1)

    def improvement_loss(self, scores: Tensor, target: float = 0.8) -> Tensor:
        """MSE loss toward a target similarity value.

        Encourages revisions that maintain a target level of cosine similarity
        with the original response.

        Args:
            scores: (B,) cosine similarity scores from score_revision.
            target: Desired similarity target (default 0.8).

        Returns:
            Scalar MSE loss.
        """
        target_tensor = torch.full_like(scores, target)
        return F.mse_loss(scores, target_tensor)


# ---------------------------------------------------------------------------
# CAITrainer
# ---------------------------------------------------------------------------


class CAITrainer:
    """Trains a CritiqueHead with constitutional preference data.

    Performs two training operations:
      - critique_step: BCE loss on the CritiqueHead against harmlessness labels.
      - revision_step: Improvement loss on the RevisionScorer.

    Args:
        model: nn.Module callable with signature (input_ids) -> (logits, hidden_states)
               where hidden_states has shape (B, T, D).
        critique_head: CritiqueHead module to train.
        optimizer: PyTorch optimizer that covers critique_head (and optionally model).
        principles: List of ConstitutionalPrinciple objects.
    """

    def __init__(
        self,
        model: nn.Module,
        critique_head: CritiqueHead,
        optimizer: torch.optim.Optimizer,
        principles: list[ConstitutionalPrinciple],
    ) -> None:
        self.model = model
        self.critique_head = critique_head
        self.optimizer = optimizer
        self.principles = principles
        self._revision_scorer = RevisionScorer(critique_head.d_model)

    def critique_step(self, input_ids: Tensor, harmless_labels: Tensor) -> dict:
        """Train the CritiqueHead with BCE loss against harmlessness labels.

        Args:
            input_ids: (B, T) integer token ids.
            harmless_labels: (B,) float harmlessness ground-truth in [0, 1].

        Returns:
            Dict with keys:
                critique_loss (float): scalar BCE loss value.
                mean_score (float): mean predicted harmlessness across batch.
                per_principle_scores (Tensor): (B, n_principles) scores.
        """
        self.model.train()
        self.critique_head.train()
        self.optimizer.zero_grad()

        _logits, hidden_states = self.model(input_ids)
        scores = self.critique_head(hidden_states)  # (B, n_principles)

        # Expand labels to match per-principle scores for BCE loss
        labels_expanded = harmless_labels.unsqueeze(1).expand_as(scores)
        critique_loss = F.binary_cross_entropy(scores, labels_expanded)

        critique_loss.backward()
        self.optimizer.step()

        return {
            "critique_loss": critique_loss.item(),
            "mean_score": scores.detach().mean().item(),
            "per_principle_scores": scores.detach(),
        }

    def revision_step(self, original_ids: Tensor, revised_ids: Tensor) -> dict:
        """Compute revision quality loss toward target cosine similarity 0.8.

        Args:
            original_ids: (B, T) token ids for original responses.
            revised_ids:  (B, T) token ids for revised responses.

        Returns:
            Dict with keys:
                revision_loss (float): scalar MSE loss value.
                similarity_score (float): mean cosine similarity.
        """
        self.model.train()
        self._revision_scorer.train()

        with torch.no_grad():
            _logits_orig, hidden_orig = self.model(original_ids)
            _logits_rev, hidden_rev = self.model(revised_ids)

        # Detach hidden states so we can attach gradient tracking manually
        hidden_orig_d = hidden_orig.detach().requires_grad_(True)
        hidden_rev_d = hidden_rev.detach().requires_grad_(True)

        sim_scores = self._revision_scorer.score_revision(hidden_orig_d, hidden_rev_d)
        revision_loss = self._revision_scorer.improvement_loss(sim_scores, target=0.8)

        return {
            "revision_loss": revision_loss.item(),
            "similarity_score": sim_scores.detach().mean().item(),
        }


# ---------------------------------------------------------------------------
# ConstitutionalFilter
# ---------------------------------------------------------------------------


class ConstitutionalFilter:
    """Inference-time filter and flagging of low-harmlessness outputs.

    Args:
        critique_head: Trained CritiqueHead to score outputs.
        threshold: Scores below this trigger the should_revise flag (default 0.5).
    """

    def __init__(self, critique_head: CritiqueHead, threshold: float = 0.5) -> None:
        self.critique_head = critique_head
        self.threshold = threshold

    def score(self, hidden_states: Tensor, principle_weights: Tensor) -> Tensor:
        """Compute aggregate harmlessness score for each batch item.

        Args:
            hidden_states: (B, T, D) hidden states from a transformer.
            principle_weights: (n_principles,) weights for aggregation.

        Returns:
            (B,) aggregate harmlessness scores in (0, 1).
        """
        self.critique_head.train(False)
        with torch.no_grad():
            per_principle = self.critique_head(hidden_states)
            aggregated = self.critique_head.aggregate(per_principle, principle_weights)
        return aggregated

    def should_revise(self, scores: Tensor) -> Tensor:
        """Return a boolean mask indicating which items need revision.

        Args:
            scores: (B,) harmlessness scores.

        Returns:
            (B,) bool tensor: True if score < threshold.
        """
        return scores < self.threshold

    def revision_priority(self, scores: Tensor) -> torch.LongTensor:
        """Return batch indices sorted from most harmful to least harmful.

        Args:
            scores: (B,) harmlessness scores.

        Returns:
            (B,) LongTensor: argsort ascending (lowest score = most harmful first).
        """
        return torch.argsort(scores, descending=False)
