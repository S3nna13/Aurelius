"""
Concept Bottleneck Model (CBM) for the Aurelius LLM project.

Implements the two-stage interpretable architecture from Koh et al. 2020:
  Stage 1: hidden states -> concept activations (interpretable bottleneck)
  Stage 2: concept activations -> task logits

Also provides utilities for concept intervention and gradient-based importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CBMConfig:
    """Configuration for Concept Bottleneck Model."""
    d_model: int = 64          # input hidden dim
    n_concepts: int = 16       # number of interpretable concepts
    n_classes: int = 2         # downstream task classes
    dropout: float = 0.1
    concept_activation: str = "sigmoid"  # "sigmoid" or "relu"


# ---------------------------------------------------------------------------
# Concept Layer
# ---------------------------------------------------------------------------

class ConceptLayer(nn.Module):
    """Projects hidden states to concept activations (B, T, n_concepts)."""

    def __init__(self, config: CBMConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.d_model, config.n_concepts)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = config.concept_activation

    def forward(self, hidden: Tensor) -> Tensor:
        """
        Args:
            hidden: (B, T, d_model) hidden state tensor

        Returns:
            concept_activations: (B, T, n_concepts) in [0, 1] for sigmoid
        """
        x = self.dropout(hidden)
        x = self.linear(x)
        if self.activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.activation == "relu":
            x = F.relu(x)
        return x


# ---------------------------------------------------------------------------
# Concept Bottleneck Model
# ---------------------------------------------------------------------------

class ConceptBottleneckModel(nn.Module):
    """Two-stage: hidden -> concepts -> task logits.

    Stage 1 (concept prediction): hidden -> concept_activations in [0, 1]
    Stage 2 (task prediction):    concept_activations -> class_logits
    """

    def __init__(self, config: CBMConfig) -> None:
        super().__init__()
        self.concept_layer = ConceptLayer(config)
        self.task_layer = nn.Linear(config.n_concepts, config.n_classes)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            hidden: (B, T, d_model) hidden state tensor

        Returns:
            concept_activations: (B, T, n_concepts)
            class_logits:        (B, T, n_classes)
        """
        concept_activations = self.concept_layer(hidden)
        class_logits = self.task_layer(self.dropout(concept_activations))
        return concept_activations, class_logits


# ---------------------------------------------------------------------------
# Concept Alignment Score
# ---------------------------------------------------------------------------

def concept_alignment_score(
    concept_activations: Tensor,   # (B, n_concepts)
    ground_truth_concepts: Tensor, # (B, n_concepts) binary
) -> Tensor:
    """Mean per-concept AUC approximation using ranking correlation.

    For each concept, computes the Wilcoxon-Mann-Whitney statistic
    (proportion of positive-negative pairs ranked correctly) as an AUC
    approximation, then averages over concepts.

    Args:
        concept_activations:   (B, n_concepts) predicted scores
        ground_truth_concepts: (B, n_concepts) binary ground truth

    Returns:
        Scalar tensor in [0, 1].
    """
    B, n_concepts = concept_activations.shape
    auc_scores = torch.zeros(n_concepts, device=concept_activations.device,
                             dtype=concept_activations.dtype)

    for c in range(n_concepts):
        scores = concept_activations[:, c]  # (B,)
        labels = ground_truth_concepts[:, c]  # (B,) binary

        pos_mask = labels > 0.5
        neg_mask = ~pos_mask

        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()

        if n_pos == 0 or n_neg == 0:
            # No discrimination possible; use 0.5 as random baseline
            auc_scores[c] = 0.5
            continue

        pos_scores = scores[pos_mask]  # (n_pos,)
        neg_scores = scores[neg_mask]  # (n_neg,)

        # AUC = proportion of (pos, neg) pairs where pos_score > neg_score
        # Compute via broadcasting: (n_pos, n_neg) comparison matrix
        pos_scores_expanded = pos_scores.unsqueeze(1)  # (n_pos, 1)
        neg_scores_expanded = neg_scores.unsqueeze(0)  # (1, n_neg)

        correct = (pos_scores_expanded > neg_scores_expanded).float()
        ties = (pos_scores_expanded == neg_scores_expanded).float() * 0.5
        auc_scores[c] = (correct + ties).mean()

    return auc_scores.mean()


# ---------------------------------------------------------------------------
# Concept Intervention
# ---------------------------------------------------------------------------

def concept_intervention(
    model: ConceptBottleneckModel,
    hidden: Tensor,              # (B, T, d_model)
    concept_idx: int,
    intervention_value: float,   # value to set concept to (0.0 or 1.0)
) -> Tensor:
    """Apply test-time concept intervention: fix concept_idx to intervention_value,
    recompute task logits. Returns new class_logits (B, T, n_classes).

    Args:
        model:              Trained ConceptBottleneckModel.
        hidden:             (B, T, d_model) input hidden states.
        concept_idx:        Index of the concept to intervene on.
        intervention_value: Value to force the concept to (typically 0.0 or 1.0).

    Returns:
        class_logits: (B, T, n_classes) with concept_idx fixed to intervention_value.
    """
    model.eval()
    with torch.no_grad():
        concept_activations, _ = model(hidden)
        # Clone to avoid in-place modification of the original
        intervened = concept_activations.clone()
        intervened[..., concept_idx] = intervention_value
        class_logits = model.task_layer(intervened)
    return class_logits


# ---------------------------------------------------------------------------
# Concept Importance
# ---------------------------------------------------------------------------

def extract_concept_importance(
    model: ConceptBottleneckModel,
    hidden: Tensor,              # (B, T, d_model)
) -> Tensor:
    """Gradient-based concept importance: d(task_logits)/d(concepts).

    Computes the mean absolute gradient of the sum of task logits with
    respect to the concept activations, averaged over batch and sequence
    dimensions.

    Args:
        model:  Trained ConceptBottleneckModel.
        hidden: (B, T, d_model) input hidden states.

    Returns:
        importance: (n_concepts,) importance scores, all finite and >= 0.
    """
    model.eval()

    # Forward through concept layer to get activations that require grad
    concept_activations = model.concept_layer(hidden)
    concept_activations.retain_grad()

    # Detach from hidden-state graph; re-attach through concept_activations only
    # so gradients flow back through task_layer to concept_activations
    class_logits = model.task_layer(concept_activations)

    # Scalar objective: sum of all class logits
    objective = class_logits.sum()
    objective.backward()

    if concept_activations.grad is None:
        # Fallback: use task layer weights as proxy
        importance = model.task_layer.weight.abs().mean(dim=0)
    else:
        # Mean absolute gradient over (B, T) dimensions -> (n_concepts,)
        importance = concept_activations.grad.abs().mean(dim=(0, 1))

    return importance.detach()
