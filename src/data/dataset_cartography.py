"""Dataset Cartography — training dynamics for data characterization.

Implements Swayamdipta et al. 2020: characterizing examples as easy-to-learn,
ambiguous, or hard-to-learn based on per-epoch confidence and correctness.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict

import torch
from torch import Tensor


@dataclass
class CartographyConfig:
    n_epochs: int = 3       # epochs to track dynamics
    smoothing: float = 0.0  # label smoothing for confidence


@dataclass
class SampleDynamics:
    """Training dynamics for a single example across epochs."""

    sample_idx: int
    confidences: List[float]   # per-epoch probability of correct label
    correctness: List[bool]    # per-epoch correct prediction

    @property
    def mean_confidence(self) -> float:
        """Mean confidence across epochs."""
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)

    @property
    def variability(self) -> float:
        """Std of confidence across epochs — measures training instability."""
        n = len(self.confidences)
        if n < 2:
            return 0.0
        mean = self.mean_confidence
        variance = sum((c - mean) ** 2 for c in self.confidences) / (n - 1)
        return math.sqrt(variance)

    @property
    def correctness_rate(self) -> float:
        """Fraction of epochs where prediction was correct."""
        if not self.correctness:
            return 0.0
        return sum(1 for c in self.correctness if c) / len(self.correctness)

    @property
    def region(self) -> str:
        """Classify into: 'easy-to-learn', 'ambiguous', 'hard-to-learn'.

        - easy-to-learn: high confidence (> 0.7), low variability (<= 0.2)
        - ambiguous:     high variability (> 0.2)
        - hard-to-learn: low confidence (<= 0.7), low variability (<= 0.2)

        Thresholds: confidence > 0.7 → easy; variability > 0.2 → ambiguous; else hard.
        """
        if self.variability > 0.2:
            return "ambiguous"
        if self.mean_confidence > 0.7:
            return "easy-to-learn"
        return "hard-to-learn"


class CartographyTracker:
    """Track per-sample training dynamics across epochs."""

    def __init__(self, n_samples: int, config: CartographyConfig):
        self.n_samples = n_samples
        self.config = config
        self.dynamics: Dict[int, SampleDynamics] = {}

    def record_epoch(
        self,
        sample_indices: List[int],   # which samples were seen
        logits: Tensor,              # (N, n_classes) predicted logits
        labels: Tensor,              # (N,) true class labels
    ) -> None:
        """Record confidence and correctness for each sample this epoch."""
        probs = torch.softmax(logits, dim=-1)          # (N, n_classes)
        preds = logits.argmax(dim=-1)                   # (N,)

        for i, idx in enumerate(sample_indices):
            true_label = int(labels[i].item())
            confidence = float(probs[i, true_label].item())
            correct = bool(preds[i].item() == true_label)

            if idx not in self.dynamics:
                self.dynamics[idx] = SampleDynamics(
                    sample_idx=idx,
                    confidences=[],
                    correctness=[],
                )
            self.dynamics[idx].confidences.append(confidence)
            self.dynamics[idx].correctness.append(correct)

    def get_dynamics(self, idx: int) -> SampleDynamics:
        """Get dynamics for a specific sample."""
        return self.dynamics[idx]

    def get_all_dynamics(self) -> List[SampleDynamics]:
        """Return all tracked sample dynamics."""
        return list(self.dynamics.values())

    def select_by_region(self, region: str) -> List[int]:
        """Return sample indices in the given region."""
        return [
            d.sample_idx
            for d in self.dynamics.values()
            if d.region == region
        ]

    def cartography_summary(self) -> Dict[str, int]:
        """Count samples per region: {'easy-to-learn': N, 'ambiguous': N, 'hard-to-learn': N}"""
        summary: Dict[str, int] = {
            "easy-to-learn": 0,
            "ambiguous": 0,
            "hard-to-learn": 0,
        }
        for d in self.dynamics.values():
            summary[d.region] += 1
        return summary


def select_training_subset(
    tracker: CartographyTracker,
    strategy: str = "ambiguous",   # "ambiguous", "easy-to-learn", "hard-to-learn"
    fraction: float = 0.33,
) -> List[int]:
    """Select a fraction of training data based on cartography region.

    Returns list of sample indices. If the target region has more samples than
    ``fraction * total``, the list is truncated to that budget.
    """
    candidates = tracker.select_by_region(strategy)
    total = len(tracker.dynamics)
    budget = max(1, int(fraction * total))
    return candidates[:budget]


def compute_forgetting_events(dynamics_list: List[SampleDynamics]) -> Dict[int, int]:
    """Count forgetting events per sample: transitions from correct→incorrect.

    Returns {sample_idx: n_forgetting_events}.
    """
    result: Dict[int, int] = {}
    for d in dynamics_list:
        events = 0
        for t in range(1, len(d.correctness)):
            if d.correctness[t - 1] and not d.correctness[t]:
                events += 1
        result[d.sample_idx] = events
    return result
