"""
concept_bottleneck.py — Concept Bottleneck Model (Koh et al. 2020).

Maps activations to human-interpretable concepts.
Pure Python, stdlib-only. No torch dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Concept:
    """A human-interpretable concept with a name, description, and unique id."""
    name: str
    description: str
    concept_id: int


@dataclass(frozen=True)
class ConceptScore:
    """Score and activation status for a single concept."""
    concept: Concept
    score: float
    active: bool


class ConceptBottleneck:
    """Maps activations to concepts via linear probes.

    Implements the concept bottleneck layer from Koh et al. 2020 using
    stdlib-only arithmetic.
    """

    def __init__(self, concepts: List[Concept], threshold: float = 0.5) -> None:
        """
        Args:
            concepts: List of Concept objects to track.
            threshold: Score threshold above which a concept is considered active.
        """
        self.concepts = concepts
        self.threshold = threshold
        # concept_id -> probe weights (list of floats)
        self._probes: Dict[int, List[float]] = {}

    def register_probe(self, concept_id: int, weights: List[float]) -> None:
        """Register a linear probe for a concept.

        Args:
            concept_id: ID of the concept this probe belongs to.
            weights: Linear probe weight vector (same dim as activations).
        """
        self._probes[concept_id] = list(weights)

    def score_concept(self, concept_id: int, activations: List[float]) -> float:
        """Score a single concept given activations via a linear probe + sigmoid.

        The dot product of probe weights and activations is passed through a
        sigmoid function: 1 / (1 + exp(-clamp(x, -500, 500))).

        Args:
            concept_id: ID of the concept to score.
            activations: Activation vector (list of floats).

        Returns:
            Float in [0, 1]. Returns 0.0 if probe is not registered.
        """
        weights = self._probes.get(concept_id)
        if weights is None:
            return 0.0
        x = sum(w * a for w, a in zip(weights, activations))
        # Sigmoid with clamping for numerical stability
        x_clamped = max(-500.0, min(500.0, x))
        return 1.0 / (1.0 + math.exp(-x_clamped))

    def predict(self, activations: List[float]) -> List[ConceptScore]:
        """Score all registered concepts and determine which are active.

        Args:
            activations: Activation vector (list of floats).

        Returns:
            List of ConceptScore (one per concept with a registered probe).
        """
        results: List[ConceptScore] = []
        for concept in self.concepts:
            if concept.concept_id in self._probes:
                score = self.score_concept(concept.concept_id, activations)
                active = score >= self.threshold
                results.append(ConceptScore(concept=concept, score=score, active=active))
        return results

    def active_concepts(self, activations: List[float]) -> List[Concept]:
        """Return only the concepts that are active (score >= threshold).

        Args:
            activations: Activation vector (list of floats).

        Returns:
            List of Concept objects whose score is >= threshold.
        """
        return [cs.concept for cs in self.predict(activations) if cs.active]

    def intervention(
        self,
        concept_id: int,
        target_score: float,
        activations: List[float],
    ) -> List[float]:
        """Stub for counterfactual concept intervention.

        Returns a copy of the activations unchanged. In a full implementation
        this would modify the activation space to achieve the target_score for
        the specified concept.

        Args:
            concept_id: The concept to intervene on.
            target_score: Desired score for the concept (unused in this stub).
            activations: Input activation vector.

        Returns:
            Copy of activations (unchanged).
        """
        return list(activations)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONCEPT_BOTTLENECK_REGISTRY = {
    "default": ConceptBottleneck,
}
