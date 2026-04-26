"""Benchmark framework for comparing activation steering methods.

Evaluates steering vectors by measuring:
  - Concept shift: how much steering changes logit probabilities toward target
  - Fluency preservation: KL divergence from unsteered distribution (should be low)
  - Steering efficiency: concept shift per unit of steering magnitude
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


class SteeringTarget:
    """Represents a concept to steer toward/away from.

    Args:
        name: Human-readable name for the concept.
        target_token_ids: Token IDs associated with the target concept (e.g., tokens for "positive").
        anti_target_token_ids: Token IDs for the opposite concept.
    """  # noqa: E501

    def __init__(
        self,
        name: str,
        target_token_ids: list[int],
        anti_target_token_ids: list[int],
    ) -> None:
        self.name = name
        self.target_token_ids = target_token_ids
        self.anti_target_token_ids = anti_target_token_ids


@dataclass
class SteeringResult:
    """Result of evaluating a steering vector against a concept target.

    Attributes:
        method_name: Name of the steering method.
        concept_shift: Mean increase in target token probability under steering.
        fluency_kl: KL divergence between steered and unsteered distributions.
        steering_efficiency: Concept shift per unit of steering magnitude.
        steering_norm: L2 norm of the steering vector.
    """

    method_name: str
    concept_shift: float
    fluency_kl: float
    steering_efficiency: float
    steering_norm: float


class SteeringEvaluator:
    """Evaluates steering vectors on a given model function.

    Args:
        model_fn: Callable mapping (1, T, d_model) -> (1, T, vocab_size).
        vocab_size: Size of the vocabulary.
    """

    def __init__(self, model_fn: Callable[[Tensor], Tensor], vocab_size: int) -> None:
        self.model_fn = model_fn
        self.vocab_size = vocab_size

    def evaluate_steering(
        self,
        hidden_states: Tensor,
        steering_vector: Tensor,
        target: SteeringTarget,
        alpha: float = 1.0,
    ) -> SteeringResult:
        """Evaluate a steering vector against a target concept.

        Args:
            hidden_states: ``(1, T, d_model)`` — baseline hidden states.
            steering_vector: ``(d_model,)`` — the direction to steer in.
            target: SteeringTarget defining which tokens to measure.
            alpha: Scalar multiplier for the steering vector.

        Returns:
            SteeringResult with all metrics computed.
        """
        # Step 1: Baseline logits at last token position
        logits_base_full = self.model_fn(hidden_states)  # (1, T, V)
        logits_base = logits_base_full[0, -1, :]  # (V,)
        probs_base = F.softmax(logits_base, dim=-1)  # (V,)

        # Step 2: Apply steering
        steered = hidden_states + alpha * steering_vector[None, None, :]  # (1, T, d_model)

        # Step 3: Steered logits at last token position
        logits_steered_full = self.model_fn(steered)  # (1, T, V)
        logits_steered = logits_steered_full[0, -1, :]  # (V,)
        probs_steered = F.softmax(logits_steered, dim=-1)  # (V,)

        # Step 4: Concept shift — mean probability increase on target tokens
        target_ids = target.target_token_ids
        concept_shift = (probs_steered[target_ids].mean() - probs_base[target_ids].mean()).item()

        # Step 5: Fluency KL — KL divergence from steered to base
        log_probs_steered = torch.log(probs_steered + 1e-10)
        fluency_kl = F.kl_div(log_probs_steered, probs_base, reduction="sum").item()

        # Step 6: Steering norm
        steering_norm = steering_vector.norm().item()

        # Step 7: Steering efficiency
        steering_efficiency = concept_shift / (steering_norm + 1e-8)

        return SteeringResult(
            method_name="",
            concept_shift=concept_shift,
            fluency_kl=fluency_kl,
            steering_efficiency=steering_efficiency,
            steering_norm=steering_norm,
        )


class SteeringComparison:
    """Compares multiple steering methods across multiple targets.

    Stateless — all state is passed in via arguments.
    """

    def __init__(self) -> None:
        pass

    def compare(self, results: list[SteeringResult]) -> dict[str, dict[str, float]]:
        """Group results by method_name and compute per-method averages.

        Args:
            results: List of SteeringResult objects from possibly multiple methods.

        Returns:
            Dict mapping method_name -> dict with keys:
                'mean_concept_shift', 'mean_fluency_kl', 'mean_efficiency', 'n_results'.
        """
        grouped: dict[str, list[SteeringResult]] = {}
        for r in results:
            grouped.setdefault(r.method_name, []).append(r)

        summary: dict[str, dict[str, float]] = {}
        for method, method_results in grouped.items():
            n = len(method_results)
            mean_concept_shift = sum(r.concept_shift for r in method_results) / n
            mean_fluency_kl = sum(r.fluency_kl for r in method_results) / n
            mean_efficiency = sum(r.steering_efficiency for r in method_results) / n
            summary[method] = {
                "mean_concept_shift": mean_concept_shift,
                "mean_fluency_kl": mean_fluency_kl,
                "mean_efficiency": mean_efficiency,
                "n_results": float(n),
            }
        return summary

    def rank_by(
        self,
        results: list[SteeringResult],
        metric: str = "steering_efficiency",
    ) -> list[SteeringResult]:
        """Sort results by the given metric, descending (highest first).

        Args:
            results: List of SteeringResult objects.
            metric: Attribute name on SteeringResult to sort by.

        Returns:
            Sorted list (descending order by metric value).
        """
        return sorted(results, key=lambda r: getattr(r, metric), reverse=True)


class SteeringVectorNormalizer:
    """Utility for normalizing and transforming steering vectors.

    Stateless — all operations are pure tensor transformations.
    """

    def __init__(self) -> None:
        pass

    def normalize(self, vector: Tensor) -> Tensor:
        """Unit-normalize a steering vector.

        Args:
            vector: ``(d_model,)`` or any shape tensor.

        Returns:
            Tensor of same shape with unit L2 norm.
        """
        return vector / (vector.norm() + 1e-8)

    def scale_to_norm(self, vector: Tensor, target_norm: float) -> Tensor:
        """Normalize then scale a vector to have the given L2 norm.

        Args:
            vector: Tensor to rescale.
            target_norm: Desired L2 norm.

        Returns:
            Tensor with L2 norm equal to ``target_norm``.
        """
        return self.normalize(vector) * target_norm

    def project_out(self, vector: Tensor, basis: Tensor) -> Tensor:
        """Remove the component of ``vector`` along ``basis``.

        Computes ``v - (v · b̂) * b̂`` where ``b̂`` is the unit-normalized basis.

        Args:
            vector: ``(d_model,)`` vector to project.
            basis: ``(d_model,)`` basis direction.

        Returns:
            ``vector`` with its component along ``basis`` removed.
        """
        basis_hat = self.normalize(basis)
        projection = torch.dot(vector, basis_hat) * basis_hat
        return vector - projection
