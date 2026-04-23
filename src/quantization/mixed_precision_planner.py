"""Mixed-precision planner: sensitivity analysis, bit-width assignment, budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LayerSensitivity:
    layer_name: str
    sensitivity_score: float
    recommended_bits: int
    current_bits: int = 16


@dataclass
class MixedPrecisionPlan:
    layer_assignments: dict[str, int]  # layer_name -> bits
    total_bits_saved: int
    compression_ratio: float


class MixedPrecisionPlanner:
    """Plans mixed-precision quantization by sensitivity-driven bit-width assignment."""

    def __init__(
        self,
        target_avg_bits: float = 4.0,
        available_bits: list[int] | None = None,
    ) -> None:
        self.target_avg_bits = target_avg_bits
        self.available_bits = sorted(
            available_bits if available_bits is not None else [2, 4, 8, 16]
        )

    def score_sensitivity(
        self,
        layer_name: str,
        weight_magnitude: float,
        gradient_magnitude: float,
    ) -> LayerSensitivity:
        """Compute sensitivity score and recommend bit-width.

        sensitivity_score = gradient_magnitude / (weight_magnitude + 1e-8)
        score > 1.0  -> 8 bits
        score > 0.1  -> 4 bits
        else         -> 2 bits
        """
        score = gradient_magnitude / (weight_magnitude + 1e-8)
        if score > 1.0:
            recommended_bits = 8
        elif score > 0.1:
            recommended_bits = 4
        else:
            recommended_bits = 2
        return LayerSensitivity(
            layer_name=layer_name,
            sensitivity_score=score,
            recommended_bits=recommended_bits,
        )

    def plan(
        self,
        sensitivities: list[LayerSensitivity],
        n_params_per_layer: dict[str, int],
    ) -> MixedPrecisionPlan:
        """Assign bit-widths respecting target_avg_bits budget.

        Sort by sensitivity descending (most sensitive = more bits).
        Use greedy approach: assign recommended bits, then downgrade
        least-sensitive layers if budget is exceeded.
        """
        if not sensitivities:
            return MixedPrecisionPlan(
                layer_assignments={},
                total_bits_saved=0,
                compression_ratio=1.0,
            )

        # Sort descending by sensitivity (most sensitive first)
        sorted_sens = sorted(sensitivities, key=lambda s: s.sensitivity_score, reverse=True)

        # Initial assignment: recommended bits for each layer
        assignments: dict[str, int] = {}
        for s in sorted_sens:
            # Clamp to available bits (pick closest available)
            assignments[s.layer_name] = self._nearest_available(s.recommended_bits)

        # Enforce budget with greedy downgrade of least-sensitive layers
        assignments = self._enforce_budget(sorted_sens, assignments, n_params_per_layer)

        # Compute total_bits_saved and compression_ratio
        total_bits_saved = 0
        total_original_bits = 0
        total_assigned_bits = 0
        for s in sorted_sens:
            n = n_params_per_layer.get(s.layer_name, 0)
            assigned = assignments[s.layer_name]
            total_bits_saved += (16 - assigned) * n
            total_original_bits += 16 * n
            total_assigned_bits += assigned * n

        if total_assigned_bits > 0:
            compression_ratio = total_original_bits / total_assigned_bits
        else:
            compression_ratio = 1.0

        return MixedPrecisionPlan(
            layer_assignments=assignments,
            total_bits_saved=max(0, total_bits_saved),
            compression_ratio=compression_ratio,
        )

    def validate_plan(
        self,
        plan: MixedPrecisionPlan,
        target_avg_bits: float,
        n_params: dict[str, int],
    ) -> bool:
        """Check that the weighted average bits <= target_avg_bits + 0.5 tolerance."""
        total_weighted_bits = 0
        total_params = 0
        for layer_name, bits in plan.layer_assignments.items():
            n = n_params.get(layer_name, 0)
            total_weighted_bits += bits * n
            total_params += n
        if total_params == 0:
            return True
        avg_bits = total_weighted_bits / total_params
        return avg_bits <= target_avg_bits + 0.5

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _nearest_available(self, bits: int) -> int:
        """Return the closest bit-width in available_bits."""
        if not self.available_bits:
            return bits
        return min(self.available_bits, key=lambda b: abs(b - bits))

    def _weighted_avg_bits(
        self,
        assignments: dict[str, int],
        n_params_per_layer: dict[str, int],
    ) -> float:
        total_bits = 0
        total_params = 0
        for name, bits in assignments.items():
            n = n_params_per_layer.get(name, 0)
            total_bits += bits * n
            total_params += n
        if total_params == 0:
            return 0.0
        return total_bits / total_params

    def _enforce_budget(
        self,
        sorted_sens: list[LayerSensitivity],
        assignments: dict[str, int],
        n_params_per_layer: dict[str, int],
    ) -> dict[str, int]:
        """Greedily downgrade least-sensitive layers until budget is met."""
        # Work on a copy
        result = dict(assignments)

        # Iterate from least sensitive to most sensitive, trying to downgrade
        for s in reversed(sorted_sens):
            avg = self._weighted_avg_bits(result, n_params_per_layer)
            if avg <= self.target_avg_bits:
                break
            # Try to downgrade this layer
            current = result[s.layer_name]
            # Find available bits strictly less than current
            lower_options = [b for b in self.available_bits if b < current]
            if lower_options:
                result[s.layer_name] = max(lower_options)

        return result
