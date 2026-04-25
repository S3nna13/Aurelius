"""Gradient checkpointing manager: segment selection, memory estimation, recompute policy."""

from dataclasses import dataclass, field
from enum import Enum


class CheckpointPolicy(str, Enum):
    NONE = "none"
    EVERY_LAYER = "every_layer"
    EVERY_K_LAYERS = "every_k_layers"
    CUSTOM = "custom"


@dataclass
class CheckpointConfig:
    policy: CheckpointPolicy = CheckpointPolicy.EVERY_K_LAYERS
    k: int = 2
    memory_budget_gb: float = 8.0


class GradientCheckpointManager:
    """Selects which layers use gradient checkpointing and estimates cost/savings."""

    def __init__(self, n_layers: int, config: CheckpointConfig | None = None) -> None:
        self._n_layers = n_layers
        self._config = config if config is not None else CheckpointConfig()
        self._custom_layers: list[int] = []

    # ------------------------------------------------------------------
    def checkpointed_layers(self) -> list[int]:
        """Return list of layer indices that are gradient-checkpointed."""
        policy = self._config.policy
        n = self._n_layers

        if policy == CheckpointPolicy.NONE:
            return []
        elif policy == CheckpointPolicy.EVERY_LAYER:
            return list(range(n))
        elif policy == CheckpointPolicy.EVERY_K_LAYERS:
            k = max(1, self._config.k)
            return list(range(0, n, k))
        elif policy == CheckpointPolicy.CUSTOM:
            return list(self._custom_layers)
        return []

    def set_custom_layers(self, layers: list[int]) -> None:
        """Set the custom layer list (used when policy is CUSTOM)."""
        self._custom_layers = list(layers)

    def memory_savings_estimate(self, n_layers: int, layer_mem_gb: float = 0.5) -> float:
        """Estimate memory saved (GB) by checkpointing.

        Checkpointed layers recompute activations instead of storing them,
        saving layer_mem_gb per checkpointed layer.
        """
        checkpointed_count = len(self.checkpointed_layers())
        return checkpointed_count * layer_mem_gb

    def recompute_cost_estimate(
        self, n_checkpointed: int, base_forward_cost: float = 1.0
    ) -> float:
        """Estimate extra compute cost from recomputation.

        Each checkpointed layer adds one extra forward pass.
        """
        return n_checkpointed * base_forward_cost
