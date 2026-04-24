"""Staleness handler: policies for accepting/weighting delayed client updates."""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import torch


class StalenessPolicy(str, Enum):
    DISCARD = "DISCARD"
    DECAY = "DECAY"
    BOUNDED_DELAY = "BOUNDED_DELAY"
    ALWAYS_ACCEPT = "ALWAYS_ACCEPT"


@dataclass
class StalenessConfig:
    policy: StalenessPolicy
    max_staleness: int = 5
    decay_factor: float = 0.9


@dataclass
class ClientUpdate:
    client_id: str
    round_number: int
    gradient: torch.Tensor
    received_at: float


class StalenessHandler:
    """Evaluates and filters delayed client updates."""

    def __init__(self) -> None:
        self.current_round: int = 0

    def advance_round(self) -> None:
        """Increment the server's current round counter."""
        self.current_round += 1

    def evaluate(
        self,
        update: ClientUpdate,
        config: StalenessConfig,
    ) -> tuple[bool, float]:
        """Return (accepted, weight) for a single update.

        staleness = current_round - update.round_number
        """
        staleness = self.current_round - update.round_number

        if config.policy == StalenessPolicy.DISCARD:
            accepted = staleness <= config.max_staleness
            return accepted, 1.0 if accepted else 0.0

        if config.policy == StalenessPolicy.DECAY:
            weight = config.decay_factor ** staleness
            return True, weight

        if config.policy == StalenessPolicy.BOUNDED_DELAY:
            if staleness > config.max_staleness:
                return False, 0.0
            weight = 1.0 / (1.0 + staleness)
            return True, weight

        # ALWAYS_ACCEPT
        return True, 1.0

    def filter_updates(
        self,
        updates: list[ClientUpdate],
        config: StalenessConfig,
    ) -> list[tuple[ClientUpdate, float]]:
        """Return accepted (update, weight) pairs."""
        result: list[tuple[ClientUpdate, float]] = []
        for upd in updates:
            accepted, weight = self.evaluate(upd, config)
            if accepted:
                result.append((upd, weight))
        return result


# Module-level singleton
_STALENESS_HANDLER = StalenessHandler()

try:
    from src.federation.federated_learning import (  # type: ignore[import]
        FEDERATION_REGISTRY as _REG,
    )
    _REG["staleness_handler"] = _STALENESS_HANDLER
except Exception:
    pass

FEDERATION_REGISTRY: dict[str, object] = {"staleness_handler": _STALENESS_HANDLER}
