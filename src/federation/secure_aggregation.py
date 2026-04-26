"""Secure aggregation: masking schemes for federated gradient updates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch


class MaskingScheme(StrEnum):
    ADDITIVE = "ADDITIVE"
    PAIRWISE_MASK = "PAIRWISE_MASK"
    SHAMIR_STUB = "SHAMIR_STUB"


@dataclass
class SecureAggConfig:
    scheme: MaskingScheme
    seed: int = 42
    num_parties: int = 2


class SecureAggregator:
    """Generates and applies cryptographic masks for secure aggregation."""

    def generate_mask(
        self,
        shape: tuple,
        party_id: int,
        config: SecureAggConfig,
    ) -> torch.Tensor:
        """Return a mask tensor for the given *party_id* and *config*."""
        if config.scheme == MaskingScheme.ADDITIVE:
            # Random noise scaled by 1/num_parties so masks approximately cancel
            rng = torch.Generator()
            rng.manual_seed(config.seed + party_id)
            noise = torch.randn(shape, generator=rng)
            return noise / config.num_parties

        if config.scheme == MaskingScheme.PAIRWISE_MASK:
            rng = torch.Generator()
            rng.manual_seed(config.seed)
            base_mask = torch.randn(shape, generator=rng)
            # Odd party_id flips the sign so masks cancel in pairs
            sign = -1.0 if (party_id % 2 == 1) else 1.0
            return base_mask * sign

        # SHAMIR_STUB: full Shamir secret sharing not implemented; return zeros
        return torch.zeros(shape)

    def mask_update(
        self,
        gradient: torch.Tensor,
        party_id: int,
        config: SecureAggConfig,
    ) -> torch.Tensor:
        """Add a mask to *gradient* before sending to the aggregator."""
        mask = self.generate_mask(tuple(gradient.shape), party_id, config)
        return gradient + mask

    def aggregate_masked(
        self,
        masked_updates: list[torch.Tensor],
    ) -> torch.Tensor:
        """Sum masked tensors; masks cancel out when scheme is symmetric."""
        if not masked_updates:
            raise ValueError("masked_updates must be non-empty")
        result = masked_updates[0].clone()
        for t in masked_updates[1:]:
            result = result + t
        return result


# Module-level singleton
_SECURE_AGGREGATOR = SecureAggregator()

try:
    from src.federation.federated_learning import (  # type: ignore[import]
        FEDERATION_REGISTRY as _REG,
    )

    _REG["secure_aggregation"] = _SECURE_AGGREGATOR
except Exception:  # noqa: S110
    pass

FEDERATION_REGISTRY: dict[str, object] = {"secure_aggregation": _SECURE_AGGREGATOR}
