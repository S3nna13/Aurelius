"""Cross-silo model comparison for federated learning."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SiloComparator:
    """Compare model weights across federated silos."""

    _snapshots: dict[str, dict] = field(default_factory=dict, repr=False)

    def snapshot(self, silo_id: str, weights: dict) -> None:
        self._snapshots[silo_id] = weights

    def cosine_similarity(self, silo_a: str, silo_b: str) -> float:
        import numpy as np

        wa = self._snapshots.get(silo_a, {})
        wb = self._snapshots.get(silo_b, {})
        if not wa or not wb:
            return 0.0
        common = [k for k in wa if k in wb and wa[k].shape == wb[k].shape]
        if not common:
            return 0.0
        dot = sum(np.sum(wa[k] * wb[k]) for k in common)
        na = sum(np.sum(wa[k] ** 2) for k in common)
        nb = sum(np.sum(wb[k] ** 2) for k in common)
        denom = (na**0.5) * (nb**0.5)
        return float(dot / denom) if denom > 0 else 0.0

    def weight_divergence(self, silo_a: str, silo_b: str) -> float:
        import numpy as np

        wa = self._snapshots.get(silo_a, {})
        wb = self._snapshots.get(silo_b, {})
        if not wa or not wb:
            return float("inf")
        common = [k for k in wa if k in wb and wa[k].shape == wb[k].shape]
        if not common:
            return float("inf")
        total = sum(np.sum(np.abs(wa[k] - wb[k])) for k in common)
        return float(total)


SILO_COMPARATOR = SiloComparator()
