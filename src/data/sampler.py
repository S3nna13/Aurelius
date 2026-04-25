"""Data sampler with stratified and weighted sampling strategies."""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class StratifiedSampler:
    """Sample data maintaining class distribution."""

    labels: list[int]
    _by_class: dict[int, list[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for i, label in enumerate(self.labels):
            self._by_class.setdefault(label, []).append(i)

    def sample(self, count: int, seed: int | None = None) -> list[int]:
        rng = random.Random(seed)
        classes = sorted(self._by_class.keys())
        total = len(self.labels)
        sampled = []
        for cls in classes:
            cls_indices = self._by_class[cls]
            cls_count = max(1, int(count * len(cls_indices) / total))
            sampled.extend(rng.sample(cls_indices, min(cls_count, len(cls_indices))))
        return sampled[:count]


STRATIFIED_SAMPLER = None  # instantiate with your labels