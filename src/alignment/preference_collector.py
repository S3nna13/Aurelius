"""Preference data collector: pairwise comparisons, preference dataset builder."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field


@dataclass
class PreferenceItem:
    prompt: str
    chosen: str
    rejected: str
    annotator_id: str = "auto"
    score_chosen: float = 1.0
    score_rejected: float = 0.0
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class PreferenceCollector:
    """Collects and manages pairwise preference data."""

    def __init__(self, max_items: int = 10000) -> None:
        self._max_items = max_items
        self._items: list[PreferenceItem] = []
        self._id_map: dict[str, PreferenceItem] = {}

    def add(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        annotator_id: str = "auto",
    ) -> PreferenceItem:
        """Add a preference pair and return the created PreferenceItem."""
        item = PreferenceItem(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            annotator_id=annotator_id,
        )
        # Evict oldest item if at capacity
        if len(self._items) >= self._max_items:
            evicted = self._items.pop(0)
            self._id_map.pop(evicted.id, None)
        self._items.append(item)
        self._id_map[item.id] = item
        return item

    def get(self, item_id: str) -> PreferenceItem | None:
        """Retrieve an item by id, or None if not found."""
        return self._id_map.get(item_id)

    def sample(self, n: int, seed: int = 42) -> list[PreferenceItem]:
        """Random sample without replacement (returns all if n >= len)."""
        rng = random.Random(seed)
        if n >= len(self._items):
            result = list(self._items)
            rng.shuffle(result)
            return result
        return rng.sample(self._items, n)

    def to_chatml_pairs(self) -> list[dict]:
        """Convert each item to ChatML-style prompt/chosen/rejected dicts."""
        result = []
        for item in self._items:
            result.append({
                "prompt": item.prompt,
                "chosen": [{"role": "assistant", "content": item.chosen}],
                "rejected": [{"role": "assistant", "content": item.rejected}],
            })
        return result

    def filter_by_annotator(self, annotator_id: str) -> list[PreferenceItem]:
        """Return all items with the given annotator_id."""
        return [item for item in self._items if item.annotator_id == annotator_id]

    def stats(self) -> dict:
        """Return summary statistics about the collected preferences."""
        total = len(self._items)
        annotators = list({item.annotator_id for item in self._items})
        if total == 0:
            avg_score_gap = 0.0
        else:
            avg_score_gap = sum(
                item.score_chosen - item.score_rejected for item in self._items
            ) / total
        return {
            "total": total,
            "annotators": annotators,
            "avg_score_gap": avg_score_gap,
        }


PREFERENCE_COLLECTOR = PreferenceCollector()
