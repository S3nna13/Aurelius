"""Aurelius leaderboard tracker: tracks model evaluation results on a leaderboard."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LeaderboardEntry:
    model_id: str
    metric_name: str
    score: float
    timestamp: float
    metadata: dict = field(default_factory=dict)


class Leaderboard:
    """Tracks model evaluation results and produces sorted rankings."""

    def __init__(self, metric_name: str, higher_is_better: bool = True) -> None:
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better
        self._entries: list[LeaderboardEntry] = []

    def submit(
        self,
        model_id: str,
        score: float,
        metadata: dict | None = None,
    ) -> LeaderboardEntry:
        """Record a new score for model_id. Returns the created LeaderboardEntry."""
        entry = LeaderboardEntry(
            model_id=model_id,
            metric_name=self.metric_name,
            score=score,
            timestamp=time.monotonic(),
            metadata=metadata if metadata is not None else {},
        )
        self._entries.append(entry)
        return entry

    def rankings(self) -> list[LeaderboardEntry]:
        """Return all entries sorted by score (desc if higher_is_better, asc otherwise).

        Tie-break by timestamp ascending (earlier submission wins).
        Only the best submission per model is included.
        """
        # Collect best entry per model
        best_per_model: dict[str, LeaderboardEntry] = {}
        for entry in self._entries:
            prev = best_per_model.get(entry.model_id)
            if prev is None:
                best_per_model[entry.model_id] = entry
            else:
                if self.higher_is_better:
                    if entry.score > prev.score or (
                        entry.score == prev.score and entry.timestamp < prev.timestamp
                    ):
                        best_per_model[entry.model_id] = entry
                else:
                    if entry.score < prev.score or (
                        entry.score == prev.score and entry.timestamp < prev.timestamp
                    ):
                        best_per_model[entry.model_id] = entry

        entries = list(best_per_model.values())
        reverse = self.higher_is_better
        entries.sort(key=lambda e: (e.score * (-1 if reverse else 1), e.timestamp))
        return entries

    def best(self) -> LeaderboardEntry | None:
        """Return the top-ranked entry, or None if empty."""
        ranked = self.rankings()
        return ranked[0] if ranked else None

    def rank_of(self, model_id: str) -> int | None:
        """Return the 1-indexed rank of model_id, or None if not submitted."""
        ranked = self.rankings()
        for idx, entry in enumerate(ranked, start=1):
            if entry.model_id == model_id:
                return idx
        return None

    def history_of(self, model_id: str) -> list[LeaderboardEntry]:
        """Return all entries submitted for model_id, sorted by timestamp ascending."""
        entries = [e for e in self._entries if e.model_id == model_id]
        entries.sort(key=lambda e: e.timestamp)
        return entries

    def __len__(self) -> int:
        """Return the number of unique models on the leaderboard."""
        return len({e.model_id for e in self._entries})


LEADERBOARD_TRACKER_REGISTRY = {"default": Leaderboard}
