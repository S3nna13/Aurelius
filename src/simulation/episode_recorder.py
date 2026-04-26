"""Episode recorder: store, query, export episodes."""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class Episode:
    env_name: str
    policy_name: str
    trajectory_data: dict
    timestamp: str
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class EpisodeRecorder:
    """Store, query, and export episodes."""

    def __init__(self, max_episodes: int = 500) -> None:
        self.max_episodes = max_episodes
        self._episodes: list[Episode] = []
        self._index: dict[str, Episode] = {}

    def record(
        self,
        env_name: str,
        policy_name: str,
        trajectory,
        metadata: dict | None = None,
    ) -> Episode:
        episode = Episode(
            env_name=env_name,
            policy_name=policy_name,
            trajectory_data=dataclasses.asdict(trajectory),
            timestamp=datetime.now(tz=UTC).isoformat(),
            metadata=metadata or {},
        )
        # Evict oldest if over capacity
        if len(self._episodes) >= self.max_episodes:
            evicted = self._episodes.pop(0)
            self._index.pop(evicted.id, None)
        self._episodes.append(episode)
        self._index[episode.id] = episode
        return episode

    def get(self, episode_id: str) -> Episode | None:
        return self._index.get(episode_id)

    def query(
        self,
        env_name: str | None = None,
        success_only: bool = False,
    ) -> list[Episode]:
        result = self._episodes
        if env_name is not None:
            result = [e for e in result if e.env_name == env_name]
        if success_only:
            result = [e for e in result if e.trajectory_data.get("success", False)]
        return result

    def stats(self) -> dict:
        total = len(self._episodes)
        by_env: dict[str, int] = {}
        successes = 0
        for ep in self._episodes:
            by_env[ep.env_name] = by_env.get(ep.env_name, 0) + 1
            if ep.trajectory_data.get("success", False):
                successes += 1
        return {
            "total": total,
            "by_env": by_env,
            "success_rate": successes / total if total else 0.0,
        }

    def export_ids(self) -> list[str]:
        return [ep.id for ep in self._episodes]


EPISODE_RECORDER = EpisodeRecorder()
