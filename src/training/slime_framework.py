"""Slime RL Framework — GLM-5 §4 (arXiv:2602.15763).
Unified post-training infrastructure: task_type → verifier → reward_fn.
Default tasks: swe (exact), terminal (substring), search (overlap).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

VerifierFn = Callable[[str, str], bool]
RewardFn = Callable[[str, str], float]


@dataclass
class SlimeTask:
    name: str
    verifier: VerifierFn
    reward_fn: RewardFn

    def verify(self, completion: str, target: str) -> bool:
        return self.verifier(completion, target)

    def reward(self, completion: str, target: str) -> float:
        return self.reward_fn(completion, target)


class SlimeTaskRouter:
    def __init__(self):
        self._registry: dict[str, SlimeTask] = {}

    def register_task(self, task: SlimeTask) -> None:
        self._registry[task.name] = task

    def route(self, task_type: str) -> SlimeTask:
        if task_type not in self._registry:
            raise ValueError(
                f"Unknown task type {task_type!r}. Registered: {sorted(self._registry.keys())}"
            )
        return self._registry[task_type]

    def registered_types(self) -> list[str]:
        return sorted(self._registry.keys())

    def __len__(self) -> int:
        return len(self._registry)


def _overlap_score(completion: str, target: str) -> float:
    if not target:
        return 1.0 if not completion else 0.0
    target_words = set(target.lower().split())
    completion_words = set(completion.lower().split())
    if not target_words:
        return 0.0
    return len(target_words & completion_words) / len(target_words)


def make_default_router() -> SlimeTaskRouter:
    router = SlimeTaskRouter()
    router.register_task(
        SlimeTask(
            name="swe",
            verifier=lambda c, t: c.strip() == t.strip(),
            reward_fn=lambda c, t: 1.0 if c.strip() == t.strip() else 0.0,
        )
    )
    router.register_task(
        SlimeTask(
            name="terminal",
            verifier=lambda c, t: t in c,
            reward_fn=lambda c, t: 1.0 if t in c else 0.0,
        )
    )
    router.register_task(
        SlimeTask(
            name="search",
            verifier=lambda c, t: _overlap_score(c, t) > 0,
            reward_fn=_overlap_score,
        )
    )
    return router
