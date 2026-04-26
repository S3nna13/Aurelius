"""Multiagent debate framework based on Du et al. 2023."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class DebateConfig:
    n_agents: int = 3
    n_rounds: int = 2
    summarize: bool = True


@dataclass(frozen=True)
class DebateRound:
    round_idx: int
    responses: tuple[str, ...]


class DebateSession:
    def __init__(self, config: DebateConfig | None = None) -> None:
        self._config = config or DebateConfig()
        self.history: list[DebateRound] = []

    def reset(self) -> None:
        self.history = []

    def run(self, question: str, agent_fn: Callable[[str, list[str]], str]) -> str:
        cfg = self._config
        prev_responses: list[str] = []

        for round_idx in range(cfg.n_rounds + 1):
            round_responses: list[str] = []
            for agent_idx in range(cfg.n_agents):
                if round_idx == 0:
                    context: list[str] = []
                else:
                    context = [r for i, r in enumerate(prev_responses) if i != agent_idx]
                round_responses.append(agent_fn(question, context))
            self.history.append(DebateRound(round_idx=round_idx, responses=tuple(round_responses)))
            prev_responses = round_responses

        final_responses = prev_responses
        if not cfg.summarize:
            return final_responses[0]

        counts = Counter(final_responses)
        winner, _ = counts.most_common(1)[0]
        return winner


DEBATE_REGISTRY: dict[str, type[DebateSession]] = {"default": DebateSession}
