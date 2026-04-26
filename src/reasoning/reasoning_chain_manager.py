"""Unified reasoning chain manager routing CoT/ToT/MCTS/scratchpad strategies.

Inspired by Kimi-Dev patch-synthesis loop (MoonshotAI, MIT) and DeepSeek-R1
reasoning chain; Aurelius-native unified API. License: MIT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ChainStrategy(Enum):
    COT = "cot"
    TOT = "tot"
    MCTS = "mcts"
    SCRATCHPAD = "scratchpad"


@dataclass
class ChainStep:
    strategy: ChainStrategy
    content: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


class ReasoningChainManager:
    def __init__(
        self,
        default_strategy: ChainStrategy = ChainStrategy.COT,
        max_steps: int = 256,
        max_step_length: int = 4096,
    ) -> None:
        self.default_strategy = default_strategy
        self.max_steps = max_steps
        self.max_step_length = max_step_length
        self._steps: list[ChainStep] = []

    def add_step(
        self,
        content: str,
        strategy: ChainStrategy | None = None,
        score: float = 0.0,
        metadata: dict | None = None,
    ) -> ChainStep:
        if len(self._steps) >= self.max_steps:
            raise ValueError(f"max_steps {self.max_steps} reached")
        content = content[: self.max_step_length]
        step = ChainStep(
            strategy=strategy if strategy is not None else self.default_strategy,
            content=content,
            score=score,
            metadata=dict(metadata) if metadata else {},
        )
        self._steps.append(step)
        return step

    def get_steps(self, strategy: ChainStrategy | None = None) -> list[ChainStep]:
        if strategy is None:
            return list(self._steps)
        return [s for s in self._steps if s.strategy == strategy]

    def summarize(self) -> str:
        return "\n".join(s.content for s in self._steps)

    def clear(self) -> None:
        self._steps.clear()

    def export(self) -> list[dict]:
        return [
            {
                "strategy": s.strategy.value,
                "content": s.content,
                "score": s.score,
                "metadata": dict(s.metadata),
            }
            for s in self._steps
        ]

    @classmethod
    def from_export(cls, data: list[dict]) -> ReasoningChainManager:
        mgr = cls()
        for item in data:
            if "strategy" not in item or "content" not in item:
                raise ValueError("each step requires 'strategy' and 'content' keys")
            mgr.add_step(
                content=item["content"],
                strategy=ChainStrategy(item["strategy"]),
                score=item.get("score", 0.0),
                metadata=item.get("metadata", {}),
            )
        return mgr


CHAIN_MANAGER_REGISTRY: dict[str, type] = {"chain_manager": ReasoningChainManager}
DEFAULT_CHAIN_MANAGER = ReasoningChainManager()
