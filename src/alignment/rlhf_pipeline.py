"""RLHF pipeline orchestrator: SFT → Reward Model → PPO phases."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class RLHFPhase(StrEnum):
    SFT = "sft"
    REWARD_MODEL = "reward_model"
    PPO = "ppo"
    EVAL = "eval"


@dataclass
class PhaseConfig:
    phase: RLHFPhase
    n_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-4
    metadata: dict = field(default_factory=dict)


@dataclass
class PhaseResult:
    phase: RLHFPhase
    epoch: int
    loss: float
    metrics: dict = field(default_factory=dict)


class RLHFPipeline:
    """Orchestrates SFT → Reward Model → PPO training phases."""

    def __init__(self, phases: list[PhaseConfig] | None = None) -> None:
        if phases is None:
            phases = [
                PhaseConfig(phase=RLHFPhase.SFT),
                PhaseConfig(phase=RLHFPhase.REWARD_MODEL),
                PhaseConfig(phase=RLHFPhase.PPO),
            ]
        self._phases: list[PhaseConfig] = phases
        self._index: int = 0
        self._history: list[PhaseResult] = []

    def current_phase(self) -> RLHFPhase | None:
        """Return the current phase, or None if all phases are complete."""
        if self._index < len(self._phases):
            return self._phases[self._index].phase
        return None

    def advance_phase(self) -> bool:
        """Advance to the next phase. Returns False if already at the end."""
        if self._index >= len(self._phases) - 1:
            self._index = len(self._phases)  # mark as fully complete
            return False
        self._index += 1
        return True

    def log_result(self, result: PhaseResult) -> None:
        """Append a PhaseResult to the history."""
        self._history.append(result)

    def history(self, phase: RLHFPhase | None = None) -> list[PhaseResult]:
        """Return all results, optionally filtered by phase."""
        if phase is None:
            return list(self._history)
        return [r for r in self._history if r.phase == phase]

    def summary(self) -> dict:
        """Return a high-level summary of pipeline state."""
        current = self.current_phase()
        phases_completed = self._index if current is not None else len(self._phases)
        return {
            "phases_completed": phases_completed,
            "current_phase": current.value if current is not None else "done",
            "total_results": len(self._history),
        }


RLHF_PIPELINE = RLHFPipeline()
