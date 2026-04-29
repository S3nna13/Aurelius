"""Curriculum design — 3-stage progressive training data curriculum.

Stage 1 (Foundation): 0 → 22T tokens
  - 80% web (FineWeb, Wikipedia)
  - 12% code (The Stack v2)
  - 5% math (OpenWebMath)
  - 3% science (arXiv)

Stage 2 (Specialization): 22T → 28T tokens
  - 60% web (high-quality filtered)
  - 20% code (high-quality subset)
  - 12% math
  - 8% science

Stage 3 (Sharpening): 28T → 30T tokens
  - 40% web (strictly filtered, edu_score >= 4)
  - 25% code (high-quality)
  - 20% math
  - 15% science
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CurriculumStage:
    name: str
    token_range: tuple[int, int]
    token_budget: int
    data_mix: dict[str, float]
    quality_threshold: float = 0.5
    lr_peak: float = 3e-4
    lr_min: float = 3e-5


CURRICULUM: list[CurriculumStage] = [
    CurriculumStage("foundation", (0, 22_000_000_000_000), 22_000_000_000_000, {
        "web": 0.80, "code": 0.12, "math": 0.05, "science": 0.03,
    }, quality_threshold=0.4, lr_peak=3e-4, lr_min=3e-5),
    CurriculumStage("specialization", (22_000_000_000_000, 28_000_000_000_000), 6_000_000_000_000, {
        "web": 0.60, "code": 0.20, "math": 0.12, "science": 0.08,
    }, quality_threshold=0.6, lr_peak=1.5e-4, lr_min=3e-5),
    CurriculumStage("sharpening", (28_000_000_000_000, 30_000_000_000_000), 2_000_000_000_000, {
        "web": 0.40, "code": 0.25, "math": 0.20, "science": 0.15,
    }, quality_threshold=0.75, lr_peak=5e-5, lr_min=1e-5),
]


class CurriculumScheduler:
    """Determines active curriculum stage and data mix based on tokens seen."""

    def __init__(self, curriculum: list[CurriculumStage] | None = None):
        self.curriculum = curriculum or CURRICULUM
        self._current_stage_idx: int = 0

    def get_stage(self, tokens_seen: int) -> CurriculumStage:
        for stage in self.curriculum:
            low, high = stage.token_range
            if low <= tokens_seen < high:
                return stage
        return self.curriculum[-1]

    def get_data_mix(self, tokens_seen: int) -> dict[str, float]:
        stage = self.get_stage(tokens_seen)
        return stage.data_mix

    def get_lr(self, tokens_seen: int, base_step: int, warmup_steps: int = 2000, total_steps: int = 150000) -> float:
        import math
        stage = self.get_stage(tokens_seen)
        if base_step < warmup_steps:
            return stage.lr_peak * (base_step / max(warmup_steps, 1))
        progress = (base_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(stage.lr_min, stage.lr_min + (stage.lr_peak - stage.lr_min) * cosine)

    def get_quality_filter(self, tokens_seen: int) -> float:
        stage = self.get_stage(tokens_seen)
        return stage.quality_threshold

    def advance(self) -> bool:
        if self._current_stage_idx < len(self.curriculum) - 1:
            self._current_stage_idx += 1
            return True
        return False

    def summary(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "token_range": s.token_range,
                "budget": s.token_budget,
                "mix": s.data_mix,
            }
            for s in self.curriculum
        ]
