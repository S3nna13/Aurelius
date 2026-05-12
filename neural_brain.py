"""Compatibility neural-brain surface for the middle tier."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BrainContext:
    state: str
    plan: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    verifications: list[str] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    output: str = ""


class NeuralBrain:
    def __init__(self) -> None:
        self._runs = 0
        self._last_input = ""
        self._last_context: BrainContext | None = None

    def run(self, input_text: str) -> BrainContext:
        self._runs += 1
        self._last_input = input_text
        tokens = [token for token in input_text.split() if token]
        digest = hashlib.sha256(input_text.encode("utf-8")).hexdigest()[:12]
        plan = [
            f"parse:{len(tokens)}",
            f"reason:{digest[:4]}",
            "compose:summary",
        ]
        reasoning = [
            f"Observed {len(tokens)} token(s).",
            f"Digest {digest}.",
            f"Focus words: {', '.join(tokens[:5]) if tokens else 'none'}.",
        ]
        actions = ["parse", "summarize", "respond"]
        verifications = ["input_present", "output_truncated"]
        reflections = ["compatibility-mode", "deterministic-output"]
        output = f"Processed {len(tokens)} token(s): {input_text[:200]}"
        self._last_context = BrainContext(
            state=f"ready:{digest}",
            plan=plan,
            reasoning=reasoning,
            actions=actions,
            verifications=verifications,
            reflections=reflections,
            output=output,
        )
        return self._last_context

    def get_stats(self) -> dict[str, Any]:
        return {
            "runs": self._runs,
            "last_input_length": len(self._last_input),
            "last_output_length": len(self._last_context.output) if self._last_context else 0,
        }
