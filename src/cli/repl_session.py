"""REPL session — history management and prompt formatting."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ReplMode(str, Enum):
    INTERACTIVE = "interactive"
    SCRIPTED = "scripted"
    PIPE = "pipe"


@dataclass
class ReplConfig:
    mode: ReplMode
    prompt: str = "aurelius> "
    history_file: str = ".aurelius_history"
    max_history: int = 1000


class ReplSession:
    """Manages REPL history and prompt rendering."""

    def __init__(self, config: ReplConfig) -> None:
        self.config = config
        self._history: collections.deque[str] = collections.deque(
            maxlen=config.max_history
        )

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def add_to_history(self, command: str) -> None:
        """Append *command* to in-memory history (capped at max_history)."""
        self._history.append(command)

    def get_history(self) -> list[str]:
        """Return a copy of the current history list."""
        return list(self._history)

    def save_history(self, path: str) -> None:
        """Write history to *path*, one entry per line."""
        dest = Path(path)
        dest.write_text("\n".join(self._history), encoding="utf-8")

    def load_history(self, path: str) -> int:
        """Load history from *path*; return the number of entries loaded."""
        src = Path(path)
        if not src.exists():
            return 0

        lines = src.read_text(encoding="utf-8").splitlines()
        loaded = 0
        for line in lines:
            if line:  # skip blank lines
                self._history.append(line)
                loaded += 1

        return loaded

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def format_prompt(self, context: Optional[dict] = None) -> str:
        """Return the prompt string with placeholders interpolated from *context*.

        Example: prompt="({model_name})> ", context={"model_name": "aurelius-1.4B"}
                 → "(aurelius-1.4B)> "

        Unknown placeholders are left as-is (no KeyError).
        """
        if not context:
            return self.config.prompt

        try:
            return self.config.prompt.format_map(context)
        except (KeyError, ValueError):
            return self.config.prompt
