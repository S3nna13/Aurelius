"""CLI configuration file parser for agent settings."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CLIAppConfig:
    """Configuration loaded from file or environment."""

    data_dir: str = "~/.aurelius"
    log_level: str = "INFO"
    default_model: str = "default"
    max_history: int = 1000
    theme: str = "dark"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> CLIAppConfig:
        return cls(
            data_dir=os.environ.get("AURELIUS_DATA_DIR", "~/.aurelius"),
            log_level=os.environ.get("AURELIUS_LOG_LEVEL", "INFO"),
            default_model=os.environ.get("AURELIUS_MODEL", "default"),
            max_history=int(os.environ.get("AURELIUS_MAX_HISTORY", "1000")),
            theme=os.environ.get("AURELIUS_THEME", "dark"),
        )

    def save(self, path: str) -> None:
        import json

        data = {
            "data_dir": self.data_dir,
            "log_level": self.log_level,
            "default_model": self.default_model,
            "max_history": self.max_history,
            "theme": self.theme,
            "extra": self.extra,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> CLIAppConfig:
        import json

        with open(path) as f:
            data = json.load(f)
        return cls(
            data_dir=data.get("data_dir", "~/.aurelius"),
            log_level=data.get("log_level", "INFO"),
            default_model=data.get("default_model", "default"),
            max_history=data.get("max_history", 1000),
            theme=data.get("theme", "dark"),
            extra=data.get("extra", {}),
        )


CLI_CONFIG = CLIAppConfig()
