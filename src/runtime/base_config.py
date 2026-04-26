"""Base configuration model with environment variable binding."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class BaseConfig:
    """Base configuration with env var loading and data validation."""

    _prefix: str = "AURELIUS_"

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key.lower(), default)

    def load_env(self, prefix: str | None = None) -> None:
        pfx = prefix or self._prefix
        for key in dir(self):
            if key.startswith("_"):
                continue
            env_key = f"{pfx}{key.upper()}"
            if env_key in os.environ:
                val = os.environ[env_key]
                setattr(self, key, self._coerce(val))

    def _coerce(self, val: str) -> Any:
        if val.lower() in ("true", "yes", "1"):
            return True
        if val.lower() in ("false", "no", "0"):
            return False
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        return val

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


BASE_CONFIG = BaseConfig()
