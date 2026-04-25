from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class FeatureFlag:
    name: str
    enabled: bool
    rollout_pct: float = 100.0
    metadata: dict[str, Any] = field(default_factory=dict)


class FeatureFlagRegistry:
    def __init__(self, config_path: str | None = None) -> None:
        self._config_path = config_path
        self._flags: dict[str, FeatureFlag] = {}
        if config_path:
            self._load_yaml(config_path)

    def _load_yaml(self, path: str) -> None:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for name, cfg in data.items():
            if isinstance(cfg, bool):
                cfg = {"enabled": cfg}
            self._flags[name] = FeatureFlag(
                name=name,
                enabled=bool(cfg.get("enabled", False)),
                rollout_pct=float(cfg.get("rollout_pct", 100.0)),
                metadata={k: v for k, v in cfg.items() if k not in {"enabled", "rollout_pct"}},
            )

    def _env_override(self, name: str) -> bool | None:
        env_key = f"AURELIUS_FF_{name.upper().replace('.', '_')}"
        val = os.environ.get(env_key)
        if val is None:
            return None
        return val.strip() not in {"0", "false", "False", "FALSE", ""}

    def is_enabled(self, name: str, user_id: str | None = None) -> bool:
        env_val = self._env_override(name)
        if env_val is not None:
            return env_val

        flag = self._flags.get(name)
        if flag is None:
            return False

        if not flag.enabled:
            return False

        if flag.rollout_pct >= 100.0:
            return True

        uid = (user_id or "").encode()
        bucket = int(hashlib.sha256(uid).hexdigest(), 16) % 100
        return bucket < flag.rollout_pct

    def register(self, flag: FeatureFlag) -> None:
        self._flags[flag.name] = flag

    def list_flags(self) -> list[FeatureFlag]:
        return list(self._flags.values())

    def reload(self) -> None:
        if self._config_path:
            self._flags.clear()
            self._load_yaml(self._config_path)


FEATURE_FLAG_REGISTRY = FeatureFlagRegistry()
