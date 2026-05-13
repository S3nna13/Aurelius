"""Runtime feature flag system.

Provides two layers:
1. A YAML-backed surface-flag system (``FeatureFlag`` dataclass,
   ``register()``, rollout-percentage bucketing) for ~50 surface-level flags.
2. A simple enum-based runtime-flag system (``RuntimeFlag`` enum,
   env-var-driven defaults, thread-safe toggling) for high-level
   runtime capabilities.

Both layers live in the same ``FeatureFlagRegistry`` so that a single
module-level singleton (``FEATURE_FLAG_REGISTRY``) is the one source of
truth for all feature-flag queries.
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Enum — runtime flags (fixed set, env-var driven, default-off)
# ---------------------------------------------------------------------------


class RuntimeFlag(Enum):
    """Well-known runtime feature flags.

    These are high-level capabilities that can be toggled at startup via
    environment variables or at runtime via the registry API.  All flags
    default to *off* (conservative).  Set ``AURELIUS_FF_<NAME>=1`` in the
    environment to enable a flag from process start.
    """

    MOCK_BACKENDS = "MOCK_BACKENDS"
    EXPERIMENTAL_AGENTS = "EXPERIMENTAL_AGENTS"
    VERBOSE_LOGGING = "VERBOSE_LOGGING"
    TOOL_SANDBOX = "TOOL_SANDBOX"
    ADVANCED_CACHE = "ADVANCED_CACHE"
    TELEMETRY = "TELEMETRY"


# ---------------------------------------------------------------------------
# Dataclass — config-style surface flag (YAML-backed, rollout-aware)
# ---------------------------------------------------------------------------


@dataclass
class FeatureFlag:
    """A single feature-flag entry backed by YAML configuration.

    Parameters
    ----------
    name : str
        Dot-separated identifier, e.g. ``"safety.hallucination_guard"``.
    enabled : bool
        Whether the feature is active (subject to rollout gating).
    rollout_pct : float
        Percentage of users (by user-id hash) that see the flag.
    metadata : dict
        Arbitrary extra information (owner, domain, threshold, …).
    """

    name: str
    enabled: bool
    rollout_pct: float = 100.0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class FeatureFlagRegistry:
    """Central feature-flag registry.

    Supports two kinds of flags transparently:

    * **Surface flags** (``FeatureFlag`` dataclass) — registered via
      ``register()``, loaded from YAML, user-aware rollout gating.
    * **Runtime flags** (``RuntimeFlag`` enum members) — initialised from
      environment variables, toggled via ``set()``/``is_enabled()``.

    Thread-safe for runtime-flag operations.
    """

    def __init__(self, config_path: str | None = None) -> None:
        self._config_path = config_path
        self._flags: dict[str, FeatureFlag] = {}
        self._runtime_flags: dict[RuntimeFlag, bool] = {}
        self._lock = threading.Lock()

        if config_path:
            self._load_yaml(config_path)

        self._init_runtime_flags()

    # -- surface-flag internals (YAML) ---------------------------------------

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
                metadata={
                    k: v
                    for k, v in cfg.items()
                    if k not in {"enabled", "rollout_pct"}
                },
            )

    def _env_override(self, name: str) -> bool | None:
        env_key = f"AURELIUS_FF_{name.upper().replace('.', '_')}"
        val = os.environ.get(env_key)
        if val is None:
            return None
        return val.strip() not in {"0", "false", "False", "FALSE", ""}

    # -- runtime-flag internals ----------------------------------------------

    def _init_runtime_flags(self) -> None:
        """Initialise *RuntimeFlag* members from env vars or defaults."""
        for flag in RuntimeFlag:
            env_key = f"AURELIUS_FF_{flag.value}"
            val = os.environ.get(env_key)
            if val is not None:
                enabled = val.strip() not in {"0", "false", "False", "FALSE", ""}
            else:
                enabled = False  # conservative default
            self._runtime_flags[flag] = enabled

    # -- public query / mutation API -----------------------------------------

    def is_enabled(
        self, name_or_flag: str | RuntimeFlag, user_id: str | None = None
    ) -> bool:
        """Check whether a feature flag is enabled.

        Accepts either a ``RuntimeFlag`` enum member (runtime-flag path) or a
        plain string name (surface-flag path).  The surface-flag path also
        respects ``user_id`` for rollout-percentage bucketing.
        """
        if isinstance(name_or_flag, RuntimeFlag):
            return self._runtime_flags.get(name_or_flag, False)

        # ---- surface-flag (string) path ----
        env_val = self._env_override(name_or_flag)
        if env_val is not None:
            return env_val

        flag = self._flags.get(name_or_flag)
        if flag is None:
            return False

        if not flag.enabled:
            return False

        if flag.rollout_pct >= 100.0:
            return True

        uid = (user_id or "").encode()
        bucket = int(hashlib.sha256(uid).hexdigest(), 16) % 100
        return bucket < flag.rollout_pct

    def set(self, flag: RuntimeFlag, enabled: bool) -> None:
        """Enable or disable a runtime flag at runtime.

        This is thread-safe.
        """
        with self._lock:
            self._runtime_flags[flag] = enabled

    def list_all(self) -> dict[str, bool]:
        """Return a snapshot of every runtime flag and its current state.

        Returns
        -------
        dict[str, bool]
            Keys are the ``RuntimeFlag`` member names (e.g. ``"TOOL_SANDBOX"``).
        """
        with self._lock:
            return {flag.name: state for flag, state in self._runtime_flags.items()}

    def to_dict(self) -> dict[str, bool]:
        """Alias for ``list_all()`` — used by the capability contract.

        Returns the same dict of runtime-flag name → enabled state.
        """
        return self.list_all()

    # -- surface-flag API (unchanged) ----------------------------------------

    def register(self, flag: FeatureFlag) -> None:
        """Register a surface-level ``FeatureFlag`` dataclass instance."""
        self._flags[flag.name] = flag

    def list_flags(self) -> list[FeatureFlag]:
        """Return all registered surface-flag entries."""
        return list(self._flags.values())

    def reload(self) -> None:
        """Reload surface flags from the YAML config path (if any)."""
        if self._config_path:
            self._flags.clear()
            self._load_yaml(self._config_path)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

FEATURE_FLAG_REGISTRY: FeatureFlagRegistry = FeatureFlagRegistry()
"""Module-level feature-flag singleton.

Usage::

    from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY, RuntimeFlag

    if FEATURE_FLAG_REGISTRY.is_enabled(RuntimeFlag.TOOL_SANDBOX):
        ...
"""
