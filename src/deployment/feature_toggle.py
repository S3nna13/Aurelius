"""
feature_toggle.py — Feature flag/toggle management for gradual rollouts.
Aurelius LLM Project — stdlib only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class ToggleState(Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"
    PERCENTAGE = "percentage"
    ALLOWLIST = "allowlist"


@dataclass
class FeatureToggle:
    name: str
    state: ToggleState
    percentage: float = 0.0
    allowlist: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class FeatureToggleManager:
    """Manages feature flags and toggles for gradual rollouts via SHA-256 hashing."""

    def __init__(self) -> None:
        self._toggles: dict[str, FeatureToggle] = {}

    def register(self, toggle: FeatureToggle) -> None:
        """Register a toggle. Raises ValueError if the name already exists."""
        if toggle.name in self._toggles:
            raise ValueError(f"Toggle '{toggle.name}' is already registered.")
        self._toggles[toggle.name] = toggle

    def is_enabled(self, name: str, user_id: str = "") -> bool:
        """
        Evaluate whether a feature is enabled for the given user.

        ENABLED    → True
        DISABLED   → False
        PERCENTAGE → deterministic hash-based check
        ALLOWLIST  → user_id in allowlist
        """
        toggle = self._toggles.get(name)
        if toggle is None:
            return False

        if toggle.state == ToggleState.ENABLED:
            return True
        if toggle.state == ToggleState.DISABLED:
            return False
        if toggle.state == ToggleState.PERCENTAGE:
            digest = hashlib.sha256((name + user_id).encode()).hexdigest()
            bucket = int(digest, 16) % 100
            return bucket < toggle.percentage
        if toggle.state == ToggleState.ALLOWLIST:
            return user_id in toggle.allowlist

        return False  # unreachable but safe

    def override(self, name: str, state: ToggleState) -> bool:
        """Override the state of an existing toggle. Returns False if not found."""
        toggle = self._toggles.get(name)
        if toggle is None:
            return False
        toggle.state = state
        return True

    def list_toggles(self) -> list:
        """Return sorted list of all registered toggle names."""
        return sorted(self._toggles.keys())

    def get(self, name: str) -> FeatureToggle | None:
        """Return the FeatureToggle for *name*, or None if not found."""
        return self._toggles.get(name)


# ---------------------------------------------------------------------------
# Module registry
# ---------------------------------------------------------------------------
FEATURE_TOGGLE_REGISTRY: dict = {"default": FeatureToggleManager}

REGISTRY = FEATURE_TOGGLE_REGISTRY
