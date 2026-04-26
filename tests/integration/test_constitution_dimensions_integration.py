"""Integration tests for constitution dimensions registry + config flag."""

from __future__ import annotations

from src.alignment import ALIGNMENT_REGISTRY
from src.alignment.constitution_dimensions import ConstitutionScorer
from src.model.config import AureliusConfig
from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY, FeatureFlag


def test_alignment_registry_has_constitution_dimensions():
    assert "constitution_dimensions" in ALIGNMENT_REGISTRY
    assert ALIGNMENT_REGISTRY["constitution_dimensions"] is ConstitutionScorer


def test_config_flag_defaults_off():
    cfg = AureliusConfig()
    assert cfg.alignment_constitution_dimensions_enabled is False


def test_config_flag_togglable():
    FEATURE_FLAG_REGISTRY.register(
        FeatureFlag(name="alignment.constitution_dimensions", enabled=True)
    )
    cfg = AureliusConfig()
    assert cfg.alignment_constitution_dimensions_enabled is True


def test_scorer_smoke_via_registry():
    cls = ALIGNMENT_REGISTRY["constitution_dimensions"]
    scorer = cls()
    report = scorer.score("trajectory")
    assert len(report.grades) == 15
