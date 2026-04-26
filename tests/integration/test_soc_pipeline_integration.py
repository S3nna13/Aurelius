"""Integration: SOC pipeline wired through DEFENSIVE_PIPELINE_REGISTRY."""

from __future__ import annotations

from src.model.config import AureliusConfig
from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY, FeatureFlag
from src.security import DEFENSIVE_PIPELINE_REGISTRY
from src.security.soc_pipeline import DEFAULT_SOC_PIPELINE, SOCPipeline


def test_soc_pipeline_registered_in_defensive_registry():
    assert "default" in DEFENSIVE_PIPELINE_REGISTRY
    assert isinstance(DEFENSIVE_PIPELINE_REGISTRY["default"], SOCPipeline)


def test_config_flag_defaults_off():
    cfg = AureliusConfig()
    assert cfg.security_soc_pipeline_enabled is False


def test_config_flag_settable():
    FEATURE_FLAG_REGISTRY.register(FeatureFlag(name="security.soc_pipeline", enabled=True))
    cfg = AureliusConfig()
    assert cfg.security_soc_pipeline_enabled is True


def test_end_to_end_when_feature_enabled():
    FEATURE_FLAG_REGISTRY.register(FeatureFlag(name="security.soc_pipeline", enabled=True))
    cfg = AureliusConfig()
    assert cfg.security_soc_pipeline_enabled
    ev = DEFAULT_SOC_PIPELINE.run_one(
        {
            "event_id": "E-int-1",
            "source": "integration",
            "severity": "critical",
            "timestamp": "2026-04-20T10:00:00Z",
            "payload": {"src_ip": "192.168.1.1"},
        }
    )
    assert ev.event_id == "E-int-1"
    assert ev.route is not None


def test_feature_flag_off_preserves_default_construction():
    FEATURE_FLAG_REGISTRY._flags.pop("security.soc_pipeline", None)
    cfg = AureliusConfig()
    assert cfg.security_soc_pipeline_enabled is False
    # Confirm no side-effects of disabled flag: registry still usable.
    assert "default" in DEFENSIVE_PIPELINE_REGISTRY
