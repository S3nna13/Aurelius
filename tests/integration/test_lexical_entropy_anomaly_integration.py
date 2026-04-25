"""Integration: lexical entropy safety filter."""

from __future__ import annotations

import src.safety as safety
from src.model.config import AureliusConfig
from src.runtime.feature_flags import FeatureFlag, FEATURE_FLAG_REGISTRY


def test_registry():
    assert safety.SAFETY_FILTER_REGISTRY["lexical_entropy"] is safety.LexicalEntropyAnomalyDetector


def test_config_default_off():
    assert AureliusConfig().safety_lexical_entropy_anomaly_enabled is False


def test_jailbreak_filter_intact():
    assert safety.SAFETY_FILTER_REGISTRY["jailbreak"] is safety.JailbreakDetector


def test_smoke_score_with_flag():
    FEATURE_FLAG_REGISTRY.register(FeatureFlag(name="safety.lexical_entropy_anomaly", enabled=True))
    cfg = AureliusConfig()
    assert cfg.safety_lexical_entropy_anomaly_enabled is True
    det = safety.LexicalEntropyAnomalyDetector()
    r = det.score("x " * 40)
    assert isinstance(r.is_anomaly, bool)
