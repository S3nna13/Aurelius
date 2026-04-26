"""Integration: kNN known-bad prompt safety filter."""

from __future__ import annotations

import src.safety as safety
from src.model.config import AureliusConfig
from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY, FeatureFlag


def test_registry():
    assert safety.SAFETY_FILTER_REGISTRY["knn_known_bad_guard"] is safety.KnnKnownBadGuard


def test_config_default_off():
    assert AureliusConfig().safety_knn_known_bad_guard_enabled is False


def test_config_opt_in():
    FEATURE_FLAG_REGISTRY.register(FeatureFlag(name="safety.knn_known_bad_guard", enabled=True))
    cfg = AureliusConfig()
    assert cfg.safety_knn_known_bad_guard_enabled is True


def test_other_safety_filters_intact():
    assert safety.SAFETY_FILTER_REGISTRY["jailbreak"] is safety.JailbreakDetector
    assert safety.SAFETY_FILTER_REGISTRY["lexical_entropy"] is safety.LexicalEntropyAnomalyDetector


def test_seed_known_bad_exported():
    assert isinstance(safety.SEED_KNOWN_BAD, tuple)
    assert len(safety.SEED_KNOWN_BAD) >= 20


def test_smoke_check():
    def emb(t: str) -> list[float]:
        return [float(len(t)), float(sum(ord(c) for c in t[:4])), 1.0]

    g = safety.KnnKnownBadGuard(embed_fn=emb)
    g.bulk_load(list(safety.SEED_KNOWN_BAD))
    v = g.check("hi")
    assert isinstance(v, safety.KnnVerdict)
