"""Integration: n-gram dedup data registry."""

from __future__ import annotations

import src.data as data
from src.model.config import AureliusConfig


def test_dedup_registry():
    assert data.DEDUP_REGISTRY["ngram_jaccard"] is data.NgramDeduplicationToolkit


def test_config_default_off():
    assert AureliusConfig().data_ngram_deduplication_enabled is False


def test_loader_registry_intact():
    assert "cwe_synthesis" in data.LOADER_REGISTRY


def test_smoke_cluster():
    cfg = AureliusConfig(data_ngram_deduplication_enabled=True)
    assert cfg.data_ngram_deduplication_enabled is True
    g = data.NgramDeduplicationToolkit.cluster_duplicates(["aa", "aa", "bb"], n=1, threshold=1.0)
    assert len(g) == 2
