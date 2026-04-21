"""Integration tests for the citation tracker wired through the registry."""

from __future__ import annotations

import importlib
import sys


def test_citation_registry_exposes_tracker():
    for mod in [m for m in list(sys.modules) if m.startswith("src.retrieval")]:
        del sys.modules[mod]
    retrieval = importlib.import_module("src.retrieval")
    assert hasattr(retrieval, "CITATION_REGISTRY")
    assert "basic" in retrieval.CITATION_REGISTRY
    cls = retrieval.CITATION_REGISTRY["basic"]
    assert isinstance(cls, type)
    tr = cls()
    Source = retrieval.CitationSource
    src = Source(
        id="s1",
        text="integration test phrase of sufficient length here",
        origin="test",
        retrieved_at="2026-04-20T00:00:00Z",
    )
    rep = tr.track(
        "prefix integration test phrase of sufficient length here tail.",
        [src],
    )
    assert len(rep.spans) == 1
    assert rep.coverage_ratio > 0.0


def test_config_flag_defaults_off():
    from src.model.config import AureliusConfig

    cfg = AureliusConfig()
    assert hasattr(cfg, "retrieval_citation_tracker_enabled")
    assert cfg.retrieval_citation_tracker_enabled is False


def test_sibling_registries_preserved():
    for mod in [m for m in list(sys.modules) if m.startswith("src.retrieval")]:
        del sys.modules[mod]
    retrieval = importlib.import_module("src.retrieval")
    # Original registries remain dicts and contain prior entries.
    assert "bm25" in retrieval.RETRIEVER_REGISTRY
    assert isinstance(retrieval.EMBEDDING_REGISTRY, dict)
    assert isinstance(retrieval.RERANKER_REGISTRY, dict)
