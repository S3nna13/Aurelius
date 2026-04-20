"""Integration: PRF query expander retrieval registry."""

from __future__ import annotations

import src.retrieval as ret
from src.model.config import AureliusConfig


def test_query_expander_registry():
    assert ret.QUERY_EXPANDER_REGISTRY["prf"] is ret.PRFQueryExpander


def test_config_default_off():
    assert AureliusConfig().retrieval_prf_query_expander_enabled is False


def test_retriever_registry_intact():
    assert "bm25" in ret.RETRIEVER_REGISTRY


def test_smoke_expand():
    ex = ret.PRFQueryExpander()
    r = ex.expand("search", ["search ranking bm25"], num_terms=2)
    assert "bm25" in r.query or "ranking" in r.query
