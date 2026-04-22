"""Integration tests for the code-aware embedder wired through the registry."""

from __future__ import annotations

import importlib
import math
import sys


def test_embedding_registry_exposes_code_aware():
    for mod in [m for m in list(sys.modules) if m.startswith("src.retrieval")]:
        del sys.modules[mod]
    retrieval = importlib.import_module("src.retrieval")
    assert "code_aware" in retrieval.EMBEDDING_REGISTRY
    cls = retrieval.EMBEDDING_REGISTRY["code_aware"]
    assert cls is retrieval.CodeAwareEmbedder


def test_end_to_end_embed_via_registry():
    retrieval = importlib.import_module("src.retrieval")
    cls = retrieval.EMBEDDING_REGISTRY["code_aware"]
    emb = cls(lambda t: retrieval.stub_token_embed(t, 128), d_embed=128)
    v = emb.embed("def foo(x): return x + 1")
    assert len(v) == 128
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-9


def test_config_flag_defaults_off():
    from src.model.config import AureliusConfig

    cfg = AureliusConfig()
    assert hasattr(cfg, "retrieval_code_aware_embedder_enabled")
    assert cfg.retrieval_code_aware_embedder_enabled is False


def test_sibling_registries_preserved():
    for mod in [m for m in list(sys.modules) if m.startswith("src.retrieval")]:
        del sys.modules[mod]
    retrieval = importlib.import_module("src.retrieval")
    assert "bm25" in retrieval.RETRIEVER_REGISTRY
    assert "dense" in retrieval.EMBEDDING_REGISTRY
    assert "cross_encoder" in retrieval.RERANKER_REGISTRY
    assert "basic" in retrieval.CITATION_REGISTRY
