"""Integration tests for the BM25 retriever wired through the registry."""

from __future__ import annotations

import importlib
import sys


def test_registry_entry_is_class_and_usable():
    # Clean re-import to verify no side effects on import.
    for mod in [m for m in list(sys.modules) if m.startswith("src.retrieval")]:
        del sys.modules[mod]

    retrieval = importlib.import_module("src.retrieval")
    assert "bm25" in retrieval.RETRIEVER_REGISTRY
    cls = retrieval.RETRIEVER_REGISTRY["bm25"]
    assert isinstance(cls, type), "registry entry must be a class"

    # Sibling registries: dicts; siblings (e.g. cross_encoder) may
    # additively register entries.
    assert isinstance(retrieval.EMBEDDING_REGISTRY, dict)
    assert isinstance(retrieval.RERANKER_REGISTRY, dict)

    r = cls()
    docs = [
        "Python is a high-level programming language",
        "Rust is a systems programming language focused on safety",
        "BM25 is a ranking function used by search engines",
        "The cat sat on the mat",
        "Retrieval augmented generation uses sparse and dense retrievers",
    ]
    r.add_documents(docs)
    top = r.query("bm25 ranking", k=1)
    assert len(top) == 1
    assert top[0][0] == 2  # the BM25 doc is the expected top-1


def test_import_has_no_side_effects(tmp_path, monkeypatch):
    # Re-importing twice must be idempotent and not mutate the module globals.
    for mod in [m for m in list(sys.modules) if m.startswith("src.retrieval")]:
        del sys.modules[mod]
    m1 = importlib.import_module("src.retrieval")
    snap_r = dict(m1.RETRIEVER_REGISTRY)
    snap_e = dict(m1.EMBEDDING_REGISTRY)
    snap_rr = dict(m1.RERANKER_REGISTRY)
    m2 = importlib.reload(m1)
    assert dict(m2.RETRIEVER_REGISTRY) == snap_r
    assert dict(m2.EMBEDDING_REGISTRY) == snap_e
    assert dict(m2.RERANKER_REGISTRY) == snap_rr
