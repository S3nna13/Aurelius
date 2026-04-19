"""End-to-end integration test for :class:`CorpusIndexer`.

Exercises the full public surface via the package-level ``src.retrieval``
namespace: walk -> chunk -> BM25 index -> query -> persist -> reload.
Also asserts that prior exports from :mod:`src.retrieval` remain intact,
so appending :class:`CorpusIndexer` to the package didn't regress
anything.
"""

from __future__ import annotations

import os

import pytest

import src.retrieval as retrieval_pkg
from src.retrieval import BM25Retriever
from src.retrieval.corpus_indexer import Chunk, CorpusIndexer


def _write(root, rel, text):
    p = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return p


def test_prior_retrieval_exports_intact():
    # Sanity: CorpusIndexer was appended additively; existing names still there.
    for name in (
        "BM25Retriever",
        "HybridRetriever",
        "RETRIEVER_REGISTRY",
        "EMBEDDING_REGISTRY",
        "RERANKER_REGISTRY",
        "reciprocal_rank_fusion",
        "CodeAwareTokenizer",
        "MMRReranker",
    ):
        assert hasattr(retrieval_pkg, name), f"missing prior export: {name}"


def test_end_to_end_walk_chunk_index_query_persist(tmp_path):
    # A small mixed-content corpus.
    _write(
        tmp_path,
        "src/alpha.py",
        "def add(a, b):\n    return a + b\n# alpha numerics module\n",
    )
    _write(
        tmp_path,
        "docs/bm25.md",
        "# BM25\n\nOkapi BM25 is a sparse retrieval scoring function.\n",
    )
    _write(
        tmp_path,
        "notes/readme.txt",
        "Retrieval augmented generation uses a retriever plus a generator.\n",
    )
    # A file with the wrong extension must be ignored.
    _write(tmp_path, "skip.log", "should not be walked")

    ci = CorpusIndexer(chunk_size=200, chunk_overlap=40)
    files = ci.walk_files(str(tmp_path))
    assert any(f.endswith("alpha.py") for f in files)
    assert any(f.endswith("bm25.md") for f in files)
    assert any(f.endswith("readme.txt") for f in files)
    assert not any(f.endswith(".log") for f in files)

    all_chunks: list[Chunk] = []
    for f in files:
        all_chunks.extend(ci.chunk_file(f))
    assert len(all_chunks) >= 3

    retriever, doc_map = ci.build_bm25_index(all_chunks)
    assert isinstance(retriever, BM25Retriever)

    hits = retriever.query("Okapi BM25 sparse retrieval", k=3)
    assert hits, "expected at least one hit for BM25 query"
    top_chunk = doc_map[hits[0][0]]
    assert top_chunk.source_path.endswith("bm25.md")

    # Persist + reload.
    index_path = tmp_path / "corpus.json"
    ci.save_index(all_chunks, str(index_path))
    assert index_path.exists()

    reloaded = ci.load_index(str(index_path))
    assert len(reloaded) == len(all_chunks)
    assert [c.chunk_id for c in reloaded] == [c.chunk_id for c in all_chunks]

    # Rebuild index from reloaded chunks, confirm identical top hit.
    retriever2, doc_map2 = ci.build_bm25_index(reloaded)
    hits2 = retriever2.query("Okapi BM25 sparse retrieval", k=3)
    assert doc_map2[hits2[0][0]].source_path == top_chunk.source_path


def test_empty_chunks_rejected_by_build(tmp_path):
    ci = CorpusIndexer()
    with pytest.raises(ValueError):
        ci.build_bm25_index([])
