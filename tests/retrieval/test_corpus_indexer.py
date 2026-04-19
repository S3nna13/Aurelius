"""Unit tests for ``src.retrieval.corpus_indexer``."""

from __future__ import annotations

import os
import time

import pytest

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.corpus_indexer import Chunk, CorpusIndexer


def _write(root, rel, text, binary=False):
    p = os.path.join(str(root), rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if binary:
        with open(p, "wb") as f:
            f.write(text)
    else:
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
    return p


def test_walk_files_filters_by_extension(tmp_path):
    _write(tmp_path, "a.py", "print('a')")
    _write(tmp_path, "sub/b.md", "# heading")
    _write(tmp_path, "c.txt", "plain")
    _write(tmp_path, "ignore.log", "x")
    _write(tmp_path, "nested/deep/d.py", "pass")

    ci = CorpusIndexer()
    found = ci.walk_files(str(tmp_path))
    # 4 matches: a.py, b.md, c.txt, d.py; ignore.log excluded.
    names = sorted(os.path.basename(f) for f in found)
    assert names == ["a.py", "b.md", "c.txt", "d.py"]


def test_chunk_file_produces_expected_chunks(tmp_path):
    # 2500 chars; chunk_size=800, overlap=100 -> step=700 -> chunks at
    # starts [0,700,1400,2100] -> 4 chunks.
    text = "x" * 2500
    p = _write(tmp_path, "a.txt", text)
    ci = CorpusIndexer(chunk_size=800, chunk_overlap=100)
    chunks = ci.chunk_file(p)
    assert len(chunks) == 4
    assert chunks[0].start_char == 0 and chunks[0].end_char == 800
    assert chunks[-1].end_char == 2500


def test_overlapping_chunks_share_boundary_content(tmp_path):
    text = "".join(chr(ord("a") + (i % 26)) for i in range(2000))
    ci = CorpusIndexer(chunk_size=500, chunk_overlap=100)
    chunks = ci.chunk_text(text, "mem://x")
    # Overlap region: chunks[0][-100:] == chunks[1][:100].
    assert len(chunks) >= 2
    assert chunks[0].content[-100:] == chunks[1].content[:100]


def test_chunk_text_short_returns_one(tmp_path):
    ci = CorpusIndexer(chunk_size=800, chunk_overlap=100)
    chunks = ci.chunk_text("hello world", "mem://x")
    assert len(chunks) == 1
    assert chunks[0].content == "hello world"
    assert chunks[0].start_char == 0 and chunks[0].end_char == 11


def test_chunk_text_long_multiple(tmp_path):
    ci = CorpusIndexer(chunk_size=100, chunk_overlap=20)
    chunks = ci.chunk_text("y" * 500, "mem://x")
    assert len(chunks) > 1
    # Every chunk (except possibly last) has exactly chunk_size content.
    for c in chunks[:-1]:
        assert len(c.content) == 100


def test_build_bm25_index_returns_functional_retriever(tmp_path):
    _write(tmp_path, "a.txt", "the quick brown fox jumps over the lazy dog")
    _write(tmp_path, "b.txt", "lorem ipsum dolor sit amet consectetur adipiscing")
    _write(tmp_path, "c.txt", "retrieval augmented generation with bm25 sparse index")
    ci = CorpusIndexer(chunk_size=800, chunk_overlap=100)
    all_chunks: list[Chunk] = []
    for f in ci.walk_files(str(tmp_path)):
        all_chunks.extend(ci.chunk_file(f))
    retriever, doc_map = ci.build_bm25_index(all_chunks)
    assert isinstance(retriever, BM25Retriever)
    assert set(doc_map.keys()) == set(range(len(all_chunks)))
    results = retriever.query("bm25 retrieval", k=3)
    assert results
    top_doc = doc_map[results[0][0]]
    assert "bm25" in top_doc.content


def test_save_and_load_index_roundtrip(tmp_path):
    ci = CorpusIndexer(chunk_size=50, chunk_overlap=10)
    chunks = ci.chunk_text("alpha beta gamma delta " * 20, "mem://x")
    # Add metadata to exercise round-trip.
    for c in chunks:
        c.metadata["k"] = "v"
        c.metadata["n"] = 7
    out = tmp_path / "idx.json"
    ci.save_index(chunks, str(out))
    reloaded = ci.load_index(str(out))
    assert len(reloaded) == len(chunks)
    for a, b in zip(chunks, reloaded):
        assert a.chunk_id == b.chunk_id
        assert a.source_path == b.source_path
        assert a.content == b.content
        assert a.start_char == b.start_char
        assert a.end_char == b.end_char
        assert a.chunk_index == b.chunk_index
        assert a.metadata == b.metadata


def test_binary_file_is_skipped(tmp_path):
    p = _write(tmp_path, "bin.txt", b"\x00\x01\x02\x03ABC\x00more", binary=True)
    ci = CorpusIndexer()
    assert ci.chunk_file(p) == []


def test_chunk_ids_are_deterministic(tmp_path):
    ci = CorpusIndexer(chunk_size=100, chunk_overlap=20)
    text = "deterministic " * 50
    a = ci.chunk_text(text, "mem://d")
    b = ci.chunk_text(text, "mem://d")
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        CorpusIndexer(chunk_size=0)
    with pytest.raises(ValueError):
        CorpusIndexer(chunk_size=-5)


def test_overlap_ge_chunk_size_raises():
    with pytest.raises(ValueError):
        CorpusIndexer(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        CorpusIndexer(chunk_size=100, chunk_overlap=200)


def test_empty_text_returns_no_chunks():
    ci = CorpusIndexer()
    assert ci.chunk_text("", "mem://e") == []


def test_hundred_file_corpus_fast(tmp_path):
    for i in range(100):
        _write(tmp_path, f"f{i:03d}.py", f"# file {i}\n" + ("word " * 80))
    ci = CorpusIndexer(chunk_size=400, chunk_overlap=50)
    t0 = time.perf_counter()
    chunks: list[Chunk] = []
    for f in ci.walk_files(str(tmp_path)):
        chunks.extend(ci.chunk_file(f))
    ci.build_bm25_index(chunks)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"indexing 100 files took {elapsed:.3f}s (>2s)"
    assert len(chunks) >= 100
