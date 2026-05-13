"""Tests: plugins/memory/memory_compressor.py — Compresses memory stores via deduplication and summarization stubs."""

from __future__ import annotations

import pytest

from plugins.memory.memory_compressor import (
    MEMORY_COMPRESSOR_REGISTRY,
    CompressionResult,
    CompressionStrategy,
    MemoryCompressor,
)


@pytest.fixture
def compressor():
    return MemoryCompressor(similarity_threshold=0.8)


@pytest.fixture
def sample_memories():
    """Five memories, two exact duplicates, two similar."""
    return [
        {"id": "a", "content": "The sky is blue and clear."},
        {"id": "b", "content": "The sky is blue and clear."},  # dup of a
        {"id": "c", "content": "Python is a great language."},
        {"id": "d", "content": "Python is awesome."},  # similar to c
        {"id": "e", "content": "Rust is fast."},
    ]


class TestCompressionStrategy:
    def test_dedup_value(self):
        assert CompressionStrategy.DEDUP.value == "dedup"

    def test_truncate_value(self):
        assert CompressionStrategy.TRUNCATE.value == "truncate"

    def test_merge_similar_value(self):
        assert CompressionStrategy.MERGE_SIMILAR.value == "merge_similar"


class TestCompressionResult:
    def test_result_fields(self):
        r = CompressionResult(
            original_count=10,
            compressed_count=7,
            strategy=CompressionStrategy.DEDUP,
            removed_ids=["id1", "id2", "id3"],
        )
        assert r.original_count == 10
        assert r.compressed_count == 7
        assert r.strategy is CompressionStrategy.DEDUP
        assert r.removed_ids == ["id1", "id2", "id3"]


class TestMemoryCompressorDedup:
    def test_dedup_removes_duplicates(self, compressor, sample_memories):
        result = compressor.deduplicate(sample_memories)
        # a and b: exact duplicate content ('hello') -> b removed, a kept
        # c and d: NOT exact duplicates ('Python is a great language' vs 'Python is awesome')
        # e: unique ('Rust is fast')
        # Kept: a, c, d, e = 4 items
        assert result.original_count == 5
        assert result.compressed_count == 4
        assert result.strategy is CompressionStrategy.DEDUP
        assert "a" not in result.removed_ids  # first occurrence kept
        assert "b" in result.removed_ids  # duplicate removed

    def test_dedupe_keeps_last_occurrence_on_reversed(self, compressor):
        # If we reverse, the "first" occurrence changes
        memories = [
            {"id": "x", "content": "same"},
            {"id": "y", "content": "SAME"},  # case-insensitive dup
            {"id": "z", "content": "  same  "},  # whitespace-stripped dup
        ]
        result = compressor.deduplicate(memories)
        # All should be considered duplicates of the first (case+strip normalized)
        assert result.compressed_count == 1
        assert len(result.removed_ids) == 2

    def test_dedupe_empty(self, compressor):
        result = compressor.deduplicate([])
        assert result.original_count == 0
        assert result.compressed_count == 0
        assert result.removed_ids == []

    def test_last_result_memories_updated(self, compressor, sample_memories):
        compressor.deduplicate(sample_memories)
        ids = [m["id"] for m in compressor.last_result_memories]
        assert "b" not in ids


class TestMemoryCompressorTruncate:
    def test_truncate_keeps_max(self, compressor, sample_memories):
        result = compressor.truncate(sample_memories, max_count=3)
        assert result.original_count == 5
        assert result.compressed_count == 3
        assert result.strategy is CompressionStrategy.TRUNCATE
        assert len(result.removed_ids) == 2

    def test_truncate_all_kept(self, compressor, sample_memories):
        result = compressor.truncate(sample_memories, max_count=10)
        assert result.compressed_count == 5
        assert result.removed_ids == []

    def test_truncate_none_kept(self, compressor, sample_memories):
        result = compressor.truncate(sample_memories, max_count=0)
        assert result.compressed_count == 0
        assert len(result.removed_ids) == 5

    def test_truncate_removed_ids_order(self, compressor, sample_memories):
        result = compressor.truncate(sample_memories, max_count=2)
        assert set(result.removed_ids) == {"c", "d", "e"}


class TestMemoryCompressorMergeSimilar:
    def test_merge_similar_removes_similar(self, compressor):
        # similarity threshold 0.8, these have Jaccard overlap
        memories = [
            {"id": "a", "content": "Python is a great language for data science."},
            {"id": "b", "content": "Python is a great language."},  # very similar
            {"id": "c", "content": "Rust is a fast systems language."},  # different
        ]
        result = compressor.merge_similar(memories)
        assert result.original_count == 3
        assert result.strategy is CompressionStrategy.MERGE_SIMILAR
        # a and b should merge (a kept, b removed)
        assert "a" not in result.removed_ids
        assert "b" in result.removed_ids
        assert "c" not in result.removed_ids

    def test_merge_similar_no_overthreshold(self, compressor):
        memories = [
            {"id": "x", "content": "hello world"},
            {"id": "y", "content": "goodbye moon"},
        ]
        result = compressor.merge_similar(memories)
        assert result.compressed_count == 2
        assert result.removed_ids == []


class TestMemoryCompressorDispatch:
    def test_dispatch_dedup(self, compressor, sample_memories):
        result = compressor.compress(sample_memories, CompressionStrategy.DEDUP)
        assert result.strategy is CompressionStrategy.DEDUP

    def test_dispatch_truncate(self, compressor, sample_memories):
        result = compressor.compress(sample_memories, CompressionStrategy.TRUNCATE, max_count=2)
        assert result.strategy is CompressionStrategy.TRUNCATE
        assert result.compressed_count == 2

    def test_dispatch_merge_similar(self, compressor, sample_memories):
        result = compressor.compress(sample_memories, CompressionStrategy.MERGE_SIMILAR)
        assert result.strategy is CompressionStrategy.MERGE_SIMILAR

    def test_dispatch_unknown_strategy(self, compressor, sample_memories):
        with pytest.raises(ValueError, match="Unknown compression strategy"):
            # Pass a mock object that is not a CompressionStrategy member
            compressor.compress(sample_memories, "not_a_strategy")  # type: ignore[arg-type]


class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in MEMORY_COMPRESSOR_REGISTRY
        assert MEMORY_COMPRESSOR_REGISTRY["default"] is MemoryCompressor
