"""
Tests for src/memory/memory_compressor.py

Coverage: CompressionStrategy, CompressionResult, MemoryCompressor,
MEMORY_COMPRESSOR_REGISTRY.
At least 28 test functions.
"""

import pytest

from src.memory.memory_compressor import (
    MEMORY_COMPRESSOR_REGISTRY,
    CompressionResult,
    CompressionStrategy,
    MemoryCompressor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mems(*contents: str) -> list[dict]:
    """Create simple memory dicts with sequential ids."""
    return [{"id": f"m{i}", "content": c} for i, c in enumerate(contents)]


# ---------------------------------------------------------------------------
# CompressionStrategy enum
# ---------------------------------------------------------------------------

class TestCompressionStrategy:
    def test_enum_members_exist(self):
        assert CompressionStrategy.DEDUP
        assert CompressionStrategy.TRUNCATE
        assert CompressionStrategy.MERGE_SIMILAR

    def test_enum_values(self):
        assert CompressionStrategy.DEDUP.value == "dedup"
        assert CompressionStrategy.TRUNCATE.value == "truncate"
        assert CompressionStrategy.MERGE_SIMILAR.value == "merge_similar"


# ---------------------------------------------------------------------------
# CompressionResult dataclass
# ---------------------------------------------------------------------------

class TestCompressionResult:
    def test_result_creation(self):
        r = CompressionResult(
            original_count=5,
            compressed_count=3,
            strategy=CompressionStrategy.DEDUP,
            removed_ids=["a", "b"],
        )
        assert r.original_count == 5
        assert r.compressed_count == 3
        assert r.strategy is CompressionStrategy.DEDUP
        assert r.removed_ids == ["a", "b"]

    def test_result_frozen(self):
        r = CompressionResult(3, 2, CompressionStrategy.TRUNCATE, [])
        with pytest.raises((AttributeError, TypeError)):
            r.original_count = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MemoryCompressor: deduplicate
# ---------------------------------------------------------------------------

class TestDeduplicate:
    def test_dedup_exact_duplicates(self):
        mems = make_mems("hello", "hello", "world")
        c = MemoryCompressor()
        result = c.deduplicate(mems)
        assert result.original_count == 3
        assert result.compressed_count == 2
        assert result.strategy is CompressionStrategy.DEDUP

    def test_dedup_case_insensitive(self):
        mems = make_mems("Hello", "HELLO", "world")
        c = MemoryCompressor()
        result = c.deduplicate(mems)
        assert result.compressed_count == 2

    def test_dedup_whitespace_stripped(self):
        mems = make_mems("  hello  ", "hello", "other")
        c = MemoryCompressor()
        result = c.deduplicate(mems)
        assert result.compressed_count == 2

    def test_dedup_no_duplicates(self):
        mems = make_mems("alpha", "beta", "gamma")
        c = MemoryCompressor()
        result = c.deduplicate(mems)
        assert result.original_count == 3
        assert result.compressed_count == 3
        assert result.removed_ids == []

    def test_dedup_removed_ids_correct(self):
        mems = make_mems("dup", "dup", "unique")
        c = MemoryCompressor()
        result = c.deduplicate(mems)
        # second occurrence (m1) should be removed
        assert "m1" in result.removed_ids
        assert "m0" not in result.removed_ids

    def test_dedup_empty_input(self):
        c = MemoryCompressor()
        result = c.deduplicate([])
        assert result.original_count == 0
        assert result.compressed_count == 0
        assert result.removed_ids == []

    def test_dedup_last_result_memories_populated(self):
        mems = make_mems("a", "a", "b")
        c = MemoryCompressor()
        c.deduplicate(mems)
        assert len(c.last_result_memories) == 2

    def test_dedup_last_result_memories_keeps_first(self):
        mems = [{"id": "first", "content": "same"}, {"id": "second", "content": "same"}]
        c = MemoryCompressor()
        c.deduplicate(mems)
        assert c.last_result_memories[0]["id"] == "first"

    def test_dedup_all_duplicates(self):
        mems = make_mems("x", "x", "x")
        c = MemoryCompressor()
        result = c.deduplicate(mems)
        assert result.compressed_count == 1
        assert len(result.removed_ids) == 2


# ---------------------------------------------------------------------------
# MemoryCompressor: truncate
# ---------------------------------------------------------------------------

class TestTruncate:
    def test_truncate_keeps_first_n(self):
        mems = make_mems("a", "b", "c", "d", "e")
        c = MemoryCompressor()
        result = c.truncate(mems, max_count=3)
        assert result.compressed_count == 3
        assert result.strategy is CompressionStrategy.TRUNCATE
        assert c.last_result_memories[0]["content"] == "a"
        assert c.last_result_memories[2]["content"] == "c"

    def test_truncate_removes_excess(self):
        mems = make_mems("a", "b", "c")
        c = MemoryCompressor()
        result = c.truncate(mems, max_count=1)
        assert result.removed_ids == ["m1", "m2"]

    def test_truncate_max_count_equals_len(self):
        mems = make_mems("a", "b")
        c = MemoryCompressor()
        result = c.truncate(mems, max_count=2)
        assert result.compressed_count == 2
        assert result.removed_ids == []

    def test_truncate_max_count_zero(self):
        mems = make_mems("a", "b")
        c = MemoryCompressor()
        result = c.truncate(mems, max_count=0)
        assert result.compressed_count == 0
        assert len(result.removed_ids) == 2

    def test_truncate_empty_input(self):
        c = MemoryCompressor()
        result = c.truncate([], max_count=5)
        assert result.original_count == 0
        assert result.compressed_count == 0

    def test_truncate_last_result_memories(self):
        mems = make_mems("p", "q", "r")
        c = MemoryCompressor()
        c.truncate(mems, max_count=2)
        assert len(c.last_result_memories) == 2


# ---------------------------------------------------------------------------
# MemoryCompressor: merge_similar
# ---------------------------------------------------------------------------

class TestMergeSimilar:
    def test_merge_similar_identical_content_merged(self):
        mems = make_mems("hello world", "hello world", "something else entirely")
        c = MemoryCompressor(similarity_threshold=0.8)
        result = c.merge_similar(mems)
        assert result.compressed_count == 2

    def test_merge_similar_dissimilar_kept_separate(self):
        mems = make_mems("abcdef", "xyz123", "QQQQQ")
        c = MemoryCompressor(similarity_threshold=0.8)
        result = c.merge_similar(mems)
        assert result.compressed_count == 3

    def test_merge_similar_last_result_memories_populated(self):
        mems = make_mems("aaa", "aaa", "bbb")
        c = MemoryCompressor(similarity_threshold=0.5)
        c.merge_similar(mems)
        assert len(c.last_result_memories) == 2

    def test_merge_similar_keeps_first_representative(self):
        mems = [
            {"id": "rep", "content": "the quick brown fox"},
            {"id": "dup", "content": "the quick brown fox"},
        ]
        c = MemoryCompressor(similarity_threshold=0.8)
        c.merge_similar(mems)
        assert c.last_result_memories[0]["id"] == "rep"

    def test_merge_similar_empty_input(self):
        c = MemoryCompressor()
        result = c.merge_similar([])
        assert result.compressed_count == 0

    def test_merge_similar_single_item(self):
        mems = make_mems("only one")
        c = MemoryCompressor()
        result = c.merge_similar(mems)
        assert result.compressed_count == 1
        assert result.removed_ids == []


# ---------------------------------------------------------------------------
# MemoryCompressor: compress dispatcher
# ---------------------------------------------------------------------------

class TestCompress:
    def test_compress_dedup_strategy(self):
        mems = make_mems("same", "same", "different")
        c = MemoryCompressor()
        result = c.compress(mems, CompressionStrategy.DEDUP)
        assert result.strategy is CompressionStrategy.DEDUP
        assert result.compressed_count == 2

    def test_compress_truncate_strategy(self):
        mems = make_mems("a", "b", "c")
        c = MemoryCompressor()
        result = c.compress(mems, CompressionStrategy.TRUNCATE, max_count=2)
        assert result.strategy is CompressionStrategy.TRUNCATE
        assert result.compressed_count == 2

    def test_compress_merge_similar_strategy(self):
        mems = make_mems("hello", "hello", "world")
        c = MemoryCompressor(similarity_threshold=0.8)
        result = c.compress(mems, CompressionStrategy.MERGE_SIMILAR)
        assert result.strategy is CompressionStrategy.MERGE_SIMILAR

    def test_compress_unknown_strategy_raises(self):
        c = MemoryCompressor()
        with pytest.raises((ValueError, AttributeError)):
            c.compress([], "not_a_strategy")  # type: ignore[arg-type]

    def test_compress_last_result_memories_after_dedup(self):
        mems = make_mems("x", "x")
        c = MemoryCompressor()
        c.compress(mems, CompressionStrategy.DEDUP)
        assert len(c.last_result_memories) == 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_exists(self):
        assert isinstance(MEMORY_COMPRESSOR_REGISTRY, dict)

    def test_registry_has_default(self):
        assert "default" in MEMORY_COMPRESSOR_REGISTRY

    def test_registry_default_is_memory_compressor(self):
        assert MEMORY_COMPRESSOR_REGISTRY["default"] is MemoryCompressor

    def test_registry_default_instantiates(self):
        cls = MEMORY_COMPRESSOR_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, MemoryCompressor)
