"""Tests for memory_fusion — multi-source memory fusion."""
from __future__ import annotations

from src.memory.memory_fusion import (
    MemoryFusion,
    MemorySource,
    FusionStrategy,
    MemoryFragment,
    fuse_weighted_average,
    fuse_most_recent,
    fuse_union,
)


class TestMemoryFragment:
    def test_fragment_creation(self):
        f = MemoryFragment(content="key info", source="episodic", confidence=0.9)
        assert f.content == "key info"
        assert f.source == "episodic"
        assert f.confidence == 0.9


class TestMemorySource:
    def test_add_and_get_fragments(self):
        src = MemorySource("working")
        src.add(MemoryFragment("a", "working", 0.8))
        src.add(MemoryFragment("b", "working", 0.6))
        assert len(src.get_all()) == 2

    def test_source_name(self):
        src = MemorySource("semantic")
        assert src.name == "semantic"


class TestFuseWeightedAverage:
    def test_weighted_average(self):
        fragments = [
            MemoryFragment("x", "src1", 0.9),
            MemoryFragment("y", "src2", 0.1),
        ]
        result = fuse_weighted_average(fragments)
        assert "x" in result.content


class TestFuseMostRecent:
    def test_most_recent_by_timestamp(self):
        fragments = [
            MemoryFragment("old", "src", 0.5, timestamp=100),
            MemoryFragment("new", "src", 0.5, timestamp=200),
        ]
        result = fuse_most_recent(fragments)
        assert result.content == "new"


class TestFuseUnion:
    def test_union_deduplicates(self):
        fragments = [
            MemoryFragment("hello", "src1", 0.9),
            MemoryFragment("hello", "src2", 0.8),
            MemoryFragment("world", "src3", 0.7),
        ]
        result = fuse_union(fragments)
        assert len(result.content.split("\n")) == 2


class TestMemoryFusion:
    def test_fuse_default_weighted(self):
        fusion = MemoryFusion()
        fusion.add_fragment(MemoryFragment("important", "episodic", 0.95))
        fusion.add_fragment(MemoryFragment("less important", "semantic", 0.1))
        result = fusion.fuse()
        assert result is not None
        assert "important" in result.content

    def test_fuse_most_recent_strategy(self):
        fusion = MemoryFusion(strategy=FusionStrategy.MOST_RECENT)
        fusion.add_fragment(MemoryFragment("old", "episodic", 0.9, timestamp=100))
        fusion.add_fragment(MemoryFragment("fresh", "working", 0.3, timestamp=500))
        result = fusion.fuse()
        assert result.content == "fresh"

    def test_fuse_union_strategy(self):
        fusion = MemoryFusion(strategy=FusionStrategy.UNION)
        fusion.add_fragment(MemoryFragment("a", "e", 0.8))
        fusion.add_fragment(MemoryFragment("b", "s", 0.7))
        result = fusion.fuse()
        assert "a" in result.content
        assert "b" in result.content

    def test_fuse_empty_returns_none(self):
        fusion = MemoryFusion()
        assert fusion.fuse() is None

    def test_source_isolation(self):
        fusion = MemoryFusion()
        fusion.add_fragment(MemoryFragment("from episodic", "episodic", 0.9))
        fusion.add_fragment(MemoryFragment("from semantic", "semantic", 0.8))
        episodics = fusion.get_by_source("episodic")
        assert len(episodics) == 1
        assert episodics[0].content == "from episodic"
