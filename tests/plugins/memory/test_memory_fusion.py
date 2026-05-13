"""Tests: plugins/memory/memory_fusion.py — Memory source fusion with multiple strategies."""

from __future__ import annotations

import time

import pytest

from plugins.memory.memory_fusion import (
    MEMORY_FUSION,
    FusionStrategy,
    MemoryFragment,
    MemoryFusion,
    MemorySource,
    fuse_most_recent,
    fuse_union,
    fuse_weighted_average,
)


@pytest.fixture
def fusion():
    return MemoryFusion()


class TestFusionStrategy:
    def test_weighted_average(self):
        assert FusionStrategy.WEIGHTED_AVERAGE.value == "weighted"

    def test_most_recent(self):
        assert FusionStrategy.MOST_RECENT.value == "recent"

    def test_union(self):
        assert FusionStrategy.UNION.value == "union"


class TestMemoryFragment:
    def test_fragment_defaults(self):
        frag = MemoryFragment(content="hello", source="test")
        assert frag.content == "hello"
        assert frag.source == "test"
        assert frag.confidence == 0.5
        assert frag.timestamp > 0

    def test_fragment_explicit_fields(self):
        ts = 999.0
        frag = MemoryFragment(content="hi", source="s", confidence=0.9, timestamp=ts)
        assert frag.timestamp == ts
        assert frag.confidence == 0.9


class TestMemorySource:
    def test_add_and_get_all(self):
        src = MemorySource("src_a")
        f1 = MemoryFragment("a", "src_a", 0.5)
        f2 = MemoryFragment("b", "src_a", 0.7)
        src.add(f1)
        src.add(f2)
        assert src.get_all() == [f1, f2]

    def test_get_all_returns_copy(self):
        src = MemorySource("s")
        frag = MemoryFragment("c", "s")
        src.add(frag)
        result = src.get_all()
        result.clear()
        assert len(src.get_all()) == 1


class TestStandaloneFusionFunctions:
    def test_fuse_weighted_average_empty(self):
        result = fuse_weighted_average([])
        assert result.content == ""
        assert result.confidence == 0.0
        assert result.source == "fusion"

    def test_fuse_weighted_average_picks_highest_confidence(self):
        frags = [
            MemoryFragment("low", "s", confidence=0.2),
            MemoryFragment("high", "s", confidence=0.9),
            MemoryFragment("mid", "s", confidence=0.5),
        ]
        result = fuse_weighted_average(frags)
        assert result.content == "high"
        assert result.confidence == 0.9

    def test_fuse_most_recent_empty(self):
        result = fuse_most_recent([])
        assert result.content == ""
        assert result.confidence == 0.0

    def test_fuse_most_recent_picks_newest(self):
        now = time.time()
        frags = [
            MemoryFragment("old", "s", timestamp=now - 100),
            MemoryFragment("newest", "s", timestamp=now),
            MemoryFragment("middle", "s", timestamp=now - 50),
        ]
        result = fuse_most_recent(frags)
        assert result.content == "newest"

    def test_fuse_union_combines_unique(self):
        frags = [
            MemoryFragment("a", "s", confidence=0.5),
            MemoryFragment("b", "s", confidence=0.6),
            MemoryFragment("a", "s", confidence=0.8),  # duplicate content
        ]
        result = fuse_union(frags)
        lines = result.content.split("\n")
        assert "a" in lines
        assert "b" in lines
        assert len(lines) == 2  # a appears only once

    def test_fuse_union_avg_confidence(self):
        frags = [
            MemoryFragment("x", "s", confidence=0.2),
            MemoryFragment("y", "s", confidence=0.8),
        ]
        result = fuse_union(frags)
        assert result.confidence == 0.5


class TestMemoryFusion:
    def test_default_strategy(self):
        fusion = MemoryFusion()
        assert fusion.strategy == FusionStrategy.WEIGHTED_AVERAGE

    def test_custom_strategy(self):
        fusion = MemoryFusion(strategy=FusionStrategy.MOST_RECENT)
        assert fusion.strategy == FusionStrategy.MOST_RECENT

    def test_add_fragment(self, fusion):
        frag = MemoryFragment("test", "src_a", 0.7)
        fusion.add_fragment(frag)
        assert len(fusion._fragments) == 1
        assert "src_a" in fusion._sources

    def test_add_fragment_creates_source(self, fusion):
        frag = MemoryFragment("content", "new_src")
        fusion.add_fragment(frag)
        assert "new_src" in fusion._sources

    def test_fuse_empty_returns_none(self, fusion):
        assert fusion.fuse() is None

    def test_fuse_with_weighted_strategy(self):
        fusion = MemoryFusion(strategy=FusionStrategy.WEIGHTED_AVERAGE)
        fusion.add_fragment(MemoryFragment("low", "s", confidence=0.1))
        fusion.add_fragment(MemoryFragment("high", "s", confidence=0.9))
        result = fusion.fuse()
        assert result is not None
        assert result.content == "high"

    def test_fuse_with_most_recent_strategy(self):
        now = time.time()
        fusion = MemoryFusion(strategy=FusionStrategy.MOST_RECENT)
        fusion.add_fragment(MemoryFragment("old", "s", timestamp=now - 10))
        fusion.add_fragment(MemoryFragment("new", "s", timestamp=now))
        result = fusion.fuse()
        assert result is not None
        assert result.content == "new"

    def test_fuse_with_union_strategy(self):
        fusion = MemoryFusion(strategy=FusionStrategy.UNION)
        fusion.add_fragment(MemoryFragment("a", "s"))
        fusion.add_fragment(MemoryFragment("b", "s"))
        result = fusion.fuse()
        assert result is not None
        assert "a" in result.content
        assert "b" in result.content

    def test_get_by_source(self, fusion):
        fusion.add_fragment(MemoryFragment("x", "src_a"))
        fusion.add_fragment(MemoryFragment("y", "src_a"))
        fusion.add_fragment(MemoryFragment("z", "src_b"))
        result = fusion.get_by_source("src_a")
        assert len(result) == 2

    def test_get_by_source_missing(self, fusion):
        assert fusion.get_by_source("ghost") == []


class TestSingleton:
    def test_module_singleton(self):
        assert isinstance(MEMORY_FUSION, MemoryFusion)
