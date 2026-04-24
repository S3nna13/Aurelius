"""Tests for src/agent/memory_writer.py"""

from __future__ import annotations

import pytest

from src.agent.memory_writer import (
    MEMORY_WRITER_REGISTRY,
    MemoryRecord,
    MemoryType,
    MemoryWriter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_writer(**kwargs) -> MemoryWriter:
    return MemoryWriter(**kwargs)


# ---------------------------------------------------------------------------
# MemoryRecord – frozen / auto-id
# ---------------------------------------------------------------------------

class TestMemoryRecord:
    def test_create_returns_memory_record(self):
        rec = MemoryRecord.create(
            memory_type=MemoryType.FACT,
            content="sky is blue",
            importance=0.8,
            timestamp=1.0,
        )
        assert isinstance(rec, MemoryRecord)

    def test_auto_record_id_is_10_chars(self):
        rec = MemoryRecord.create(
            memory_type=MemoryType.FACT,
            content="x",
            importance=0.5,
            timestamp=0.0,
        )
        assert len(rec.record_id) == 10

    def test_explicit_record_id_preserved(self):
        rec = MemoryRecord.create(
            memory_type=MemoryType.FACT,
            content="x",
            importance=0.5,
            timestamp=0.0,
            record_id="myid123456",
        )
        assert rec.record_id == "myid123456"

    def test_frozen_record_cannot_be_mutated(self):
        rec = MemoryRecord.create(
            memory_type=MemoryType.FACT,
            content="immutable",
            importance=0.5,
            timestamp=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            rec.content = "changed"  # type: ignore[misc]

    def test_default_tags_empty_list(self):
        rec = MemoryRecord.create(
            memory_type=MemoryType.OBSERVATION,
            content="no tags",
            importance=0.3,
            timestamp=0.0,
        )
        assert rec.tags == []

    def test_tags_stored_correctly(self):
        rec = MemoryRecord.create(
            memory_type=MemoryType.OBSERVATION,
            content="tagged",
            importance=0.3,
            timestamp=0.0,
            tags=["alpha", "beta"],
        )
        assert "alpha" in rec.tags
        assert "beta" in rec.tags


# ---------------------------------------------------------------------------
# MemoryWriter – write
# ---------------------------------------------------------------------------

class TestMemoryWriterWrite:
    def test_write_returns_memory_record(self):
        w = make_writer()
        rec = w.write(MemoryType.OBSERVATION, "saw something")
        assert isinstance(rec, MemoryRecord)

    def test_write_stores_record(self):
        w = make_writer()
        rec = w.write(MemoryType.DECISION, "go left")
        assert len(w) == 1
        assert w.recall()[0].record_id == rec.record_id

    def test_write_observation_convenience(self):
        w = make_writer()
        rec = w.write_observation("noticed event")
        assert rec.memory_type is MemoryType.OBSERVATION

    def test_write_decision_convenience(self):
        w = make_writer()
        rec = w.write_decision("chose path A")
        assert rec.memory_type is MemoryType.DECISION

    def test_write_observation_passes_kwargs(self):
        w = make_writer()
        rec = w.write_observation("tagged obs", importance=0.9, tags=["x"])
        assert rec.importance == 0.9
        assert "x" in rec.tags

    def test_write_decision_passes_kwargs(self):
        w = make_writer()
        rec = w.write_decision("critical call", importance=1.0)
        assert rec.importance == 1.0

    def test_write_default_importance_is_half(self):
        w = make_writer()
        rec = w.write(MemoryType.FACT, "default importance")
        assert rec.importance == 0.5

    def test_write_timestamp_is_positive(self):
        w = make_writer()
        rec = w.write(MemoryType.PLAN, "plan step")
        assert rec.timestamp >= 0

    def test_write_raises_at_max_records(self):
        w = make_writer(max_records=2)
        w.write(MemoryType.FACT, "one")
        w.write(MemoryType.FACT, "two")
        with pytest.raises(ValueError, match="capacity"):
            w.write(MemoryType.FACT, "three")

    def test_write_at_exactly_max_succeeds(self):
        w = make_writer(max_records=3)
        w.write(MemoryType.FACT, "a")
        w.write(MemoryType.FACT, "b")
        rec = w.write(MemoryType.FACT, "c")
        assert isinstance(rec, MemoryRecord)


# ---------------------------------------------------------------------------
# MemoryWriter – recall
# ---------------------------------------------------------------------------

class TestMemoryWriterRecall:
    def test_recall_all_returns_all(self):
        w = make_writer()
        w.write(MemoryType.FACT, "f1")
        w.write(MemoryType.OBSERVATION, "o1")
        assert len(w.recall()) == 2

    def test_recall_sorted_by_importance_desc(self):
        w = make_writer()
        w.write(MemoryType.FACT, "low",  importance=0.1)
        w.write(MemoryType.FACT, "high", importance=0.9)
        w.write(MemoryType.FACT, "mid",  importance=0.5)
        results = w.recall()
        importances = [r.importance for r in results]
        assert importances == sorted(importances, reverse=True)

    def test_recall_by_tag(self):
        w = make_writer()
        w.write(MemoryType.FACT, "tagged", tags=["env"])
        w.write(MemoryType.FACT, "untagged")
        results = w.recall(tags=["env"])
        assert len(results) == 1
        assert "env" in results[0].tags

    def test_recall_by_multiple_tags_all_required(self):
        w = make_writer()
        w.write(MemoryType.FACT, "both",  tags=["a", "b"])
        w.write(MemoryType.FACT, "one_a", tags=["a"])
        results = w.recall(tags=["a", "b"])
        assert len(results) == 1

    def test_recall_by_memory_type(self):
        w = make_writer()
        w.write(MemoryType.DECISION, "d1")
        w.write(MemoryType.OBSERVATION, "o1")
        results = w.recall(memory_type=MemoryType.DECISION)
        assert all(r.memory_type is MemoryType.DECISION for r in results)
        assert len(results) == 1

    def test_recall_min_importance_filter(self):
        w = make_writer()
        w.write(MemoryType.FACT, "low",  importance=0.2)
        w.write(MemoryType.FACT, "high", importance=0.8)
        results = w.recall(min_importance=0.5)
        assert len(results) == 1
        assert results[0].importance >= 0.5

    def test_recall_combined_filters(self):
        w = make_writer()
        w.write(MemoryType.FACT, "match", importance=0.9, tags=["prod"])
        w.write(MemoryType.FACT, "low_imp", importance=0.1, tags=["prod"])
        w.write(MemoryType.OBSERVATION, "wrong_type", importance=0.9, tags=["prod"])
        results = w.recall(
            tags=["prod"],
            memory_type=MemoryType.FACT,
            min_importance=0.5,
        )
        assert len(results) == 1
        assert results[0].content == "match"

    def test_recall_empty_store(self):
        w = make_writer()
        assert w.recall() == []


# ---------------------------------------------------------------------------
# MemoryWriter – forget
# ---------------------------------------------------------------------------

class TestMemoryWriterForget:
    def test_forget_existing_record_returns_true(self):
        w = make_writer()
        rec = w.write(MemoryType.FACT, "to forget")
        assert w.forget(rec.record_id) is True

    def test_forget_removes_record(self):
        w = make_writer()
        rec = w.write(MemoryType.FACT, "ephemeral")
        w.forget(rec.record_id)
        assert len(w) == 0

    def test_forget_nonexistent_returns_false(self):
        w = make_writer()
        assert w.forget("doesnotexist") is False

    def test_forget_frees_capacity_for_new_write(self):
        w = make_writer(max_records=1)
        rec = w.write(MemoryType.FACT, "a")
        w.forget(rec.record_id)
        new_rec = w.write(MemoryType.FACT, "b")  # should not raise
        assert isinstance(new_rec, MemoryRecord)


# ---------------------------------------------------------------------------
# MemoryWriter – len & stats
# ---------------------------------------------------------------------------

class TestMemoryWriterStats:
    def test_len_empty(self):
        w = make_writer()
        assert len(w) == 0

    def test_len_after_writes(self):
        w = make_writer()
        w.write(MemoryType.FACT, "a")
        w.write(MemoryType.FACT, "b")
        assert len(w) == 2

    def test_stats_total_count(self):
        w = make_writer()
        w.write(MemoryType.FACT, "x")
        assert w.stats()["total"] == 1

    def test_stats_by_type_keys(self):
        w = make_writer()
        stats = w.stats()
        for mt in MemoryType:
            assert mt.name in stats["by_type"]

    def test_stats_by_type_counts(self):
        w = make_writer()
        w.write(MemoryType.DECISION, "d1")
        w.write(MemoryType.DECISION, "d2")
        w.write(MemoryType.FACT, "f1")
        stats = w.stats()
        assert stats["by_type"]["DECISION"] == 2
        assert stats["by_type"]["FACT"] == 1

    def test_stats_mean_importance_empty(self):
        w = make_writer()
        assert w.stats()["mean_importance"] == 0.0

    def test_stats_mean_importance_correct(self):
        w = make_writer()
        w.write(MemoryType.FACT, "a", importance=0.2)
        w.write(MemoryType.FACT, "b", importance=0.8)
        mean = w.stats()["mean_importance"]
        assert abs(mean - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_contains_default(self):
        assert "default" in MEMORY_WRITER_REGISTRY

    def test_registry_default_is_memory_writer_class(self):
        assert MEMORY_WRITER_REGISTRY["default"] is MemoryWriter

    def test_registry_default_is_instantiable(self):
        cls = MEMORY_WRITER_REGISTRY["default"]
        obj = cls()
        assert isinstance(obj, MemoryWriter)
