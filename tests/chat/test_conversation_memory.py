"""Unit tests for src.chat.conversation_memory."""

from __future__ import annotations

import json
import os
import time

import pytest

from src.chat.conversation_memory import (
    ConversationMemory,
    Fact,
    InMemoryStore,
    JSONFileStore,
    MalformedMemoryFileError,
)
from src.retrieval.bm25_retriever import BM25Retriever


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


class _FakeClock:
    def __init__(self, t: float = 1_700_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _SeqIdFactory:
    def __init__(self) -> None:
        self.i = 0

    def __call__(self) -> str:
        self.i += 1
        return f"id-{self.i:06d}"


def _new_mem(bm25=None, *, clock=None, id_factory=None) -> ConversationMemory:
    return ConversationMemory(
        InMemoryStore(),
        bm25_retriever=bm25,
        clock=clock,
        id_factory=id_factory,
    )


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_record_and_retrieve_by_query():
    mem = _new_mem()
    mem.record_fact("sess-1", "user prefers pytest for python projects")
    mem.record_fact("sess-1", "deployment target is kubernetes")
    mem.record_fact("sess-1", "cloud budget is three hundred fifty dollars")

    hits = mem.retrieve("sess-1", "pytest", k=5)
    assert len(hits) == 1
    assert "pytest" in hits[0].content


def test_namespace_isolation():
    mem = _new_mem()
    mem.record_fact("alpha", "shared keyword apple")
    mem.record_fact("beta", "shared keyword apple")

    alpha_hits = mem.retrieve("alpha", "apple", k=5)
    beta_hits = mem.retrieve("beta", "apple", k=5)
    assert len(alpha_hits) == 1
    assert len(beta_hits) == 1
    assert alpha_hits[0].namespace == "alpha"
    assert beta_hits[0].namespace == "beta"
    assert alpha_hits[0].id != beta_hits[0].id


def test_ttl_expiry_hides_facts():
    clock = _FakeClock()
    mem = _new_mem(clock=clock)
    mem.record_fact("n", "ephemeral note about error", ttl=10.0)
    mem.record_fact("n", "durable note about error")

    assert len(mem.retrieve("n", "error", k=5)) == 2
    clock.advance(11.0)
    live = mem.retrieve("n", "error", k=5)
    assert len(live) == 1
    assert live[0].content == "durable note about error"


def test_top_by_importance_ordering():
    mem = _new_mem()
    mem.record_fact("n", "low", importance=0.1)
    mem.record_fact("n", "high", importance=0.9)
    mem.record_fact("n", "mid", importance=0.5)

    top = mem.top_by_importance("n", k=3)
    assert [f.content for f in top] == ["high", "mid", "low"]


def test_prune_expired_returns_count():
    clock = _FakeClock()
    mem = _new_mem(clock=clock)
    mem.record_fact("n", "a", ttl=5.0)
    mem.record_fact("n", "b", ttl=5.0)
    mem.record_fact("n", "c")  # no ttl

    clock.advance(10.0)
    removed = mem.prune_expired("n")
    assert removed == 2
    remaining = mem.store.list_by_namespace("n")
    assert [f.content for f in remaining] == ["c"]


def test_jsonfile_store_survives_reload(tmp_path):
    path = tmp_path / "mem.json"
    mem = ConversationMemory(JSONFileStore(path))
    mem.record_fact("n", "first durable fact", importance=0.7, tags=["x"])
    mem.record_fact("n", "second durable fact", ttl=3600.0)

    mem2 = ConversationMemory(JSONFileStore(path))
    facts = mem2.store.list_by_namespace("n")
    contents = sorted(f.content for f in facts)
    assert contents == ["first durable fact", "second durable fact"]
    tagged = [f for f in facts if f.tags == ["x"]]
    assert len(tagged) == 1
    assert tagged[0].importance == 0.7


def test_jsonfile_atomic_write_no_corruption(tmp_path):
    path = tmp_path / "mem.json"
    store = JSONFileStore(path)
    mem = ConversationMemory(store)
    for i in range(20):
        mem.record_fact("n", f"fact number {i}", importance=0.5)

    # The only file at the target path should be valid JSON -- the
    # atomic rename leaves no half-written file behind. Any stale temp
    # siblings would be hidden (dotfile prefix) but the target must parse.
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert len(data["facts"]) == 20

    # No .tmp files should remain after a clean sequence of writes.
    leftovers = [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]
    assert leftovers == []


def test_delete_fact():
    mem = _new_mem()
    f = mem.record_fact("n", "doomed fact")
    assert mem.delete(f.id) is True
    assert mem.get(f.id) is None
    assert mem.delete(f.id) is False


def test_invalid_importance_raises():
    mem = _new_mem()
    with pytest.raises(ValueError):
        mem.record_fact("n", "c", importance=-0.1)
    with pytest.raises(ValueError):
        mem.record_fact("n", "c", importance=1.5)


def test_uuids_unique_across_records():
    mem = _new_mem()
    ids = {mem.record_fact("n", f"fact-{i}").id for i in range(500)}
    assert len(ids) == 500


def test_query_with_bm25_uses_bm25_ranking():
    mem = _new_mem(bm25=BM25Retriever)
    mem.record_fact("n", "python unit tests with pytest are great")
    mem.record_fact("n", "deployment uses kubernetes manifests")
    mem.record_fact("n", "random unrelated document")

    hits = mem.retrieve("n", "pytest", k=1)
    assert len(hits) == 1
    assert "pytest" in hits[0].content


def test_query_without_bm25_falls_back_to_substring():
    mem = _new_mem()  # no bm25
    mem.record_fact("n", "The CAPS keyword Alpha matters")
    mem.record_fact("n", "beta unrelated")

    hits = mem.retrieve("n", "alpha", k=5)
    assert len(hits) == 1
    assert "Alpha" in hits[0].content


def test_query_1000_facts_under_200ms():
    mem = _new_mem()
    for i in range(1000):
        mem.record_fact(
            "n",
            f"fact {i} about topic {'pytest' if i % 97 == 0 else 'misc'}",
        )
    start = time.perf_counter()
    hits = mem.retrieve("n", "pytest", k=10)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.2, f"retrieve took {elapsed*1000:.1f}ms"
    assert 1 <= len(hits) <= 10


def test_determinism_given_fixed_clock_and_ids():
    def run():
        clock = _FakeClock(1_700_000_000.0)
        ids = _SeqIdFactory()
        mem = _new_mem(clock=clock, id_factory=ids)
        mem.record_fact("n", "alpha beta", importance=0.3)
        clock.advance(1.0)
        mem.record_fact("n", "alpha gamma", importance=0.9)
        clock.advance(1.0)
        mem.record_fact("n", "delta epsilon", importance=0.5)
        return [
            (f.id, f.content, f.created_at, f.importance)
            for f in mem.top_by_importance("n", k=3)
        ]

    assert run() == run()


def test_malformed_json_on_disk_raises(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(MalformedMemoryFileError):
        JSONFileStore(path)


def test_malformed_schema_on_disk_raises(tmp_path):
    path = tmp_path / "mem.json"
    path.write_text(json.dumps({"not_facts": []}), encoding="utf-8")
    with pytest.raises(MalformedMemoryFileError):
        JSONFileStore(path)


def test_fact_dataclass_validation():
    with pytest.raises(ValueError):
        Fact(id="", namespace="n", content="c", created_at=0.0)
    with pytest.raises(ValueError):
        Fact(id="x", namespace="n", content="c", created_at=0.0, importance=2.0)
    with pytest.raises(ValueError):
        Fact(
            id="x",
            namespace="n",
            content="c",
            created_at=0.0,
            ttl_seconds=0.0,
        )


def test_empty_query_returns_nothing():
    mem = _new_mem()
    mem.record_fact("n", "anything")
    assert mem.retrieve("n", "   ", k=5) == []
