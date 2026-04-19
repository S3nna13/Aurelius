"""Integration tests for the chat.conversation_memory surface."""

from __future__ import annotations

import src.chat as chat_pkg
from src.chat import ConversationMemory, Fact, InMemoryStore, JSONFileStore
from src.retrieval.bm25_retriever import BM25Retriever


def test_public_symbols_exposed():
    for name in ("ConversationMemory", "Fact", "InMemoryStore", "JSONFileStore"):
        assert hasattr(chat_pkg, name), f"{name} missing from src.chat"


def test_existing_chat_template_registry_intact():
    # Sanity: conversation_memory must not have clobbered sibling registries.
    assert "chatml" in chat_pkg.CHAT_TEMPLATE_REGISTRY
    assert "llama3" in chat_pkg.CHAT_TEMPLATE_REGISTRY
    assert "harmony" in chat_pkg.CHAT_TEMPLATE_REGISTRY
    assert "chatml" in chat_pkg.MESSAGE_FORMAT_REGISTRY
    assert "harmony" in chat_pkg.MESSAGE_FORMAT_REGISTRY
    assert "tool_result" in chat_pkg.MESSAGE_FORMAT_REGISTRY


def test_round_trip_jsonfile_with_bm25(tmp_path):
    path = tmp_path / "conv_mem.json"

    # Session 1: record facts through the public surface, persist to disk.
    mem = ConversationMemory(JSONFileStore(path), bm25_retriever=BM25Retriever)
    mem.record_fact(
        "agent-session-1",
        "User prefers typed Python with mypy strict mode",
        importance=0.9,
        tags=["preference"],
    )
    mem.record_fact(
        "agent-session-1",
        "Deployment target is a Kubernetes cluster in us-east-1",
        importance=0.7,
        tags=["infra"],
    )
    mem.record_fact(
        "agent-session-1",
        "Current build uses Bazel with remote execution",
        importance=0.5,
        tags=["build"],
    )
    mem.record_fact(
        "agent-session-2",
        "Different session completely unrelated fact",
        importance=0.4,
    )

    # Session 2: fresh process would re-instantiate from disk.
    mem2 = ConversationMemory(JSONFileStore(path), bm25_retriever=BM25Retriever)
    hits = mem2.retrieve("agent-session-1", "mypy typed python", k=2)
    assert len(hits) >= 1
    assert any("mypy" in f.content for f in hits)

    # Namespace isolation survives reload.
    s2 = mem2.store.list_by_namespace("agent-session-2")
    assert len(s2) == 1
    assert s2[0].content.startswith("Different session")

    # Top-by-importance still ranks correctly post-reload.
    top = mem2.top_by_importance("agent-session-1", k=1)
    assert top[0].importance == 0.9


def test_fact_and_inmemory_store_via_surface():
    store = InMemoryStore()
    store.add(
        Fact(
            id="manual-1",
            namespace="n",
            content="hand-built fact",
            created_at=0.0,
            importance=0.5,
        )
    )
    mem = ConversationMemory(store)
    assert mem.get("manual-1").content == "hand-built fact"
