"""Tests: plugins/memory/episodic_memory.py — event-stamped entries with recency scoring."""

from __future__ import annotations

from plugins.memory.episodic_memory import EpisodicMemory, MemoryEntry


class TestMemoryEntry:
    """MemoryEntry dataclass."""

    def test_entry_creation(self):
        entry = MemoryEntry(role="user", content="Hello world", importance=0.8)
        assert entry.role == "user"
        assert entry.content == "Hello world"
        assert entry.importance == 0.8
        assert len(entry.id) == 8
        assert entry.timestamp is not None

    def test_entry_default_importance(self):
        entry = MemoryEntry(role="assistant", content="Hi")
        assert entry.importance == 1.0


class TestEpisodicMemory:
    """Episodic memory store."""

    def test_store_and_retrieve_recent(self):
        mem = EpisodicMemory(max_entries=10)
        _e1 = mem.store("user", "First message")
        _e2 = mem.store("assistant", "Second reply")
        recent = mem.retrieve_recent(2)
        assert len(recent) == 2
        assert recent[0].content == "First message"
        assert recent[1].content == "Second reply"

    def test_store_evicts_oldest_over_capacity(self):
        mem = EpisodicMemory(max_entries=3)
        for i in range(5):
            mem.store("user", f"Message {i}")
        assert len(mem) == 3
        recent = mem.retrieve_recent(3)
        assert recent[0].content == "Message 2"
        assert recent[1].content == "Message 3"
        assert recent[2].content == "Message 4"

    def test_retrieve_recent_n_less_than_available(self):
        mem = EpisodicMemory(max_entries=10)
        for i in range(5):
            mem.store("user", f"Msg {i}")
        recent = mem.retrieve_recent(2)
        assert len(recent) == 2
        assert recent[0].content == "Msg 3"
        assert recent[1].content == "Msg 4"

    def test_retrieve_by_importance(self):
        mem = EpisodicMemory(max_entries=100)
        mem.store("user", "Low priority note", importance=0.2)
        mem.store("user", "High priority note", importance=0.9)
        mem.store("assistant", "Medium priority", importance=0.5)
        results = mem.retrieve_by_importance(threshold=0.5)
        assert len(results) == 2
        assert results[0].content == "High priority note"
        assert results[1].content == "Medium priority"

    def test_retrieve_by_importance_threshold(self):
        mem = EpisodicMemory(max_entries=100)
        mem.store("user", "Low", importance=0.1)
        mem.store("user", "High", importance=0.9)
        results = mem.retrieve_by_importance(threshold=0.5)
        assert len(results) == 1
        assert results[0].content == "High"

    def test_search(self):
        mem = EpisodicMemory(max_entries=100)
        mem.store("user", "The sky is blue today")
        mem.store("assistant", "That sounds nice")
        mem.store("user", "The clouds are white")
        results = mem.search("the sky")
        assert len(results) == 1
        assert results[0].content == "The sky is blue today"

    def test_search_case_insensitive(self):
        mem = EpisodicMemory(max_entries=100)
        mem.store("user", "Hello WORLD")
        results = mem.search("hello")
        assert len(results) == 1
        assert results[0].content == "Hello WORLD"

    def test_forget(self):
        mem = EpisodicMemory(max_entries=10)
        entry = mem.store("user", "Forget me")
        assert len(mem) == 1
        removed = mem.forget(entry.id)
        assert removed is True
        assert len(mem) == 0

    def test_forget_not_found(self):
        mem = EpisodicMemory(max_entries=10)
        removed = mem.forget("nonexistent-id")
        assert removed is False

    def test_clear(self):
        mem = EpisodicMemory(max_entries=10)
        mem.store("user", "One")
        mem.store("user", "Two")
        assert len(mem) == 2
        count = mem.clear()
        assert count == 2
        assert len(mem) == 0

    def test_empty_memory_retrieve_recent(self):
        mem = EpisodicMemory(max_entries=10)
        recent = mem.retrieve_recent(5)
        assert recent == []

    def test_empty_memory_search(self):
        mem = EpisodicMemory(max_entries=10)
        results = mem.search("anything")
        assert results == []

    def test_len(self):
        mem = EpisodicMemory(max_entries=10)
        assert len(mem) == 0
        mem.store("user", "First")
        assert len(mem) == 1
        mem.store("assistant", "Second")
        assert len(mem) == 2
