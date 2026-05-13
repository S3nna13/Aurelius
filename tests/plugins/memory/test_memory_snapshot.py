"""Tests: plugins/memory/memory_snapshot.py — Persistent serialization of memory stores with integrity checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plugins.memory.memory_snapshot import (
    MemorySnapshot,
    SnapshotCorruptedError,
    SnapshotHeader,
)


@pytest.fixture
def tmp_snapshot(tmp_path):
    return tmp_path / "snapshot.json"


class TestSnapshotHeader:
    def test_header_frozen(self):
        header = SnapshotHeader(version="1.0", hash="abc", compressed=True, store_type="test")
        assert header.version == "1.0"
        assert header.hash == "abc"
        assert header.compressed is True
        assert header.store_type == "test"


class TestSanitizePath:
    def test_resolve_path(self, tmp_path):
        p = MemorySnapshot._sanitize_path(str(tmp_path / "test.json"))
        assert isinstance(p, Path)

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="path traversal"):
            MemorySnapshot._sanitize_path("../etc/passwd")


class TestMemorySnapshotSaveLoad:
    def test_save_load_episodic_memory(self, tmp_path):
        from plugins.memory.episodic_memory import EpisodicMemory

        mem = EpisodicMemory()
        mem.store("user", "Hello world")
        mem.store("assistant", "Hi there")

        path = tmp_path / "episodic.json"
        MemorySnapshot.save(mem, str(path), compressed=False)
        loaded = MemorySnapshot.load(str(path))
        assert isinstance(loaded, EpisodicMemory)

    def test_save_load_working_memory(self, tmp_path):
        from plugins.memory.working_memory import WorkingMemory

        mem = WorkingMemory()
        mem.set("key1", "val1")
        mem.set("key2", "val2")

        path = tmp_path / "working.json"
        MemorySnapshot.save(mem, str(path), compressed=False)
        loaded = MemorySnapshot.load(str(path))
        assert isinstance(loaded, WorkingMemory)

    def test_save_load_semantic_memory(self, tmp_path):
        from plugins.memory.semantic_memory import SemanticMemory

        mem = SemanticMemory()
        mem.add_concept("AI", {"field": "computer science"})
        mem.add_concept("ML", {"field": "machine learning"})
        mem.add_relation("AI", "relates_to", "ML")

        path = tmp_path / "semantic.json"
        MemorySnapshot.save(mem, str(path), compressed=False)
        loaded = MemorySnapshot.load(str(path))
        assert isinstance(loaded, SemanticMemory)

    def test_save_compressed_flag(self, tmp_path):
        from plugins.memory.episodic_memory import EpisodicMemory

        mem = EpisodicMemory()
        mem.store("u", "c")

        path = tmp_path / "compressed.json"
        MemorySnapshot.save(mem, str(path), compressed=True)
        with open(path) as f:
            envelope = json.load(f)
        assert envelope["compressed"] is True
        assert "payload_compressed" in envelope

    def test_save_uncompressed_flag(self, tmp_path):
        from plugins.memory.episodic_memory import EpisodicMemory

        mem = EpisodicMemory()
        mem.store("u", "c")

        path = tmp_path / "uncompressed.json"
        MemorySnapshot.save(mem, str(path), compressed=False)
        with open(path) as f:
            envelope = json.load(f)
        assert envelope["compressed"] is False

    def test_load_integrity_hash_valid(self, tmp_path):
        from plugins.memory.episodic_memory import EpisodicMemory

        mem = EpisodicMemory()
        mem.store("u", "c")
        path = tmp_path / "good.json"
        MemorySnapshot.save(mem, str(path), compressed=False)
        loaded = MemorySnapshot.load(str(path))
        assert isinstance(loaded, EpisodicMemory)

    def test_load_integrity_hash_invalid(self, tmp_path):
        path = tmp_path / "tampered.json"
        mem_dict = {
            "schema_version": MemorySnapshot.CURRENT_VERSION,
            "compressed": False,
            "hash": "0000000000000000000000000000000000000000000000000000000000000000",
            "payload": '{"store_type":"EpisodicMemory","entries":[],"slots":[],"concepts":[],"relations":[]}',
        }
        path.write_text(json.dumps(mem_dict))
        with pytest.raises(SnapshotCorruptedError, match="integrity hash mismatch"):
            MemorySnapshot.load(str(path))

    def test_load_version_mismatch(self, tmp_path):
        path = tmp_path / "old_version.json"
        mem_dict = {
            "schema_version": "99.99",
            "compressed": False,
            "hash": "0" * 64,
            "payload": "{}",
        }
        path.write_text(json.dumps(mem_dict))
        with pytest.raises(SnapshotCorruptedError, match="unsupported schema version"):
            MemorySnapshot.load(str(path))

    def test_load_file_not_found(self, tmp_path):
        with pytest.raises(SnapshotCorruptedError, match="not found"):
            MemorySnapshot.load(str(tmp_path / "ghost.json"))

    def test_load_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json{{{")
        with pytest.raises(SnapshotCorruptedError, match="invalid JSON"):
            MemorySnapshot.load(str(path))

    def test_from_memory_unsupported_store(self):
        class FakeStore:
            pass

        with pytest.raises(SnapshotCorruptedError, match="unsupported store type"):
            MemorySnapshot.from_memory(FakeStore())

    def test_to_memory_unsupported_store(self):
        with pytest.raises(SnapshotCorruptedError, match="unsupported store type"):
            MemorySnapshot.to_memory({"store_type": "FakeStoreX"})

    def test_compressed_roundtrip(self, tmp_path):
        from plugins.memory.episodic_memory import EpisodicMemory

        mem = EpisodicMemory()
        for i in range(5):
            mem.store("u", f"content {i}")
        path = tmp_path / "roundtrip.json"
        MemorySnapshot.save(mem, str(path), compressed=True)
        loaded = MemorySnapshot.load(str(path))
        assert isinstance(loaded, EpisodicMemory)
