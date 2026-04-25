"""Tests for memory_snapshot."""

from __future__ import annotations

import gzip
import hashlib
import json
import os

import pytest

from src.memory.memory_snapshot import MemorySnapshot, SnapshotCorruptedError
from src.memory.episodic_memory import EpisodicMemory
from src.memory.working_memory import WorkingMemory
from src.memory.semantic_memory import SemanticMemory


class TestEpisodicRoundtrip:
    def test_save_and_load(self, tmp_path):
        mem = EpisodicMemory()
        mem.store("user", "hello", 1.0)
        mem.store("assistant", "hi there", 0.9)
        path = str(tmp_path / "snap.json")
        MemorySnapshot.save(mem, path, compressed=False)
        loaded = MemorySnapshot.load(path)
        assert isinstance(loaded, EpisodicMemory)
        assert len(loaded._entries) == 2

    def test_compressed_roundtrip(self, tmp_path):
        mem = EpisodicMemory()
        mem.store("user", "hello")
        path = str(tmp_path / "snap.gz.json")
        MemorySnapshot.save(mem, path, compressed=True)
        loaded = MemorySnapshot.load(path)
        assert isinstance(loaded, EpisodicMemory)
        assert len(loaded._entries) == 1


class TestWorkingMemoryRoundtrip:
    def test_working_memory(self, tmp_path):
        mem = WorkingMemory()
        mem.set("user", "task")
        path = str(tmp_path / "wm.json")
        MemorySnapshot.save(mem, path)
        loaded = MemorySnapshot.load(path)
        assert isinstance(loaded, WorkingMemory)


class TestSemanticMemoryRoundtrip:
    def test_semantic_memory(self, tmp_path):
        mem = SemanticMemory()
        mem.add_concept("cat", {"legs": 4})
        mem.add_concept("animal", {})
        mem.add_relation("cat", "is_a", "animal")
        path = str(tmp_path / "sm.json")
        MemorySnapshot.save(mem, path)
        loaded = MemorySnapshot.load(path)
        assert isinstance(loaded, SemanticMemory)


class TestIntegrityChecks:
    def test_hash_mismatch(self, tmp_path):
        mem = EpisodicMemory()
        mem.store("user", "hello")
        path = str(tmp_path / "bad.json")
        MemorySnapshot.save(mem, path, compressed=False)
        data = json.loads(open(path).read())
        data["hash"] = "0000" * 16
        open(path, "w").write(json.dumps(data))
        with pytest.raises(SnapshotCorruptedError):
            MemorySnapshot.load(path)

    def test_unknown_version(self, tmp_path):
        mem = EpisodicMemory()
        path = str(tmp_path / "ver.json")
        MemorySnapshot.save(mem, path)
        data = json.loads(open(path).read())
        data["schema_version"] = "99.0"
        open(path, "w").write(json.dumps(data))
        with pytest.raises(SnapshotCorruptedError):
            MemorySnapshot.load(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(SnapshotCorruptedError):
            MemorySnapshot.load(str(tmp_path / "nofile.json"))

    def test_invalid_json(self, tmp_path):
        path = str(tmp_path / "bad.json")
        open(path, "w").write("not json")
        with pytest.raises(SnapshotCorruptedError):
            MemorySnapshot.load(path)


class TestEmptyMemory:
    def test_empty_episodic(self, tmp_path):
        mem = EpisodicMemory()
        path = str(tmp_path / "empty.json")
        MemorySnapshot.save(mem, path)
        loaded = MemorySnapshot.load(path)
        assert isinstance(loaded, EpisodicMemory)
        assert len(loaded._entries) == 0


class TestPathTraversal:
    def test_traversal_in_save(self, tmp_path):
        mem = EpisodicMemory()
        with pytest.raises(ValueError):
            MemorySnapshot.save(mem, "../outside.json")

    def test_traversal_in_load(self, tmp_path):
        with pytest.raises(ValueError):
            MemorySnapshot.load("../outside.json")
