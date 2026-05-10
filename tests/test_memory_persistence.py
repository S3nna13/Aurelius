from __future__ import annotations

import os
import tempfile

from src.memory import MemoryManager


def test_memory_export_import_round_trip() -> None:
    manager = MemoryManager()
    manager.remember(
        "Quantum computing threatens classical cryptography",
        importance=0.9,
        tags=["security", "quantum"],
        context="research",
        working_key="latest_topic",
    )
    state = manager.export_state()
    assert "layered" in state
    assert "episodic" in state
    assert "long_term" in state
    assert "zettel" in state
    assert "working" in state

    fresh = MemoryManager()
    fresh.import_state(state)
    assert fresh.get_working_memory("latest_topic") is not None
    result = fresh.recall("quantum", top_k=5)
    assert len(result.fused) >= 1


def test_memory_save_load_disk() -> None:
    manager = MemoryManager()
    manager.remember(
        "Python is a dynamic programming language",
        tags=["python", "language"],
    )
    manager.remember(
        "FastAPI is a modern Python web framework",
        tags=["python", "fastapi", "web"],
        working_key="last_tool",
    )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        path = tmp.name

    try:
        manager.save(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        fresh = MemoryManager(snapshot_path=path)
        result = fresh.recall("python", top_k=5)
        assert len(result.fused) >= 1
        assert fresh.get_working_memory("last_tool") is not None
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_memory_snapshot_integrity() -> None:
    manager = MemoryManager()
    manager.remember("Entry A", tags=["a"])
    manager.remember("Entry B", tags=["b"])

    state = manager.export_state()
    assert len(state["layered"]) > 0

    manager2 = MemoryManager()
    manager2.import_state(state)
    assert manager2.stats()["layered_entries"] >= 2


def test_auto_load_nonexistent_path() -> None:
    manager = MemoryManager(snapshot_path="/tmp/nonexistent_ark_test_file.bin")
    stats = manager.stats()
    assert stats["layered_entries"] == 0


def test_memory_preserves_all_stores() -> None:
    manager = MemoryManager()
    manager.remember("Test content", tags=["test"], working_key="wk")

    state = manager.export_state()
    fresh = MemoryManager()
    fresh.import_state(state)

    orig = manager.stats()
    restored = fresh.stats()
    assert restored["layered_entries"] == orig["layered_entries"]
    assert restored["episodic_entries"] == orig["episodic_entries"]
    assert restored["long_term_entries"] == orig["long_term_entries"]
    assert restored["zettel_entries"] == orig["zettel_entries"]
    assert restored["working_slots"] == orig["working_slots"]
