from __future__ import annotations

from src.memory import MemoryManager


def test_memory_manager_remember_and_recall() -> None:
    manager = MemoryManager()
    manager.remember(
        "Quantum computing threatens classical cryptography",
        importance=0.9,
        tags=["security", "quantum"],
        context="research",
        working_key="latest_topic",
    )
    result = manager.recall("quantum", top_k=5)
    assert len(result.layered) >= 1
    assert len(result.episodic) >= 1
    assert len(result.fused) >= 1
    assert manager.get_working_memory("latest_topic") is not None


def test_memory_manager_contextualize() -> None:
    manager = MemoryManager()
    manager.remember("Python is used for backend APIs", tags=["python", "api"])
    manager.remember("FastAPI is a Python web framework", tags=["python", "fastapi"])
    context = manager.contextualize("Python", top_k=3)
    assert len(context) >= 1
    assert any("Python" in item or "FastAPI" in item for item in context)


def test_memory_manager_consolidate() -> None:
    manager = MemoryManager()
    manager.remember("Transformers are sequence models", importance=0.8, tags=["ml"])
    count = manager.consolidate()
    assert count >= 0


def test_memory_manager_stats() -> None:
    manager = MemoryManager()
    manager.remember("hello memory")
    stats = manager.stats()
    assert stats["layered_entries"] >= 1
    assert stats["episodic_entries"] >= 1
