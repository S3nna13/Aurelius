"""Tests for src.agent.agent_persistence."""

from __future__ import annotations

from src.agent.agent_memory import AgentMemory
from src.agent.agent_persistence import AgentPersistence


def test_save_and_load_memory_round_trip_preserves_entries(tmp_path) -> None:
    persistence = AgentPersistence(base_path=str(tmp_path))
    memory = AgentMemory()

    episodic_text = "episodic event " + ("x" * 220)
    semantic_text = "semantic fact " + ("y" * 220)

    memory.remember(episodic_text, memory_type="episodic", tags=["event"], importance=0.7)
    memory.remember(semantic_text, memory_type="semantic", tags=["fact"], importance=0.9)
    memory.learn_procedure("cleanup", "step 1 -> step 2")

    persistence.save_memory(memory, name="memory")

    restored = AgentMemory()
    assert persistence.load_memory(restored, name="memory") is True

    assert len(restored.episodic) == 1
    assert restored.episodic[0].content == episodic_text
    assert restored.episodic[0].tags == ["event"]
    assert restored.episodic[0].importance == 0.7

    assert len(restored.semantic) == 1
    restored_semantic = next(iter(restored.semantic.values()))
    assert restored_semantic.content == semantic_text
    assert restored_semantic.tags == ["fact"]
    assert restored_semantic.importance == 0.9

    assert restored.procedural == {"cleanup": "step 1 -> step 2"}


def test_load_memory_missing_file_returns_false(tmp_path) -> None:
    persistence = AgentPersistence(base_path=str(tmp_path))
    restored = AgentMemory()

    assert persistence.load_memory(restored, name="missing") is False
