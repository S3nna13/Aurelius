"""Tests for agent.agent_persistence."""

from __future__ import annotations

from src.agent.agent_memory import AgentMemory
from src.agent.agent_persistence import AgentPersistence
from src.agent.agent_runtime import AgentMessage, AgentRuntime, AgentSpec, AgentStatus
from src.agent.react_loop import AgentStep, AgentTrace


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


def test_save_and_load_trace_round_trip_preserves_prompt_metadata(tmp_path) -> None:
    persistence = AgentPersistence(base_path=str(tmp_path))
    trace = AgentTrace(
        task="summarize the notes",
        system_prompt="You are concise.",
        status="success",
        steps_used=1,
        final_answer="summary",
        steps=[AgentStep(role="assistant", content="thinking")],
    )

    persistence.save_trace(trace, agent_id="agent-1")
    restored = persistence.load_latest_trace(agent_id="agent-1")

    assert restored is not None
    assert restored.task == "summarize the notes"
    assert restored.system_prompt == "You are concise."
    assert restored.status == "success"
    assert restored.steps_used == 1
    assert restored.final_answer == "summary"


def test_save_and_load_agents_round_trip_restores_status_and_mailbox(tmp_path) -> None:
    persistence = AgentPersistence(base_path=str(tmp_path))
    runtime = AgentRuntime()
    runtime.register_agent(
        AgentSpec(
            id="agent-1",
            name="Alpha",
            role="planner",
            tools=["search"],
            max_iterations=3,
            status=AgentStatus.RUNNING,
        )
    )
    runtime.send_message(
        AgentMessage(
            sender="user",
            recipient="agent-1",
            content="hello " + ("world " * 50),
            metadata={"priority": "high"},
        )
    )

    persistence.save_agents(runtime, name="agents")

    restored = AgentRuntime()
    assert persistence.load_agents(restored, name="agents") is True

    agent = restored.get_agent("agent-1")
    assert agent is not None
    assert agent.status == AgentStatus.RUNNING
    assert restored.mailbox[0].recipient == "agent-1"
    assert restored.mailbox[0].content == "hello " + ("world " * 50)
    assert restored.mailbox[0].metadata == {"priority": "high"}
