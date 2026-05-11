"""Tests for agent.training_pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from agent.react_loop import AgentStep, AgentTrace
from agent.training_pipeline import (
    AgentTrainingCollector,
    trace_to_training_entry,
)


def test_trace_to_training_entry_preserves_instruction_context() -> None:
    trace = AgentTrace(
        task="write a brief summary",
        system_prompt="You are concise.",
        final_answer="done",
        steps_used=2,
        steps=[
            AgentStep(role="assistant", content="thinking"),
            AgentStep(role="assistant", content="still thinking"),
        ],
    )

    entry = trace_to_training_entry(trace, agent_id="agent-1")

    assert entry["agent_id"] == "agent-1"
    assert entry["task"] == "write a brief summary"
    assert entry["system_prompt"] == "You are concise."
    assert entry["instruction"] == "You are concise.\n\nTask: write a brief summary"
    assert entry["final_answer"] == "done"


def test_agent_training_collector_exports_final_answer(tmp_path) -> None:
    collector = AgentTrainingCollector(output_dir=str(tmp_path))
    trace = AgentTrace(
        task="write a brief summary",
        system_prompt="You are concise.",
        final_answer="done",
        steps_used=2,
        steps=[
            AgentStep(role="assistant", content="thinking"),
            AgentStep(role="assistant", content="still thinking"),
        ],
    )
    collector._entries.append(trace_to_training_entry(trace, agent_id="agent-1"))

    output_path = Path(collector.export_sft_format(name="sft"))
    data = json.loads(output_path.read_text())

    assert data == [
        {
            "instruction": "You are concise.\n\nTask: write a brief summary",
            "output": "done",
            "agent_id": "agent-1",
        }
    ]
