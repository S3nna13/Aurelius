from __future__ import annotations

import json
import tempfile
from pathlib import Path

from training_data.agent_generator import AgentDataGenerator


def test_default_config() -> None:
    g = AgentDataGenerator({})
    assert len(g.TOOL_TYPES) == 6
    assert "code_execution" in g.TOOL_TYPES
    assert "web_search" in g.TOOL_TYPES
    assert "file_ops" in g.TOOL_TYPES
    assert "shell_commands" in g.TOOL_TYPES
    assert "api_calls" in g.TOOL_TYPES
    assert "database" in g.TOOL_TYPES


def test_generate_trajectory_code() -> None:
    g = AgentDataGenerator({})
    traj = g.generate_trajectory("code_execution")
    assert "conversations" in traj
    assert "tools_used" in traj
    assert "steps" in traj
    assert "success" in traj
    assert len(traj["conversations"]) >= 4
    assert traj["steps"] >= 2


def test_generate_trajectory_search() -> None:
    g = AgentDataGenerator({})
    traj = g.generate_trajectory("web_search")
    assert len(traj["conversations"]) > 0
    assert any("web_search" in tool or "web_search" in str(traj["tools_used"]) for tool in traj["tools_used"])


def test_generate_trajectory_file_ops() -> None:
    g = AgentDataGenerator({})
    traj = g.generate_trajectory("file_ops")
    assert len(traj["conversations"]) > 0
    assert any("file" in tool.lower() or "read" in tool.lower() or "list" in tool.lower() or "glob" in tool.lower() or "grep" in tool.lower() for tool in traj["tools_used"])


def test_generate_trajectory_shell() -> None:
    g = AgentDataGenerator({})
    traj = g.generate_trajectory("shell_commands")
    assert len(traj["conversations"]) > 0
    assert any("shell" in tool for tool in traj["tools_used"])


def test_generate_trajectory_api() -> None:
    g = AgentDataGenerator({})
    traj = g.generate_trajectory("api_calls")
    assert len(traj["conversations"]) > 0
    assert any("api" in tool for tool in traj["tools_used"])


def test_generate_trajectory_db() -> None:
    g = AgentDataGenerator({})
    traj = g.generate_trajectory("database")
    assert len(traj["conversations"]) > 0
    assert any("sql" in tool.lower() for tool in traj["tools_used"])


def test_generate_trajectory_unknown_type() -> None:
    g = AgentDataGenerator({})
    try:
        g.generate_trajectory("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_conversation_format() -> None:
    g = AgentDataGenerator({})
    traj = g.generate_trajectory("code_execution")
    for msg in traj["conversations"]:
        assert "from" in msg
        assert "value" in msg
        assert msg["from"] in ("human", "gpt", "tool")
        assert isinstance(msg["value"], str)


def test_tool_call_format() -> None:
    g = AgentDataGenerator({})
    traj = g.generate_trajectory("database")
    for msg in traj["conversations"]:
        if msg["from"] == "gpt" and "<tool_call>" in msg["value"]:
                assert '"name"' in msg["value"]
                assert '"args"' in msg["value"]


def test_at_least_15_templates() -> None:
    g = AgentDataGenerator({})
    total = 0
    for tt in g.TOOL_TYPES:
        templates = g._get_templates(tt)
        total += len(templates)
    assert total >= 15, f"Only {total} templates"


def test_run_creates_jsonl_files() -> None:
    g = AgentDataGenerator({})
    with tempfile.TemporaryDirectory() as tmpdir:
        g.run(100, tmpdir)
        agent_dir = Path(tmpdir) / "agent"
        assert (agent_dir / "train.jsonl").exists()
        assert (agent_dir / "val.jsonl").exists()


def test_run_train_val_split() -> None:
    g = AgentDataGenerator({})
    with tempfile.TemporaryDirectory() as tmpdir:
        g.run(100, tmpdir)
        agent_dir = Path(tmpdir) / "agent"
        train_lines = (agent_dir / "train.jsonl").read_text().strip().split("\n")
        val_lines = (agent_dir / "val.jsonl").read_text().strip().split("\n")
        assert len(train_lines) == 90
        assert len(val_lines) == 10


def test_tool_observation_format() -> None:
    g = AgentDataGenerator({})
    traj = g.generate_trajectory("api_calls")
    for msg in traj["conversations"]:
        if msg["from"] == "tool":
            parsed = json.loads(msg["value"])
            assert "result" in parsed


def test_output_is_json_serializable() -> None:
    g = AgentDataGenerator({})
    traj = g.generate_trajectory("code_execution")
    json.dumps(traj)
