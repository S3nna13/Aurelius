import os
import sys
from types import SimpleNamespace

sys.path.insert(1, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

import aurelius_cli.agent_engine as ae


@pytest.mark.parametrize("command", ['python -c "print(123)"', 'python3.14 -c "print(123)"'])
def test_tool_executor_rewrites_python_launchers_to_sys_executable(monkeypatch, command):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(ae.subprocess, "run", fake_run)

    result = ae.ToolExecutor().execute(command)

    assert result.success is True
    assert result.stdout == "ok"
    assert captured["argv"][0] == ae.sys.executable
    assert captured["kwargs"]["shell"] is False
