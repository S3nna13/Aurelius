"""Tests for ShellTool: allow-list, execution, output truncation."""
from __future__ import annotations

import pytest

from src.tools.shell_tool import ShellTool, SHELL_TOOL, SHELL_ALLOWLIST
from src.tools.tool_registry import ToolResult, ToolSpec, TOOL_REGISTRY


# ---------------------------------------------------------------------------
# SHELL_ALLOWLIST
# ---------------------------------------------------------------------------

class TestShellAllowlist:
    def test_is_frozenset(self):
        assert isinstance(SHELL_ALLOWLIST, frozenset)

    def test_has_at_least_eight_entries(self):
        assert len(SHELL_ALLOWLIST) >= 8

    def test_contains_safe_commands(self):
        assert "echo" in SHELL_ALLOWLIST
        assert "ls" in SHELL_ALLOWLIST
        assert "cat" in SHELL_ALLOWLIST
        assert "git" in SHELL_ALLOWLIST

    def test_does_not_contain_dangerous_commands(self):
        assert "rm" not in SHELL_ALLOWLIST
        assert "mkfs" not in SHELL_ALLOWLIST
        assert "shutdown" not in SHELL_ALLOWLIST
        assert "reboot" not in SHELL_ALLOWLIST


# ---------------------------------------------------------------------------
# ShellTool._validate
# ---------------------------------------------------------------------------

class TestShellToolValidate:
    def test_empty_command_rejected(self):
        tool = ShellTool()
        argv, error = tool._validate("")
        assert argv is None
        assert "non-empty" in error

    def test_metachar_pipe_rejected(self):
        tool = ShellTool()
        argv, error = tool._validate("echo hello | cat")
        assert argv is None
        assert "metacharacter" in error

    def test_metachar_semicolon_rejected(self):
        tool = ShellTool()
        argv, error = tool._validate("echo a; echo b")
        assert argv is None
        assert "metacharacter" in error

    def test_metachar_dollar_rejected(self):
        tool = ShellTool()
        argv, error = tool._validate("echo $HOME")
        assert argv is None
        assert "metacharacter" in error

    def test_allowlisted_command_accepted(self):
        tool = ShellTool()
        argv, error = tool._validate("echo hello")
        assert argv == ["echo", "hello"]
        assert error == ""

    def test_denylisted_command_rejected(self):
        tool = ShellTool()
        argv, error = tool._validate("rm -rf /tmp")
        assert argv is None
        assert "not in the allowlist" in error

    def test_absolute_path_resolved(self):
        tool = ShellTool()
        argv, error = tool._validate("/bin/echo hello")
        assert argv == ["/bin/echo", "hello"]
        assert error == ""

    def test_custom_allowlist(self):
        tool = ShellTool(allowlist=frozenset(["allowed_cmd"]))
        argv, error = tool._validate("allowed_cmd arg")
        assert argv == ["allowed_cmd", "arg"]
        assert error == ""

        argv, error = tool._validate("echo hello")
        assert argv is None
        assert "not in the allowlist" in error


# ---------------------------------------------------------------------------
# ShellTool.run
# ---------------------------------------------------------------------------

class TestShellToolRun:
    def test_denied_command_returns_failure(self):
        tool = ShellTool()
        result = tool.run("rm -rf /tmp/fake_test_path")
        assert result.success is False

    def test_denied_command_error_mentions_allowlist(self):
        tool = ShellTool()
        result = tool.run("rm -rf /tmp/fake_test_path")
        assert "not in the allowlist" in result.error

    def test_denied_result_is_toolresult(self):
        tool = ShellTool()
        result = tool.run("rm -rf /")
        assert isinstance(result, ToolResult)

    def test_safe_echo_command_succeeds(self):
        tool = ShellTool()
        result = tool.run("echo hello")
        assert result.success is True

    def test_safe_echo_output_contains_hello(self):
        tool = ShellTool()
        result = tool.run("echo hello")
        assert "hello" in result.output

    def test_run_returns_toolresult(self):
        tool = ShellTool()
        result = tool.run("echo test")
        assert isinstance(result, ToolResult)

    def test_failing_command_returns_failure(self):
        tool = ShellTool()
        result = tool.run("python3 -c 'import sys; sys.exit(1)'")
        assert result.success is False

    def test_output_truncated_to_2000_chars(self):
        tool = ShellTool()
        result = tool.run("python3 -c \"print('x' * 5000)\"")
        assert len(result.output) <= 2000

    def test_stderr_truncated_to_500_chars(self):
        tool = ShellTool()
        result = tool.run("python3 -c \"import sys; sys.stderr.write('e' * 1000)\"")
        assert len(result.error) <= 500

    def test_timeout_returns_failure(self):
        tool = ShellTool(timeout_seconds=1)
        result = tool.run("sleep 5")
        assert result.success is False

    def test_tool_name_in_result(self):
        tool = ShellTool()
        result = tool.run("echo hi")
        assert result.tool_name == "shell"

    def test_empty_command_rejected(self):
        tool = ShellTool()
        result = tool.run("")
        assert result.success is False
        assert "non-empty" in result.error

    def test_multiline_output(self):
        tool = ShellTool()
        result = tool.run("printf 'line1\\nline2\\nline3'")
        assert "line1" in result.output
        assert "line3" in result.output

    def test_pipeline_rejected(self):
        tool = ShellTool()
        result = tool.run("echo hello | wc -l")
        assert result.success is False
        assert "metacharacter" in result.error


# ---------------------------------------------------------------------------
# ShellTool.spec
# ---------------------------------------------------------------------------

class TestShellToolSpec:
    def test_spec_returns_toolspec(self):
        tool = ShellTool()
        assert isinstance(tool.spec(), ToolSpec)

    def test_spec_name_is_shell(self):
        tool = ShellTool()
        assert tool.spec().name == "shell"

    def test_spec_has_description(self):
        tool = ShellTool()
        assert tool.spec().description != ""

    def test_spec_parameters_has_command_property(self):
        tool = ShellTool()
        props = tool.spec().parameters.get("properties", {})
        assert "command" in props

    def test_spec_required_contains_command(self):
        tool = ShellTool()
        assert "command" in tool.spec().required


# ---------------------------------------------------------------------------
# SHELL_TOOL module-level instance + TOOL_REGISTRY integration
# ---------------------------------------------------------------------------

class TestShellToolInstance:
    def test_shell_tool_exists(self):
        assert SHELL_TOOL is not None

    def test_shell_tool_is_shell_tool_instance(self):
        assert isinstance(SHELL_TOOL, ShellTool)

    def test_shell_tool_registered_in_tool_registry(self):
        assert "shell" in TOOL_REGISTRY.list_tools()

    def test_shell_tool_invocable_via_registry(self):
        result = TOOL_REGISTRY.invoke("shell", command="echo registry_test")
        assert result.success is True
        assert "registry_test" in result.output

    def test_shell_tool_spec_in_registry(self):
        spec = TOOL_REGISTRY.get_spec("shell")
        assert spec is not None
        assert spec.name == "shell"

    def test_shell_tool_in_openai_format(self):
        openai_tools = TOOL_REGISTRY.to_openai_format()
        names = [t["function"]["name"] for t in openai_tools]
        assert "shell" in names
