"""Tests for ShellTool: deny list, execution, output truncation."""
from __future__ import annotations

import pytest

from src.tools.shell_tool import ShellTool, SHELL_TOOL, SHELL_DENY_PATTERNS
from src.tools.tool_registry import ToolResult, ToolSpec, TOOL_REGISTRY


# ---------------------------------------------------------------------------
# SHELL_DENY_PATTERNS
# ---------------------------------------------------------------------------

class TestShellDenyPatterns:
    def test_is_frozenset(self):
        assert isinstance(SHELL_DENY_PATTERNS, frozenset)

    def test_has_at_least_eight_entries(self):
        assert len(SHELL_DENY_PATTERNS) >= 8

    def test_contains_rm_rf(self):
        assert "rm -rf" in SHELL_DENY_PATTERNS

    def test_contains_fork_bomb(self):
        assert ":(){ :|:& };:" in SHELL_DENY_PATTERNS

    def test_contains_dd_if(self):
        assert "dd if=" in SHELL_DENY_PATTERNS

    def test_contains_mkfs(self):
        assert "mkfs" in SHELL_DENY_PATTERNS

    def test_contains_shutdown(self):
        assert "shutdown" in SHELL_DENY_PATTERNS

    def test_contains_reboot(self):
        assert "reboot" in SHELL_DENY_PATTERNS

    def test_contains_chmod_777(self):
        assert "chmod 777 /" in SHELL_DENY_PATTERNS

    def test_contains_eval_dollar(self):
        assert "eval $(" in SHELL_DENY_PATTERNS


# ---------------------------------------------------------------------------
# ShellTool.is_denied
# ---------------------------------------------------------------------------

class TestShellToolIsDenied:
    def test_rm_rf_is_denied(self):
        tool = ShellTool()
        assert tool.is_denied("rm -rf /tmp/test") is True

    def test_echo_is_not_denied(self):
        tool = ShellTool()
        assert tool.is_denied("echo hello") is False

    def test_ls_is_not_denied(self):
        tool = ShellTool()
        assert tool.is_denied("ls -la /tmp") is False

    def test_shutdown_is_denied(self):
        tool = ShellTool()
        assert tool.is_denied("sudo shutdown now") is True

    def test_reboot_is_denied(self):
        tool = ShellTool()
        assert tool.is_denied("reboot") is True

    def test_mkfs_is_denied(self):
        tool = ShellTool()
        assert tool.is_denied("mkfs.ext4 /dev/sdb") is True

    def test_custom_deny_patterns(self):
        tool = ShellTool(deny_patterns=frozenset(["forbidden_cmd"]))
        assert tool.is_denied("run forbidden_cmd now") is True
        assert tool.is_denied("echo safe") is False

    def test_empty_deny_patterns(self):
        tool = ShellTool(deny_patterns=frozenset())
        assert tool.is_denied("rm -rf /") is False

    def test_case_sensitive(self):
        tool = ShellTool()
        # deny patterns are lowercase; uppercase ECHO should not match
        assert tool.is_denied("ECHO hello") is False


# ---------------------------------------------------------------------------
# ShellTool.run
# ---------------------------------------------------------------------------

class TestShellToolRun:
    def test_denied_command_returns_failure(self):
        tool = ShellTool()
        result = tool.run("rm -rf /tmp/fake_test_path")
        assert result.success is False

    def test_denied_command_error_mentions_pattern(self):
        tool = ShellTool()
        result = tool.run("rm -rf /tmp/fake_test_path")
        assert "rm -rf" in result.error

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
        result = tool.run("exit 1")
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

    def test_empty_command_runs_without_crash(self):
        tool = ShellTool()
        result = tool.run("")
        assert isinstance(result, ToolResult)

    def test_multiline_output(self):
        tool = ShellTool()
        result = tool.run("printf 'line1\\nline2\\nline3'")
        assert "line1" in result.output
        assert "line3" in result.output


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
