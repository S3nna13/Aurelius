from src.tools.code_runner_tool import CODE_RUNNER_REGISTRY, CodeRunnerConfig, CodeRunnerTool


def test_run_simple_math():
    tool = CodeRunnerTool()
    result = tool.run("print(2 + 2)")
    assert result.stdout.strip() == "4"
    assert result.exit_code == 0
    assert not result.timed_out


def test_run_stderr_on_error():
    tool = CodeRunnerTool()
    result = tool.run("raise ValueError('oops')")
    assert result.exit_code != 0
    assert "ValueError" in result.stderr


def test_is_safe_allows_clean_code():
    tool = CodeRunnerTool()
    assert tool.is_safe("x = 1 + 1") is True
    assert tool.is_safe("import math\nprint(math.pi)") is True


def test_is_safe_blocks_import_os():
    tool = CodeRunnerTool()
    assert tool.is_safe("import os\nos.listdir('.')") is False


def test_is_safe_blocks_import_sys():
    tool = CodeRunnerTool()
    assert tool.is_safe("import sys") is False


def test_is_safe_blocks_subprocess():
    tool = CodeRunnerTool()
    assert tool.is_safe("import subprocess") is False


def test_is_safe_blocks_dunder_import():
    tool = CodeRunnerTool()
    assert tool.is_safe("__import__('os')") is False


def test_is_safe_blocks_open():
    tool = CodeRunnerTool()
    assert tool.is_safe("open('/etc/passwd')") is False


def test_is_safe_blocks_eval():
    tool = CodeRunnerTool()
    assert tool.is_safe("eval('1+1')") is False


def test_is_safe_blocks_exec():
    tool = CodeRunnerTool()
    assert tool.is_safe("exec('x=1')") is False


def test_timeout():
    config = CodeRunnerConfig(timeout_s=0.1)
    tool = CodeRunnerTool(config)
    result = tool.run("import time; time.sleep(10)")
    assert result.timed_out is True
    assert result.exit_code == -1
    assert result.stderr == "timeout"


def test_output_truncation():
    config = CodeRunnerConfig(max_output_bytes=10)
    tool = CodeRunnerTool(config)
    result = tool.run("print('A' * 100)")
    assert len(result.stdout) <= 10


def test_registry_key():
    assert "default" in CODE_RUNNER_REGISTRY
    assert CODE_RUNNER_REGISTRY["default"] is CodeRunnerTool
