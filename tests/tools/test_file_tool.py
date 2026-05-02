"""Tests for FileTool: deny list, read/write/list_dir, spec."""

from __future__ import annotations

import os
import tempfile

from src.tools.file_tool import FILE_DENY_PATHS, FILE_TOOL, FileTool
from src.tools.tool_registry import TOOL_REGISTRY, ToolResult, ToolSpec

# ---------------------------------------------------------------------------
# FILE_DENY_PATHS
# ---------------------------------------------------------------------------


class TestFileDenyPaths:
    def test_is_frozenset(self):
        assert isinstance(FILE_DENY_PATHS, frozenset)

    def test_contains_etc_passwd(self):
        assert os.path.realpath("/etc/passwd") in FILE_DENY_PATHS

    def test_contains_etc_shadow(self):
        assert os.path.realpath("/etc/shadow") in FILE_DENY_PATHS

    def test_contains_proc(self):
        assert os.path.realpath("/proc/") in FILE_DENY_PATHS

    def test_contains_sys(self):
        assert os.path.realpath("/sys/") in FILE_DENY_PATHS

    def test_has_at_least_four_entries(self):
        assert len(FILE_DENY_PATHS) >= 4


# ---------------------------------------------------------------------------
# FileTool.is_denied
# ---------------------------------------------------------------------------


class TestFileToolIsDenied:
    def test_etc_passwd_is_denied(self):
        tool = FileTool()
        assert tool.is_denied("/etc/passwd") is True

    def test_tmp_test_txt_is_not_denied(self):
        tool = FileTool()
        assert tool.is_denied("/tmp/test.txt") is False

    def test_etc_shadow_is_denied(self):
        tool = FileTool()
        assert tool.is_denied("/etc/shadow") is True

    def test_proc_cpuinfo_is_denied(self):
        tool = FileTool()
        assert tool.is_denied("/proc/cpuinfo") is True

    def test_sys_kernel_is_denied(self):
        tool = FileTool()
        assert tool.is_denied("/sys/kernel") is True

    def test_regular_user_path_not_denied(self):
        tool = FileTool()
        assert tool.is_denied("/home/user/notes.txt") is False


# ---------------------------------------------------------------------------
# FileTool.read
# ---------------------------------------------------------------------------


class TestFileToolRead:
    def test_read_non_existent_returns_failure(self):
        tool = FileTool()
        result = tool.read("/tmp/aurelius_no_such_file_xyz123.txt")
        assert result.success is False

    def test_read_non_existent_has_error(self):
        tool = FileTool()
        result = tool.read("/tmp/aurelius_no_such_file_xyz123.txt")
        assert result.error != ""

    def test_read_denied_path_returns_failure(self):
        tool = FileTool()
        result = tool.read("/etc/passwd")
        assert result.success is False

    def test_read_denied_error_mentions_denied(self):
        tool = FileTool()
        result = tool.read("/etc/passwd")
        assert "denied" in result.error.lower()

    def test_read_returns_toolresult(self):
        tool = FileTool()
        result = tool.read("/tmp/aurelius_no_such_file_xyz123.txt")
        assert isinstance(result, ToolResult)

    def test_read_valid_file(self):
        tool = FileTool()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
            fh.write("hello world")
            path = fh.name
        try:
            result = tool.read(path)
            assert result.success is True
            assert "hello world" in result.output
        finally:
            os.unlink(path)

    def test_read_respects_max_bytes(self):
        tool = FileTool()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
            fh.write("A" * 200)
            path = fh.name
        try:
            result = tool.read(path, max_bytes=10)
            assert len(result.output) <= 10
        finally:
            os.unlink(path)

    def test_read_outside_base_dir_denied(self):
        with tempfile.TemporaryDirectory() as base:
            tool = FileTool(base_dir=base)
            result = tool.read("/tmp/outside.txt")
            assert result.success is False


# ---------------------------------------------------------------------------
# FileTool.write + round-trip
# ---------------------------------------------------------------------------


class TestFileToolWrite:
    def test_write_creates_file(self):
        tool = FileTool()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.txt")
            result = tool.write(path, "test content")
            assert result.success is True
            assert os.path.exists(path)

    def test_write_returns_byte_count_in_output(self):
        tool = FileTool()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.txt")
            content = "hello"
            result = tool.write(path, content)
            assert str(len(content)) in result.output

    def test_write_returns_toolresult(self):
        tool = FileTool()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.txt")
            result = tool.write(path, "data")
            assert isinstance(result, ToolResult)

    def test_write_denied_path_fails(self):
        tool = FileTool()
        result = tool.write("/etc/passwd", "bad")
        assert result.success is False

    def test_write_denied_error_mentions_denied(self):
        tool = FileTool()
        result = tool.write("/etc/shadow", "bad")
        assert "denied" in result.error.lower()

    def test_write_read_round_trip(self):
        tool = FileTool()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "round_trip.txt")
            original = "round trip content 12345"
            tool.write(path, original)
            result = tool.read(path)
            assert result.success is True
            assert result.output == original

    def test_write_output_mentions_path(self):
        tool = FileTool()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.txt")
            result = tool.write(path, "x")
            assert path in result.output


# ---------------------------------------------------------------------------
# FileTool.list_dir
# ---------------------------------------------------------------------------


class TestFileToolListDir:
    def test_list_dir_valid_dir_succeeds(self):
        tool = FileTool()
        with tempfile.TemporaryDirectory() as d:
            result = tool.list_dir(d)
            assert result.success is True

    def test_list_dir_returns_toolresult(self):
        tool = FileTool()
        with tempfile.TemporaryDirectory() as d:
            result = tool.list_dir(d)
            assert isinstance(result, ToolResult)

    def test_list_dir_nonexistent_returns_failure(self):
        tool = FileTool()
        result = tool.list_dir("/tmp/aurelius_no_such_dir_xyz123")
        assert result.success is False

    def test_list_dir_nonexistent_has_error(self):
        tool = FileTool()
        result = tool.list_dir("/tmp/aurelius_no_such_dir_xyz123")
        assert result.error != ""

    def test_list_dir_contains_files(self):
        tool = FileTool()
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "alpha.txt"), "w").close()
            open(os.path.join(d, "beta.txt"), "w").close()
            result = tool.list_dir(d)
            assert "alpha.txt" in result.output
            assert "beta.txt" in result.output

    def test_list_dir_sorted(self):
        tool = FileTool()
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "z.txt"), "w").close()
            open(os.path.join(d, "a.txt"), "w").close()
            result = tool.list_dir(d)
            lines = result.output.splitlines()
            assert lines == sorted(lines)

    def test_list_dir_denied_path_fails(self):
        tool = FileTool()
        result = tool.list_dir("/proc")
        assert result.success is False


# ---------------------------------------------------------------------------
# FileTool.spec
# ---------------------------------------------------------------------------


class TestFileToolSpec:
    def test_spec_returns_toolspec(self):
        tool = FileTool()
        assert isinstance(tool.spec(), ToolSpec)

    def test_spec_name_is_file(self):
        tool = FileTool()
        assert tool.spec().name == "file"

    def test_spec_has_description(self):
        tool = FileTool()
        assert tool.spec().description != ""

    def test_spec_parameters_has_operation(self):
        tool = FileTool()
        props = tool.spec().parameters.get("properties", {})
        assert "operation" in props

    def test_spec_parameters_has_path(self):
        tool = FileTool()
        props = tool.spec().parameters.get("properties", {})
        assert "path" in props

    def test_spec_required_contains_operation(self):
        tool = FileTool()
        assert "operation" in tool.spec().required

    def test_spec_required_contains_path(self):
        tool = FileTool()
        assert "path" in tool.spec().required

    def test_spec_operation_enum(self):
        tool = FileTool()
        op_schema = tool.spec().parameters["properties"]["operation"]
        assert "enum" in op_schema
        assert "read" in op_schema["enum"]
        assert "write" in op_schema["enum"]


# ---------------------------------------------------------------------------
# FILE_TOOL module-level instance + TOOL_REGISTRY integration
# ---------------------------------------------------------------------------


class TestFileToolInstance:
    def test_file_tool_exists(self):
        assert FILE_TOOL is not None

    def test_file_tool_is_file_tool_instance(self):
        assert isinstance(FILE_TOOL, FileTool)

    def test_file_tool_registered_in_tool_registry(self):
        assert "file" in TOOL_REGISTRY.list_tools()

    def test_file_tool_invocable_via_registry_read(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as fh:
            fh.write("registry read test")
            path = fh.name
        try:
            result = TOOL_REGISTRY.invoke("file", operation="read", path=path)
            assert result.success is True
            assert "registry read test" in result.output
        finally:
            os.unlink(path)

    def test_file_tool_invocable_via_registry_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reg_write.txt")
            result = TOOL_REGISTRY.invoke("file", operation="write", path=path, content="written")
            assert result.success is True

    def test_file_tool_spec_in_registry(self):
        spec = TOOL_REGISTRY.get_spec("file")
        assert spec is not None
        assert spec.name == "file"

    def test_file_tool_in_openai_format(self):
        openai_tools = TOOL_REGISTRY.to_openai_format()
        names = [t["function"]["name"] for t in openai_tools]
        assert "file" in names
