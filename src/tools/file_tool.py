"""File read/write tool with path validation."""
from __future__ import annotations

import os

from .tool_registry import ToolResult, ToolSpec, TOOL_REGISTRY

FILE_DENY_PATHS: frozenset[str] = frozenset([
    "/etc/passwd",
    "/etc/shadow",
    os.path.expanduser("~/.ssh/"),
    "/proc/",
    "/sys/",
])


class FileTool:
    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = os.path.abspath(base_dir) if base_dir is not None else None

    def is_denied(self, path: str) -> bool:
        """Check if any deny path is a prefix of the absolute path."""
        abs_path = os.path.abspath(path)
        return any(abs_path.startswith(deny) for deny in FILE_DENY_PATHS)

    def _check_base_dir(self, path: str) -> str | None:
        """Return error string if path is outside base_dir, else None."""
        if self.base_dir is not None:
            abs_path = os.path.abspath(path)
            if not abs_path.startswith(self.base_dir):
                return f"path {path!r} is outside base_dir {self.base_dir!r}"
        return None

    def read(self, path: str, max_bytes: int = 65536) -> ToolResult:
        """Read up to max_bytes from path."""
        if self.is_denied(path):
            return ToolResult(tool_name="file", success=False, output="", error=f"path denied: {path!r}")
        base_err = self._check_base_dir(path)
        if base_err:
            return ToolResult(tool_name="file", success=False, output="", error=base_err)
        try:
            with open(path, "r", errors="replace") as fh:
                content = fh.read(max_bytes)
            return ToolResult(tool_name="file", success=True, output=content)
        except Exception as e:
            return ToolResult(tool_name="file", success=False, output="", error=str(e))

    def write(self, path: str, content: str) -> ToolResult:
        """Write content to path."""
        if self.is_denied(path):
            return ToolResult(tool_name="file", success=False, output="", error=f"path denied: {path!r}")
        base_err = self._check_base_dir(path)
        if base_err:
            return ToolResult(tool_name="file", success=False, output="", error=base_err)
        try:
            with open(path, "w") as fh:
                fh.write(content)
            return ToolResult(
                tool_name="file",
                success=True,
                output=f"wrote {len(content)} bytes to {path}",
            )
        except Exception as e:
            return ToolResult(tool_name="file", success=False, output="", error=str(e))

    def list_dir(self, path: str) -> ToolResult:
        """List directory entries sorted."""
        if self.is_denied(path):
            return ToolResult(tool_name="file", success=False, output="", error=f"path denied: {path!r}")
        base_err = self._check_base_dir(path)
        if base_err:
            return ToolResult(tool_name="file", success=False, output="", error=base_err)
        try:
            entries = sorted(os.listdir(path))
            return ToolResult(tool_name="file", success=True, output="\n".join(entries))
        except Exception as e:
            return ToolResult(tool_name="file", success=False, output="", error=str(e))

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="file",
            description="Read or write files",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["read", "write", "list"]},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
            required=["operation", "path"],
        )


FILE_TOOL = FileTool()
TOOL_REGISTRY.register(
    FILE_TOOL.spec(),
    lambda **kw: FILE_TOOL.read(kw["path"]) if kw.get("operation") == "read" else FILE_TOOL.write(kw["path"], kw.get("content", "")),
)
