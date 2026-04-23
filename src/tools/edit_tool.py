"""Diff-based code editing tool for the Aurelius agentic-coding surface.

Inspired by Aider edit format (Aider-AI/aider, Apache-2.0) and Kimi-Dev
patch-synthesis (MoonshotAI, MIT); Aurelius-native implementation. License: MIT.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from .tool_registry import ToolResult, ToolSpec, TOOL_REGISTRY

_MAX_CONTENT_LEN = 500_000   # 500KB per file content
_MAX_DIFF_LEN = 200_000      # 200KB per diff

@dataclass
class EditOperation:
    """A single search-and-replace edit operation."""
    search: str
    replace: str

class EditTool:
    """Apply search-and-replace edits to file content strings."""

    def apply_edit(self, content: str, search: str, replace: str) -> ToolResult:
        """Apply a single search-and-replace.
        Fails if search string not found exactly once (no ambiguous edits).
        """
        if len(content) > _MAX_CONTENT_LEN:
            return ToolResult(tool_name="edit", success=False, output="",
                              error=f"content exceeds {_MAX_CONTENT_LEN} chars")
        if not search:
            return ToolResult(tool_name="edit", success=False, output="",
                              error="search string must not be empty")
        count = content.count(search)
        if count == 0:
            return ToolResult(tool_name="edit", success=False, output="",
                              error="search string not found in content")
        if count > 1:
            return ToolResult(tool_name="edit", success=False, output="",
                              error=f"search string found {count} times (ambiguous edit)")
        new_content = content.replace(search, replace, 1)
        return ToolResult(tool_name="edit", success=True, output=new_content, error="")

    def apply_edits(self, content: str, operations: list[EditOperation]) -> ToolResult:
        """Apply a sequence of edits. Stops at first failure."""
        current = content
        for op in operations:
            result = self.apply_edit(current, op.search, op.replace)
            if not result.success:
                return result
            current = result.output
        return ToolResult(tool_name="edit", success=True, output=current, error="")

    def unified_diff(self, original: str, modified: str,
                     fromfile: str = "original", tofile: str = "modified") -> str:
        """Generate a unified diff string."""
        orig_lines = original.splitlines(keepends=True)
        mod_lines = modified.splitlines(keepends=True)
        return "".join(difflib.unified_diff(orig_lines, mod_lines,
                                             fromfile=fromfile, tofile=tofile))

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="edit",
            description="Apply search-and-replace edits to file content",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "search": {"type": "string"},
                    "replace": {"type": "string"},
                },
            },
            required=["content", "search", "replace"],
        )


EDIT_TOOL = EditTool()
TOOL_REGISTRY.register(EDIT_TOOL.spec(), handler=EDIT_TOOL.apply_edit)
