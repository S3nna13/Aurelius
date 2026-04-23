"""Grep tool for code search and navigation.

Inspired by repo-map patterns from Aider (Aider-AI/aider, Apache-2.0)
and SWE-agent ACIβ search commands (SWE-agent/SWE-agent, MIT). License: MIT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from .tool_registry import ToolResult, ToolSpec, TOOL_REGISTRY

_MAX_CONTENT_LEN = 500_000
_MAX_RESULTS = 1000
_MAX_PATTERN_LEN = 1024

@dataclass
class GrepMatch:
    line_number: int
    line: str
    context_before: list[str]
    context_after: list[str]

class GrepTool:
    def search(self, pattern: str, content: str,
               context_lines: int = 2,
               max_results: int = 50,
               flags: int = 0) -> ToolResult:
        """
        Search for pattern in content (line-by-line regex).
        Returns JSON-serializable output with match list.
        Raises no exceptions on malformed content — returns error ToolResult.
        """
        if len(pattern) > _MAX_PATTERN_LEN:
            return ToolResult(tool_name="grep", success=False, output="",
                              error=f"pattern exceeds {_MAX_PATTERN_LEN} chars")
        if len(content) > _MAX_CONTENT_LEN:
            return ToolResult(tool_name="grep", success=False, output="",
                              error=f"content exceeds {_MAX_CONTENT_LEN} chars")
        if max_results > _MAX_RESULTS:
            max_results = _MAX_RESULTS
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult(tool_name="grep", success=False, output="",
                              error=f"invalid pattern: {e}")
        lines = content.splitlines()
        matches: list[dict] = []
        for i, line in enumerate(lines):
            if compiled.search(line):
                before = lines[max(0, i - context_lines): i]
                after = lines[i + 1: i + 1 + context_lines]
                matches.append({
                    "line_number": i + 1,
                    "line": line,
                    "context_before": before,
                    "context_after": after,
                })
                if len(matches) >= max_results:
                    break
        import json
        return ToolResult(tool_name="grep", success=True,
                          output=json.dumps({"matches": matches, "total": len(matches)}),
                          error="")

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="grep",
            description="Search for a regex pattern in text content",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "content": {"type": "string"},
                    "context_lines": {"type": "integer"},
                },
            },
            required=["pattern", "content"],
        )


GREP_TOOL = GrepTool()
TOOL_REGISTRY.register(GREP_TOOL.spec(), handler=GREP_TOOL.search)
