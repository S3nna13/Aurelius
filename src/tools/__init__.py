"""Aurelius tools surface: agent-callable tool definitions and registry."""

__all__ = [
    "ToolSpec",
    "ToolResult",
    "ToolRegistry",
    "TOOL_REGISTRY",
    "ShellTool",
    "SHELL_TOOL",
    "FileTool",
    "FILE_TOOL",
    "EditTool",
    "EditOperation",
    "EDIT_TOOL",
    "GrepTool",
    "GrepMatch",
    "GREP_TOOL",
    "WebTool",
    "WEB_TOOL",
]
from .tool_registry import ToolSpec, ToolResult, ToolRegistry, TOOL_REGISTRY
from .shell_tool import ShellTool, SHELL_TOOL
from .file_tool import FileTool, FILE_TOOL
from .edit_tool import EditTool, EditOperation, EDIT_TOOL
from .grep_tool import GrepTool, GrepMatch, GREP_TOOL
from .web_tool import WebTool, WEB_TOOL

# --- Cycle-147 tool deepening --------------------------------------------------
from .code_runner_tool import (
    CodeRunnerTool,
    CodeRunnerConfig,
    CodeRunnerResult,
    CODE_RUNNER_REGISTRY,
)  # noqa: F401
from .linter_tool import (
    LinterTool,
    LintIssue,
    LintResult,
    LINTER_REGISTRY,
)  # noqa: F401
from .json_tool import JSONTool, JSONToolConfig, JSON_TOOL_REGISTRY  # noqa: F401
from .diff_tool import (
    DiffTool,
    DiffFormat,
    DiffResult,
    DIFF_TOOL_REGISTRY,
)  # noqa: F401

# --- Cycle-210 document converter (markitdown-inspired) ----------------------
from .document_converter import (  # noqa: F401
    DocumentConverter,
    ConversionResult,
    DocumentConversionError,
    DEFAULT_DOCUMENT_CONVERTER,
    DOCUMENT_CONVERTER_REGISTRY,
)

__all__ += [
    "DocumentConverter",
    "ConversionResult",
    "DocumentConversionError",
    "DEFAULT_DOCUMENT_CONVERTER",
    "DOCUMENT_CONVERTER_REGISTRY",
]
