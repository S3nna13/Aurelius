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
# --- Cycle-147 tool deepening --------------------------------------------------
from .code_runner_tool import (
    CODE_RUNNER_REGISTRY as CODE_RUNNER_REGISTRY,
)  # noqa: F401
from .code_runner_tool import (
    CodeRunnerConfig as CodeRunnerConfig,
)
from .code_runner_tool import (
    CodeRunnerResult as CodeRunnerResult,
)
from .code_runner_tool import (
    CodeRunnerTool as CodeRunnerTool,
)
from .diff_tool import (
    DIFF_TOOL_REGISTRY as DIFF_TOOL_REGISTRY,
)  # noqa: F401
from .diff_tool import (
    DiffFormat as DiffFormat,
)
from .diff_tool import (
    DiffResult as DiffResult,
)
from .diff_tool import (
    DiffTool as DiffTool,
)

# --- Cycle-210 document converter (markitdown-inspired) ----------------------
from .document_converter import (  # noqa: F401
    DEFAULT_DOCUMENT_CONVERTER,
    DOCUMENT_CONVERTER_REGISTRY,
    ConversionResult,
    DocumentConversionError,
    DocumentConverter,
)
from .edit_tool import EDIT_TOOL, EditOperation, EditTool
from .file_tool import FILE_TOOL, FileTool
from .grep_tool import GREP_TOOL, GrepMatch, GrepTool
from .json_tool import JSON_TOOL_REGISTRY, JSONTool, JSONToolConfig  # noqa: F401
from .linter_tool import (
    LINTER_REGISTRY as LINTER_REGISTRY,
)  # noqa: F401
from .linter_tool import (
    LinterTool as LinterTool,
)
from .linter_tool import (
    LintIssue as LintIssue,
)
from .linter_tool import (
    LintResult as LintResult,
)
from .shell_tool import SHELL_TOOL, ShellTool
from .tool_registry import TOOL_REGISTRY, ToolRegistry, ToolResult, ToolSpec
from .web_tool import WEB_TOOL, WebTool

__all__ += [
    "DocumentConverter",
    "ConversionResult",
    "DocumentConversionError",
    "DEFAULT_DOCUMENT_CONVERTER",
    "DOCUMENT_CONVERTER_REGISTRY",
]
