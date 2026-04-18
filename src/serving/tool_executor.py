import ast
import re
import json
import datetime
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any


@dataclass
class ToolResult:
    tool_name: str
    output: str
    success: bool
    error: Optional[str] = None


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable


def _safe_eval(expression: str) -> Any:
    """Evaluate an arithmetic expression using only literal numbers and safe operators."""
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
        ast.Pow, ast.USub, ast.UAdd,
    )
    # ast.Num was removed in Python 3.14; ast.Constant covers all literals
    tree = ast.parse(expression.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Disallowed node type: {type(node).__name__}")
    code = compile(tree, "<string>", "eval")
    # ast-validated expression — no builtins exposed
    return eval(code, {"__builtins__": {}})  # noqa: S307


def calculator(expression: str) -> str:
    try:
        result = _safe_eval(expression)
        return str(result)
    except Exception as exc:
        raise ValueError(f"Invalid expression: {exc}") from exc


def word_count(text: str) -> str:
    count = len(text.split())
    return f"{count} words"


def current_time() -> str:
    return datetime.datetime.now().isoformat()


def echo(message: str) -> str:
    return message


_BUILTIN_TOOLS = [
    Tool(name="calculator", description="Evaluate a safe arithmetic expression.", fn=calculator),
    Tool(name="word_count", description="Count words in text.", fn=word_count),
    Tool(name="current_time", description="Return the current ISO datetime.", fn=current_time),
    Tool(name="echo", description="Return the message unchanged.", fn=echo),
]

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class ToolExecutor:
    def __init__(self, tools: Optional[List[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in (_BUILTIN_TOOLS if tools is None else tools):
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def parse_tool_call(self, text: str) -> Optional[Dict]:
        match = _TOOL_CALL_RE.search(text)
        if not match:
            return None
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def execute(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        if tool_name not in self._tools:
            return ToolResult(
                tool_name=tool_name,
                output="",
                success=False,
                error=f"Unknown tool: {tool_name!r}",
            )
        tool = self._tools[tool_name]
        try:
            output = tool.fn(**args)
            return ToolResult(tool_name=tool_name, output=str(output), success=True)
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                output="",
                success=False,
                error=str(exc),
            )

    def process(self, model_output: str) -> Tuple[Optional[ToolResult], str]:
        parsed = self.parse_tool_call(model_output)
        if parsed is None:
            return None, model_output
        tool_name = parsed.get("name", "")
        args = parsed.get("args", {})
        result = self.execute(tool_name, args)
        cleaned = _TOOL_CALL_RE.sub("", model_output).strip()
        return result, cleaned
