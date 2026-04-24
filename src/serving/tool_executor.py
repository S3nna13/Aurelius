import ast
import operator
import re
import json
import datetime
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Any


# Maximum length of a calculator expression accepted by the built-in tool.
# Guards against pathological inputs and protects the parser.
_MAX_EXPR_LEN = 1_000

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _calc_walk(node: ast.AST) -> Any:
    """AST walker that computes an arithmetic expression without going
    through the dynamic code evaluator. Used by :func:`_safe_eval` to avoid
    the B307 finding (AUR-SEC-2026-0022)."""
    if isinstance(node, ast.Expression):
        return _calc_walk(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(
            node.value, bool
        ):
            return node.value
        raise ValueError(
            f"Disallowed constant type: {type(node.value).__name__}"
        )
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](
            _calc_walk(node.left), _calc_walk(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_calc_walk(node.operand))
    raise ValueError(f"Disallowed node type: {type(node).__name__}")


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
    """Compute an arithmetic expression using only numeric literals and safe operators.

    Hardened against B307 (AUR-SEC-2026-0022) by:
      * enforcing a length guard via ``_MAX_EXPR_LEN``,
      * AST-validating every node against an allowlist, and
      * computing the result via a strict AST walker (``_calc_walk``) instead
        of the Python dynamic-code primitive.
    """
    if not expression or len(expression) > _MAX_EXPR_LEN:
        raise ValueError(
            f"expression rejected by size guard: len={len(expression)}"
        )
    # ast.Num was removed in Python 3.14; ast.Constant covers all literals.
    tree = ast.parse(expression.strip(), mode="eval")
    return _calc_walk(tree)


def parse_literal(text: str) -> Any:
    """Safely parse a Python literal (dict/list/tuple/number/string/bool/None).

    Uses :func:`ast.literal_eval`, which rejects anything that is not a
    composite of literals. This is the sanctioned replacement for dynamic
    code evaluation when the input is structured data (AUR-SEC-2026-0022).
    """
    if not isinstance(text, str):
        raise TypeError(f"parse_literal expects str, got {type(text).__name__}")
    if not text or len(text) > _MAX_EXPR_LEN:
        raise ValueError(
            f"literal rejected by size guard: len={len(text)}"
        )
    return ast.literal_eval(text)


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
