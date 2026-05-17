"""aurelius_cli/pipeline_commands.py

CLI for streaming JSONL data transformations using Pipeline.
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import sys
from collections.abc import Callable
from typing import Any

from aurelius_cli.pipeline_processor import Pipeline


def build_pipeline_parser(subparsers: argparse._SubParsersAction) -> None:
    """Add ``aurelius pipeline`` command to the top-level CLI."""
    parser = subparsers.add_parser(
        "pipeline",
        help="Stream JSONL transformations (filter/map/sort/head/tail/dedup)",
    )
    parser.add_argument(
        "--filter",
        dest="filter_expr",
        help='Keep items where expression is true; expression uses "x" (e.g. x["age"] > 18)',
    )
    parser.add_argument(
        "--map",
        dest="map_expr",
        help='Transform each item; expression uses "x" (e.g. x["name"].upper())',
    )
    parser.add_argument(
        "--sort",
        dest="sort_key_expr",
        help='Sort items by key expression using "x" (e.g. x["score"])',
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        default=False,
        help="Sort in descending order (requires --sort)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=None,
        help="Emit only the first N items",
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=None,
        help="Emit only the last N items",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        default=False,
        help="Remove duplicate items (keeps first occurrence, preserves order)",
    )
    parser.add_argument(
        "--dedup-key",
        dest="dedup_key_expr",
        help='Deduplicate by key expression using "x" (e.g. x["id"])',
    )
    parser.set_defaults(func=handle_pipeline)


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}
_UNARY_OPS = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}
_SAFE_FUNCTIONS = {
    "abs": abs,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "str": str,
}
_SAFE_METHODS = {
    "count",
    "endswith",
    "find",
    "get",
    "index",
    "isalnum",
    "isalpha",
    "isdigit",
    "islower",
    "isspace",
    "istitle",
    "isupper",
    "join",
    "lower",
    "replace",
    "split",
    "startswith",
    "strip",
    "title",
    "upper",
}


class _SafeExpression:
    """Small expression evaluator for JSONL pipeline transforms.

    This intentionally supports data-shaping expressions over the single
    variable ``x`` and rejects imports, comprehensions, attribute traversal,
    assignment, and arbitrary function calls. It replaces the previous CLI
    ``eval`` path while preserving common examples such as ``x["age"] > 18``
    and ``x["name"].upper()``.
    """

    def __init__(self, source: str) -> None:
        self.source = source
        tree = ast.parse(source, mode="eval")
        body = tree.body
        if isinstance(body, ast.Lambda):
            args = body.args.args
            if len(args) != 1 or args[0].arg != "x":
                raise ValueError("lambda expressions must accept exactly one argument named 'x'")
            body = body.body
        self.body = body

    def __call__(self, x: object) -> object:
        return self._eval(self.body, {"x": x})

    def _eval(self, node: ast.AST, env: dict[str, object]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == "x":
                return env["x"]
            raise ValueError(f"unknown name {node.id!r}; only 'x' is allowed")
        if isinstance(node, ast.List):
            return [self._eval(elt, env) for elt in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(self._eval(elt, env) for elt in node.elts)
        if isinstance(node, ast.Dict):
            return {
                self._eval(key, env): self._eval(value, env)
                for key, value in zip(node.keys, node.values, strict=True)
            }
        if isinstance(node, ast.Subscript):
            return self._eval(node.value, env)[self._eval_slice(node.slice, env)]
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](self._eval(node.left, env), self._eval(node.right, env))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](self._eval(node.operand, env))
        if isinstance(node, ast.BoolOp):
            values = [self._eval(value, env) for value in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, env)
            for op, comparator in zip(node.ops, node.comparators, strict=True):
                right = self._eval(comparator, env)
                op_fn = _CMP_OPS.get(type(op))
                if op_fn is None:
                    raise ValueError(f"unsupported comparison operator {type(op).__name__}")
                if not op_fn(left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self._eval(node.body if self._eval(node.test, env) else node.orelse, env)
        if isinstance(node, ast.Call):
            return self._eval_call(node, env)
        raise ValueError(f"unsupported expression node {type(node).__name__}")

    def _eval_slice(self, node: ast.AST, env: dict[str, object]) -> Any:
        if isinstance(node, ast.Slice):
            lower = self._eval(node.lower, env) if node.lower is not None else None
            upper = self._eval(node.upper, env) if node.upper is not None else None
            step = self._eval(node.step, env) if node.step is not None else None
            return slice(lower, upper, step)
        return self._eval(node, env)

    def _eval_call(self, node: ast.Call, env: dict[str, object]) -> object:
        if node.keywords:
            raise ValueError("keyword arguments are not supported in pipeline expressions")
        args = [self._eval(arg, env) for arg in node.args]
        if isinstance(node.func, ast.Name):
            fn = _SAFE_FUNCTIONS.get(node.func.id)
            if fn is None:
                raise ValueError(f"function {node.func.id!r} is not allowed")
            return fn(*args)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr.startswith("_") or node.func.attr not in _SAFE_METHODS:
                raise ValueError(f"method {node.func.attr!r} is not allowed")
            target = self._eval(node.func.value, env)
            return getattr(target, node.func.attr)(*args)
        raise ValueError("unsupported callable in pipeline expression")


def _compile_expr(expr: str) -> Callable[[object], object]:
    """Compile a user-supplied data expression into a safe callable."""
    return _SafeExpression(expr.strip())


def handle_pipeline(args: argparse.Namespace) -> int:
    """Read JSON lines from stdin, apply transformations, write JSON lines to stdout."""
    filter_fn = _compile_expr(args.filter_expr) if args.filter_expr else None
    map_fn = _compile_expr(args.map_expr) if args.map_expr else None
    sort_key_fn = _compile_expr(args.sort_key_expr) if args.sort_key_expr else None
    dedup_key_fn = _compile_expr(args.dedup_key_expr) if args.dedup_key_expr else None

    try:
        items = [json.loads(line) for line in sys.stdin if line.strip()]
    except json.JSONDecodeError as exc:
        print(f"error: invalid JSON on stdin — {exc}", file=sys.stderr)
        return 1

    p = Pipeline(items)

    try:
        if filter_fn:
            p = p.filter(filter_fn)
        if map_fn:
            p = p.map(map_fn)
        if sort_key_fn:
            p = p.sort(key=sort_key_fn, reverse=args.reverse)
        if args.head is not None:
            if args.head < 0:
                print("error: --head must be non-negative", file=sys.stderr)
                return 1
            p = p.head(args.head)
        if args.tail is not None:
            if args.tail < 0:
                print("error: --tail must be non-negative", file=sys.stderr)
                return 1
            p = p.tail(args.tail)
    except Exception as exc:
        print(f"error: pipeline stage failed — {exc}", file=sys.stderr)
        return 1

    results = list(p)

    if args.dedup or args.dedup_key_expr:
        if args.dedup_key_expr:
            seen = set()
            uniq = []
            for item in results:
                k = dedup_key_fn(item)
                if k not in seen:
                    seen.add(k)
                    uniq.append(item)
            results = uniq
        else:
            seen = set()
            uniq = []
            for item in results:
                key = json.dumps(item, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    uniq.append(item)
            results = uniq

    for item in results:
        print(json.dumps(item, ensure_ascii=False))

    return 0
