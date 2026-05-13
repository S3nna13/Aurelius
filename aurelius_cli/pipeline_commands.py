"""aurelius_cli/pipeline_commands.py

CLI for streaming JSONL data transformations using Pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

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


def _compile_expr(expr: str) -> Callable[[object], object]:
    """
    Compile a user-supplied expression into a callable.

    Accepts either:
    - A full ``lambda x: ...`` form — evaluated directly
    - A bare expression like ``x["age"] > 18`` — wrapped in ``lambda x: ...``

    Returns a function taking one argument.
    """
    code = expr.strip()
    if code.startswith("lambda "):
        return eval(code, {}, {})  # noqa: S307 - safe context
    lambda_code = f"lambda x: {code}"
    return eval(lambda_code, {}, {})  # noqa: S307 - safe context


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
