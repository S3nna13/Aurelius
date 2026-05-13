"""Tests for aurelius_cli.pipeline_commands — JSONL streaming transformations."""

import argparse
import io
import json
import sys

from aurelius_cli.pipeline_commands import (
    _compile_expr,
    build_pipeline_parser,
    handle_pipeline,
)

# ---------------------------------------------------------------------------
# Expression compilation
# ---------------------------------------------------------------------------

def test_compile_lambda_expression():
    fn = _compile_expr("lambda x: x > 5")
    assert fn(10) is True
    assert fn(3) is False


def test_compile_expression_wrapped():
    fn = _compile_expr("x['age'] >= 18")
    assert fn({"age": 20}) is True
    assert fn({"age": 16}) is False


def test_compile_expression_with_method_call():
    fn = _compile_expr("x.strip().lower()")
    assert fn("  Hello ") == "hello"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def test_pipeline_parser_accepts_all_flags():
    build_pipeline_parser(argparse.ArgumentParser().add_subparsers())  # register subparser
    # We can't easily parse top-level; but we can build a parser and parse args via separate parser.
    # Simpler: build a top-level parser like in other tests.
    top = argparse.ArgumentParser()
    sub = top.add_subparsers(dest="cmd")
    build_pipeline_parser(sub)
    args = top.parse_args(["pipeline", "--filter", "x > 1", "--map", "x * 2", "--head", "5"])
    assert args.cmd == "pipeline"
    assert args.filter_expr == "x > 1"
    assert args.map_expr == "x * 2"
    assert args.head == 5


# ---------------------------------------------------------------------------
# Integration-like tests (no subprocess; call handler directly)
# ---------------------------------------------------------------------------

def _run_handler(stdin_text: str, extra_args: list[str] | None = None) -> tuple[int, str, str]:
    """Helper: invoke handle_pipeline with mocked stdin/stdout."""
    extra_args = extra_args or []
    top = argparse.ArgumentParser()
    sub = top.add_subparsers(dest="cmd")
    build_pipeline_parser(sub)
    args = top.parse_args(["pipeline"] + extra_args)

    # Redirect stdin/stdout
    old_stdin, old_stdout = sys.stdin, sys.stdout
    sys.stdin = io.StringIO(stdin_text)
    out_buf = io.StringIO()
    sys.stdout = out_buf
    try:
        rc = handle_pipeline(args)
    finally:
        sys.stdin, sys.stdout = old_stdin, old_stdout
    return rc, out_buf.getvalue(), ""


def test_pipeline_filter_only():
    stdin = '\n'.join([json.dumps(i) for i in range(10)])
    rc, out, _ = _run_handler(stdin, ["--filter", "x % 2 == 0"])
    assert rc == 0
    results = [json.loads(line) for line in out.strip().split('\n') if line]
    assert results == [0, 2, 4, 6, 8]


def test_pipeline_map_only():
    stdin = '\n'.join(json.dumps(i) for i in [1, 2, 3])
    rc, out, _ = _run_handler(stdin, ["--map", "x * 10"])
    assert rc == 0
    results = [json.loads(line) for line in out.strip().split('\n')]
    assert results == [10, 20, 30]


def test_pipeline_head():
    stdin = '\n'.join(json.dumps(i) for i in range(100))
    rc, out, _ = _run_handler(stdin, ["--head", "3"])
    assert rc == 0
    results = [json.loads(line) for line in out.strip().split('\n')]
    assert results == [0, 1, 2]


def test_pipeline_tail():
    stdin = '\n'.join(json.dumps(i) for i in range(5))
    rc, out, _ = _run_handler(stdin, ["--tail", "2"])
    assert rc == 0
    results = [json.loads(line) for line in out.strip().split('\n')]
    assert results == [3, 4]


def test_pipeline_sort():
    stdin = '\n'.join(json.dumps(i) for i in [5, 1, 3, 2, 4])
    rc, out, _ = _run_handler(stdin, ["--sort", "x"])
    assert rc == 0
    results = [json.loads(line) for line in out.strip().split('\n')]
    assert results == [1, 2, 3, 4, 5]


def test_pipeline_sort_reverse():
    stdin = '\n'.join(json.dumps(i) for i in [1, 2, 3])
    rc, out, _ = _run_handler(stdin, ["--sort", "x", "--reverse"])
    assert rc == 0
    results = [json.loads(line) for line in out.strip().split('\n')]
    assert results == [3, 2, 1]


def test_pipeline_dedup():
    stdin = '\n'.join(json.dumps(i) for i in [1, 2, 2, 3, 1, 4])
    rc, out, _ = _run_handler(stdin, ["--dedup"])
    assert rc == 0
    results = [json.loads(line) for line in out.strip().split('\n')]
    # Dedup by JSON string equality; order preserved (first occurrence kept)
    assert results == [1, 2, 3, 4]


def test_pipeline_dedup_key():
    stdin = '\n'.join(
        json.dumps({"id": i, "v": i * 10}) for i in [1, 2, 1, 3, 2]
    )
    rc, out, _ = _run_handler(stdin, ["--dedup-key", "x['id']"])
    assert rc == 0
    results = [json.loads(line) for line in out.strip().split('\n')]
    assert results == [{"id": 1, "v": 10}, {"id": 2, "v": 20}, {"id": 3, "v": 30}]


def test_pipeline_chained_filter_map_head():
    stdin = '\n'.join(json.dumps(i) for i in range(20))
    rc, out, _ = _run_handler(stdin, ["--filter", "x % 3 == 0", "--map", "x // 2", "--head", "3"])
    assert rc == 0
    results = [json.loads(line) for line in out.strip().split('\n')]
    assert results == [0, 1, 3]  # filter keep multiples of 3, map //2, head 3

