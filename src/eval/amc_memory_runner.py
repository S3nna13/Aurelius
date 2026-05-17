"""Tiny JSON runner for the AMC-Memory benchmark.

This module is deliberately small: it gives CI and local experiments a stable
way to execute the deterministic AMC-Memory scaffold before real checkpoint or
endpoint runners exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from src.eval.amc_memory_benchmark import AMCMemoryBenchmark

GeneratorName = Literal["oracle", "null"]


def _parse_tasks(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "":
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _oracle_generate_fn(
    bench: AMCMemoryBenchmark,
    tasks: Sequence[str] | None,
    context_tokens: int,
    samples_per: int,
):
    selected_tasks = list(bench.TASKS if tasks is None else tasks)
    lookup: dict[str, str] = {}
    for task in selected_tasks:
        for seed in range(samples_per):
            example = bench._build_task(task, context_tokens=context_tokens, seed=seed)
            lookup[example.prompt] = example.expected
    return lambda prompt: lookup[prompt]


def _null_generate_fn(_prompt: str) -> str:
    return ""


def run_benchmark(
    *,
    generator: GeneratorName = "null",
    tasks: Sequence[str] | None = None,
    context_tokens: int = 1024,
    samples_per: int = 5,
) -> dict[str, Any]:
    """Run AMC-Memory and return a JSON-serializable payload."""
    bench = AMCMemoryBenchmark()
    if generator == "oracle":
        generate_fn = _oracle_generate_fn(bench, tasks, context_tokens, samples_per)
    elif generator == "null":
        generate_fn = _null_generate_fn
    else:
        raise ValueError(f"unknown generator {generator!r}; expected 'oracle' or 'null'")

    results = bench.evaluate(
        generate_fn,
        tasks=tasks,
        context_tokens=context_tokens,
        samples_per=samples_per,
    )
    scores = bench.score_per_task(results)
    return {
        "suite": "amc_memory",
        "generator": generator,
        "context_tokens": context_tokens,
        "samples_per": samples_per,
        "tasks": list(results),
        "scores": scores,
        "overall_score": bench.overall_score(results),
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deterministic AMC-Memory benchmark")
    parser.add_argument(
        "--generator",
        choices=("null", "oracle"),
        default="null",
        help="Built-in generator to use. 'oracle' is a smoke-test upper bound.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task subset. Defaults to all AMC-Memory tasks.",
    )
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--samples-per", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_benchmark(
        generator=args.generator,
        tasks=_parse_tasks(args.tasks),
        context_tokens=args.context_tokens,
        samples_per=args.samples_per,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
