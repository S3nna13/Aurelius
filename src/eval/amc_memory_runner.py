"""Tiny JSON runner for the AMC-Memory benchmark.

This module is deliberately small: it gives CI and local experiments a stable
way to execute the deterministic AMC-Memory scaffold before real checkpoint or
endpoint runners exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Optional

from src.eval.amc_memory_benchmark import AMCMemoryBenchmark

GeneratorName = Literal["oracle", "null", "engine"]
EngineBuilder = Callable[..., tuple[Callable[[Any], str], str, Optional[object]]]


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


def build_engine_generate_fn(
    *,
    backend: str,
    model_path: str,
    model: str | None = None,
    max_tokens: int = 64,
    temperature: float = 0.0,
    system_prompt: str | None = None,
    engine_builder: EngineBuilder | None = None,
) -> Callable[[str], str]:
    """Adapt a serving backend/checkpoint into ``generate_fn(prompt) -> str``.

    The AMC benchmark intentionally talks to plain prompt callables. Aurelius
    serving backends talk to ``ChatRequest`` objects. This adapter is the narrow
    seam between those two surfaces, so real checkpoints can be evaluated
    without changing the deterministic benchmark core.
    """
    if engine_builder is None:
        from src.serving.engine_loader import build_engine as engine_builder
    from src.serving.api_server import ChatRequest

    request_generate_fn, _backend_label, _engine_obj = engine_builder(
        backend=backend,
        model_path=model_path,
    )
    request_model = model or model_path or backend

    def _generate(prompt: str) -> str:
        request = ChatRequest(
            model=request_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            system=system_prompt,
        )
        return request_generate_fn(request)

    return _generate


def run_benchmark(
    *,
    generator: GeneratorName = "null",
    tasks: Sequence[str] | None = None,
    context_tokens: int = 1024,
    samples_per: int = 5,
    backend: str = "mock",
    model_path: str = "",
    model: str | None = None,
    max_tokens: int = 64,
    temperature: float = 0.0,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Run AMC-Memory and return a JSON-serializable payload."""
    bench = AMCMemoryBenchmark()
    if generator == "oracle":
        generate_fn = _oracle_generate_fn(bench, tasks, context_tokens, samples_per)
    elif generator == "null":
        generate_fn = _null_generate_fn
    elif generator == "engine":
        generate_fn = build_engine_generate_fn(
            backend=backend,
            model_path=model_path,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )
    else:
        raise ValueError(f"unknown generator {generator!r}; expected 'oracle', 'null', or 'engine'")

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
        "backend": backend if generator == "engine" else None,
        "model": model or model_path or None,
        "context_tokens": context_tokens,
        "samples_per": samples_per,
        "tasks": list(results),
        "scores": scores,
        "overall_score": bench.overall_score(results),
        "results": results,
    }


def append_jsonl_result(path: Path, payload: dict[str, Any]) -> None:
    """Append one compact JSON benchmark payload to ``path`` as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the deterministic AMC-Memory benchmark")
    parser.add_argument(
        "--generator",
        choices=("null", "oracle", "engine"),
        default="null",
        help="Generator to use. 'engine' adapts an Aurelius serving backend/checkpoint.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task subset. Defaults to all AMC-Memory tasks.",
    )
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--samples-per", type=int, default=5)
    parser.add_argument(
        "--backend",
        default="mock",
        help="Aurelius serving backend for --generator engine",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Checkpoint path or model ID for --generator engine",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model label recorded in engine-mode requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Generation cap for engine mode",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature for engine mode",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt for engine mode",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--jsonl-output",
        type=Path,
        default=None,
        help="Optional JSONL append path, e.g. benchmark-results/amc_memory_runs.jsonl",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_benchmark(
        generator=args.generator,
        tasks=_parse_tasks(args.tasks),
        context_tokens=args.context_tokens,
        samples_per=args.samples_per,
        backend=args.backend,
        model_path=args.model_path,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        system_prompt=args.system_prompt,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if args.jsonl_output is not None:
        append_jsonl_result(args.jsonl_output, payload)
    if args.output is None:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
