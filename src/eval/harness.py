"""Automated evaluation harness for Aurelius checkpoints.

Wraps EleutherAI's ``lm-evaluation-harness`` to run a standard benchmark suite
and persist results as JSON for trend analysis across training checkpoints.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.eval.benchmark_config import (
    ALL_BENCHMARKS,
    BENCHMARK_BY_NAME,
    BenchmarkResult,
    BenchmarkSpec,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("./results")


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------

class EvalHarness:
    """Run lm-evaluation-harness benchmarks against Aurelius checkpoints.

    Parameters
    ----------
    results_dir:
        Base directory for JSON result files.
    benchmarks:
        Which benchmarks to run.  Defaults to :data:`ALL_BENCHMARKS`.
    device:
        Device string passed to the harness (e.g. ``"cuda:0"``).
    batch_size:
        Per-device batch size for evaluation.
    """

    def __init__(
        self,
        *,
        results_dir: str | Path = RESULTS_DIR,
        benchmarks: list[BenchmarkSpec] | None = None,
        device: str = "cuda:0",
        batch_size: int | str = "auto",
    ) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.benchmarks = benchmarks or list(ALL_BENCHMARKS)
        self.device = device
        self.batch_size = str(batch_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_checkpoint(
        self,
        checkpoint_path: str | Path,
        *,
        checkpoint_step: int | None = None,
        benchmarks: list[str] | None = None,
    ) -> list[BenchmarkResult]:
        """Run the full benchmark suite against *checkpoint_path*.

        Parameters
        ----------
        checkpoint_path:
            Path to the model checkpoint directory (HuggingFace-format).
        checkpoint_step:
            Optional training step for tagging results.
        benchmarks:
            Run only these benchmark names (e.g. ``["MMLU", "GSM8K"]``).
            ``None`` means run all configured benchmarks.

        Returns
        -------
        list[BenchmarkResult]
            One result per benchmark, also persisted to disk.
        """
        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        specs = self._resolve_benchmarks(benchmarks)
        results: list[BenchmarkResult] = []

        for spec in specs:
            logger.info("Running benchmark %s ...", spec.name)
            t0 = time.monotonic()
            raw = self._run_single_benchmark(checkpoint_path, spec)
            elapsed = time.monotonic() - t0

            score = self._extract_score(raw, spec)
            result = BenchmarkResult(
                benchmark=spec,
                score=score,
                checkpoint_path=str(checkpoint_path),
                checkpoint_step=checkpoint_step,
                raw_results=raw,
            )
            results.append(result)
            logger.info(
                "  %s  (%.1fs elapsed)", result.summary_line(), elapsed,
            )

        # Persist to disk.
        self._save_results(results, checkpoint_path, checkpoint_step)
        self._update_trend_log(results, checkpoint_path, checkpoint_step)

        return results

    def print_summary(self, results: list[BenchmarkResult]) -> None:
        """Print a formatted table of results to stdout."""
        print("\n" + "=" * 72)
        print("  AURELIUS EVALUATION SUMMARY")
        print("=" * 72)
        for r in results:
            print("  " + r.summary_line())
        print("=" * 72 + "\n")

        below = [r for r in results if r.status == "BELOW_EXPECTED"]
        if below:
            logger.warning(
                "%d benchmark(s) below expected range: %s",
                len(below),
                ", ".join(r.benchmark.name for r in below),
            )

    def load_trend(self, benchmark_name: str | None = None) -> list[dict[str, Any]]:
        """Load the trend log, optionally filtered to one benchmark.

        Returns a list of dicts with keys: ``timestamp``, ``checkpoint_step``,
        ``benchmark``, ``score``, ``status``.
        """
        trend_path = self.results_dir / "trend_log.jsonl"
        if not trend_path.exists():
            return []

        entries: list[dict[str, Any]] = []
        for line in trend_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if benchmark_name is None or entry.get("benchmark") == benchmark_name:
                entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_benchmarks(
        self,
        names: list[str] | None,
    ) -> list[BenchmarkSpec]:
        if names is None:
            return self.benchmarks
        specs: list[BenchmarkSpec] = []
        for name in names:
            if name not in BENCHMARK_BY_NAME:
                raise ValueError(
                    f"Unknown benchmark {name!r}. "
                    f"Available: {list(BENCHMARK_BY_NAME)}"
                )
            specs.append(BENCHMARK_BY_NAME[name])
        return specs

    def _run_single_benchmark(
        self,
        checkpoint_path: Path,
        spec: BenchmarkSpec,
    ) -> dict[str, Any]:
        """Invoke ``lm_eval`` as a subprocess and return parsed JSON results.

        Falls back to the Python API (``lm_eval.simple_evaluate``) if the CLI
        is unavailable.
        """
        try:
            return self._run_via_python_api(checkpoint_path, spec)
        except ImportError:
            logger.info(
                "lm_eval Python package not importable; falling back to CLI"
            )
            return self._run_via_cli(checkpoint_path, spec)

    def _run_via_python_api(
        self,
        checkpoint_path: Path,
        spec: BenchmarkSpec,
    ) -> dict[str, Any]:
        """Use the lm-evaluation-harness Python API directly."""
        import lm_eval  # type: ignore[import-untyped]

        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={checkpoint_path},dtype=bfloat16",
            tasks=[spec.task],
            num_fewshot=spec.num_fewshot,
            batch_size=self.batch_size,
            device=self.device,
        )
        return results  # type: ignore[return-value]

    def _run_via_cli(
        self,
        checkpoint_path: Path,
        spec: BenchmarkSpec,
    ) -> dict[str, Any]:
        """Invoke lm_eval via the command line."""
        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={checkpoint_path},dtype=bfloat16",
            "--tasks", spec.task,
            "--num_fewshot", str(spec.num_fewshot),
            "--batch_size", self.batch_size,
            "--device", self.device,
            "--output_path", str(self.results_dir / "raw"),
            "--log_samples",
        ]
        logger.info("CLI command: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse the JSON from stdout (lm_eval prints JSON at the end).
        for line in reversed(proc.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)  # type: ignore[no-any-return]
        raise RuntimeError(
            f"Could not parse lm_eval JSON output for {spec.name}.\n"
            f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}"
        )

    @staticmethod
    def _extract_score(raw: dict[str, Any], spec: BenchmarkSpec) -> float:
        """Pull the primary metric from the raw harness output.

        The harness nests results under ``results.<task>.<metric>``.
        """
        task_results = raw.get("results", {}).get(spec.task, {})
        if spec.metric in task_results:
            return float(task_results[spec.metric])

        # Some tasks use comma-separated sub-tasks — try top-level average.
        # E.g., MMLU reports per-subject and an aggregate.
        for key, val in task_results.items():
            if spec.metric in str(key):
                return float(val)

        # Last resort: scan all tasks for the metric.
        for task_name, task_vals in raw.get("results", {}).items():
            if spec.metric in task_vals:
                return float(task_vals[spec.metric])

        raise KeyError(
            f"Metric {spec.metric!r} not found in results for task {spec.task!r}. "
            f"Available keys: {list(task_results)}"
        )

    def _save_results(
        self,
        results: list[BenchmarkResult],
        checkpoint_path: Path,
        checkpoint_step: int | None,
    ) -> Path:
        """Write a per-checkpoint JSON results file."""
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        step_tag = f"_step{checkpoint_step}" if checkpoint_step is not None else ""
        filename = f"eval_{checkpoint_path.name}{step_tag}_{ts}.json"
        out_path = self.results_dir / filename

        payload: dict[str, Any] = {
            "timestamp": ts,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "results": [],
        }
        for r in results:
            payload["results"].append({
                "benchmark": r.benchmark.name,
                "task": r.benchmark.task,
                "metric": r.benchmark.metric,
                "score": r.score,
                "expected_low": r.benchmark.expected_low,
                "expected_high": r.benchmark.expected_high,
                "status": r.status,
                "num_fewshot": r.benchmark.num_fewshot,
            })

        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Results saved to %s", out_path)
        return out_path

    def _update_trend_log(
        self,
        results: list[BenchmarkResult],
        checkpoint_path: Path,
        checkpoint_step: int | None,
    ) -> None:
        """Append one JSONL entry per benchmark to the trend log."""
        trend_path = self.results_dir / "trend_log.jsonl"
        ts = datetime.now(tz=timezone.utc).isoformat()

        lines: list[str] = []
        for r in results:
            entry = {
                "timestamp": ts,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_step": checkpoint_step,
                "benchmark": r.benchmark.name,
                "score": r.score,
                "status": r.status,
            }
            lines.append(json.dumps(entry, ensure_ascii=False))

        with trend_path.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        logger.info("Trend log updated at %s (%d entries)", trend_path, len(lines))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Evaluate an Aurelius checkpoint from the command line."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Evaluate an Aurelius checkpoint on standard benchmarks",
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the model checkpoint directory",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Training step for this checkpoint (for trend tracking)",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        default=None,
        help="Benchmark names to run (default: all)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory for result JSON files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="Batch size (integer or 'auto')",
    )
    args = parser.parse_args()

    harness = EvalHarness(
        results_dir=args.results_dir,
        device=args.device,
        batch_size=args.batch_size,
    )

    results = harness.evaluate_checkpoint(
        args.checkpoint,
        checkpoint_step=args.step,
        benchmarks=args.benchmarks,
    )

    harness.print_summary(results)


if __name__ == "__main__":
    main()
