"""OSWorld scorer: Xie et al. 2024 (2406.14800, Apache-2.0), clean-room reimplementation.

OSWorld-style computer-use task scorer for the Aurelius eval surface.
Measures task completion rate across apps and difficulty levels, mirroring
the evaluation methodology of the OSWorld benchmark (Xie et al. 2024).

Pure standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "OSWorldTask",
    "OSWorldResult",
    "OSWorldMetrics",
    "OSWorldScorer",
    "OSWORLD_TASK_REGISTRY",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OSWorldTask:
    """Specification of a single OSWorld computer-use task."""

    task_id: str
    app: str
    instruction: str
    reference_answers: list[str]
    subtasks: list[str] = field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "medium"


@dataclass
class OSWorldResult:
    """Outcome produced by an agent attempting an :class:`OSWorldTask`."""

    task_id: str
    app: str
    completed: bool
    steps_taken: int
    final_screen_state: str | None = None
    error: str | None = None


@dataclass
class OSWorldMetrics:
    """Aggregated metrics across a batch of scored OSWorld results."""

    n_tasks: int
    n_completed: int
    completion_rate: float
    by_app: dict[str, float]
    by_difficulty: dict[str, float]
    avg_steps: float


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class OSWorldScorer:
    """Score OSWorld computer-use task results.

    Follows the completion criterion from Xie et al. 2024:
    a task is considered complete when the agent signals success AND
    the final screen state (if provided) contains at least one reference
    answer string.
    """

    def score_result(self, result: OSWorldResult, task: OSWorldTask) -> bool:
        """Return True iff *result* counts as a completed task.

        Rules:
        * result.completed must be True.
        * If result.final_screen_state is None → accept (no screenshot to check).
        * Otherwise at least one reference_answer must appear as a substring
          of final_screen_state.
        """
        if not result.completed:
            return False
        if result.final_screen_state is None:
            return True
        return any(
            ref in result.final_screen_state for ref in task.reference_answers
        )

    def score_batch(
        self,
        results: list[OSWorldResult],
        tasks: dict[str, OSWorldTask],
    ) -> OSWorldMetrics:
        """Score a list of results against the provided task dict.

        Results whose task_id is not found in *tasks* are skipped.
        """
        n_tasks = len(results)
        n_completed = 0
        total_steps = 0

        # Accumulators: {key: [n_completed, n_total]}
        app_counts: dict[str, list[int]] = {}
        diff_counts: dict[str, list[int]] = {}

        for res in results:
            task = tasks.get(res.task_id)
            if task is None:
                continue

            passed = self.score_result(res, task)
            if passed:
                n_completed += 1
            total_steps += res.steps_taken

            app = res.app
            if app not in app_counts:
                app_counts[app] = [0, 0]
            app_counts[app][1] += 1
            if passed:
                app_counts[app][0] += 1

            diff = task.difficulty
            if diff not in diff_counts:
                diff_counts[diff] = [0, 0]
            diff_counts[diff][1] += 1
            if passed:
                diff_counts[diff][0] += 1

        completion_rate = n_completed / n_tasks if n_tasks > 0 else 0.0
        avg_steps = total_steps / n_tasks if n_tasks > 0 else 0.0

        by_app = {
            app: (counts[0] / counts[1] if counts[1] > 0 else 0.0)
            for app, counts in app_counts.items()
        }
        by_difficulty = {
            diff: (counts[0] / counts[1] if counts[1] > 0 else 0.0)
            for diff, counts in diff_counts.items()
        }

        return OSWorldMetrics(
            n_tasks=n_tasks,
            n_completed=n_completed,
            completion_rate=completion_rate,
            by_app=by_app,
            by_difficulty=by_difficulty,
            avg_steps=avg_steps,
        )

    def format_report(self, metrics: OSWorldMetrics) -> str:
        """Return a human-readable summary of *metrics*."""
        lines: list[str] = [
            "OSWorld Evaluation Report",
            "=" * 40,
            f"Tasks evaluated : {metrics.n_tasks}",
            f"Tasks completed : {metrics.n_completed}",
            f"Completion rate : {metrics.completion_rate:.1%}",
            f"Avg steps taken : {metrics.avg_steps:.1f}",
            "",
            "By application:",
        ]
        for app, rate in sorted(metrics.by_app.items()):
            lines.append(f"  {app:<20} {rate:.1%}")
        lines.append("")
        lines.append("By difficulty:")
        for diff, rate in sorted(metrics.by_difficulty.items()):
            lines.append(f"  {diff:<20} {rate:.1%}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-registered stub tasks (6 tasks, 3 apps, all difficulties)
# ---------------------------------------------------------------------------

OSWORLD_TASK_REGISTRY: dict[str, OSWorldTask] = {
    "chrome-easy-001": OSWorldTask(
        task_id="chrome-easy-001",
        app="chrome",
        instruction="Open a new tab in Chrome.",
        reference_answers=["New Tab", "chrome://newtab"],
        subtasks=["click new tab button"],
        difficulty="easy",
    ),
    "chrome-medium-001": OSWorldTask(
        task_id="chrome-medium-001",
        app="chrome",
        instruction="Navigate to https://example.com and take a screenshot.",
        reference_answers=["example.com", "Example Domain"],
        subtasks=["enter URL", "press enter", "wait for load"],
        difficulty="medium",
    ),
    "vscode-easy-001": OSWorldTask(
        task_id="vscode-easy-001",
        app="vscode",
        instruction="Open the command palette in VS Code.",
        reference_answers=["Command Palette", ">"],
        subtasks=["press Ctrl+Shift+P"],
        difficulty="easy",
    ),
    "vscode-hard-001": OSWorldTask(
        task_id="vscode-hard-001",
        app="vscode",
        instruction="Install the Python extension in VS Code.",
        reference_answers=["Python", "ms-python", "installed"],
        subtasks=["open extensions view", "search Python", "click install"],
        difficulty="hard",
    ),
    "terminal-medium-001": OSWorldTask(
        task_id="terminal-medium-001",
        app="terminal",
        instruction="List all files in the home directory.",
        reference_answers=["Desktop", "Documents", "Downloads"],
        subtasks=["open terminal", "type ls ~", "press enter"],
        difficulty="medium",
    ),
    "terminal-hard-001": OSWorldTask(
        task_id="terminal-hard-001",
        app="terminal",
        instruction="Create a Python virtual environment named 'env' and activate it.",
        reference_answers=["(env)", "env/bin/activate"],
        subtasks=["python3 -m venv env", "source env/bin/activate"],
        difficulty="hard",
    ),
}


# ---------------------------------------------------------------------------
# Wire into BENCHMARK_REGISTRY and EVAL_HARNESS_REGISTRY from src.eval
# ---------------------------------------------------------------------------
try:
    import src.eval as _eval_pkg  # type: ignore[attr-defined]

    _BENCHMARK_REGISTRY: dict = getattr(_eval_pkg, "BENCHMARK_REGISTRY", {})
    _BENCHMARK_REGISTRY.setdefault("osworld", OSWorldScorer)

    # EVAL_HARNESS_REGISTRY may not exist yet — create it on the package.
    if not hasattr(_eval_pkg, "EVAL_HARNESS_REGISTRY"):
        _eval_pkg.EVAL_HARNESS_REGISTRY = {}
    _eval_pkg.EVAL_HARNESS_REGISTRY.setdefault("osworld", OSWorldScorer)
except Exception:  # noqa: BLE001
    # Running in isolation (e.g. during module unit tests before package init).
    pass
