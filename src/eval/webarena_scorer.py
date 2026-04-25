"""WebArena task-completion scorer for autonomous web agent evaluation.

WebArena benchmark (Zhou et al. 2307.13854, Apache-2.0), GitHub Actions (MIT spec),
clean-room implementation.

Reference: Shuyan Zhou et al., "WebArena: A Realistic Web Environment for Building
Autonomous Agents", arXiv:2307.13854.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class WebArenaTask:
    """A single WebArena task specification."""

    task_id: str
    category: str
    intent: str
    start_url: str
    reference_answers: list[str]
    difficulty: Literal["easy", "medium", "hard"] = "medium"


@dataclass
class WebArenaResult:
    """The result of running an agent on a WebArenaTask."""

    task_id: str
    predicted_answer: str | None
    steps_taken: int
    success: bool = False
    trajectory_len: int = 0


@dataclass
class WebArenaMetrics:
    """Aggregate metrics over a batch of WebArena results."""

    n_tasks: int
    n_success: int
    success_rate: float
    by_category: dict[str, float]
    by_difficulty: dict[str, float]
    avg_steps: float


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class WebArenaScorer:
    """Scores WebArena task results against reference answers.

    Scoring uses case-insensitive substring matching: a result is considered
    successful if any reference answer appears as a substring of the predicted
    answer (case-insensitive).
    """

    def score_result(self, result: WebArenaResult, task: WebArenaTask) -> bool:
        """Return True if predicted_answer contains any reference answer (case-insensitive).

        Args:
            result: The agent's result for the task.
            task: The corresponding task specification.

        Returns:
            True if matched, False otherwise.
        """
        if result.predicted_answer is None:
            return False
        predicted_lower = result.predicted_answer.lower()
        return any(ref.lower() in predicted_lower for ref in task.reference_answers)

    def score_batch(
        self,
        results: list[WebArenaResult],
        tasks: dict[str, WebArenaTask],
    ) -> WebArenaMetrics:
        """Compute aggregate metrics over a list of results.

        Tasks not present in ``tasks`` are skipped gracefully.

        Args:
            results: List of agent results.
            tasks: Mapping from task_id to WebArenaTask.

        Returns:
            WebArenaMetrics with overall and per-category/difficulty breakdowns.
        """
        category_success: dict[str, list[bool]] = {}
        difficulty_success: dict[str, list[bool]] = {}
        total_steps: list[int] = []
        n_tasks = 0
        n_success = 0

        for result in results:
            task = tasks.get(result.task_id)
            if task is None:
                continue  # skip gracefully

            success = self.score_result(result, task)
            n_tasks += 1
            if success:
                n_success += 1

            total_steps.append(result.steps_taken)

            category_success.setdefault(task.category, []).append(success)
            difficulty_success.setdefault(task.difficulty, []).append(success)

        success_rate = (n_success / n_tasks) if n_tasks > 0 else 0.0
        avg_steps = (sum(total_steps) / len(total_steps)) if total_steps else 0.0

        by_category = {
            cat: sum(vals) / len(vals)
            for cat, vals in category_success.items()
        }
        by_difficulty = {
            diff: sum(vals) / len(vals)
            for diff, vals in difficulty_success.items()
        }

        return WebArenaMetrics(
            n_tasks=n_tasks,
            n_success=n_success,
            success_rate=success_rate,
            by_category=by_category,
            by_difficulty=by_difficulty,
            avg_steps=avg_steps,
        )

    def format_report(self, metrics: WebArenaMetrics) -> str:
        """Render a human-readable text report of WebArena metrics.

        Args:
            metrics: The metrics to format.

        Returns:
            A multi-line string report.
        """
        lines: list[str] = [
            "=" * 60,
            "WebArena Benchmark Report",
            "=" * 60,
            f"Tasks evaluated : {metrics.n_tasks}",
            f"Tasks succeeded : {metrics.n_success}",
            f"Success rate    : {metrics.success_rate:.1%}",
            f"Average steps   : {metrics.avg_steps:.1f}",
            "",
            "By category:",
        ]
        for cat, rate in sorted(metrics.by_category.items()):
            lines.append(f"  {cat:<20} {rate:.1%}")

        lines.append("")
        lines.append("By difficulty:")
        for diff, rate in sorted(metrics.by_difficulty.items()):
            lines.append(f"  {diff:<20} {rate:.1%}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pre-registered stub tasks (5 tasks across 3 categories, all 3 difficulties)
# ---------------------------------------------------------------------------

WEBARENA_TASK_REGISTRY: dict[str, WebArenaTask] = {
    "shopping_001": WebArenaTask(
        task_id="shopping_001",
        category="shopping",
        intent="Find the cheapest red sneakers under $50 and add them to the cart.",
        start_url="http://shop.webarena.example/",
        reference_answers=["added to cart", "cart updated"],
        difficulty="easy",
    ),
    "shopping_002": WebArenaTask(
        task_id="shopping_002",
        category="shopping",
        intent="Compare the reviews of the top 3 laptops and report the highest-rated one.",
        start_url="http://shop.webarena.example/electronics",
        reference_answers=["ThinkPad X1", "MacBook Pro", "Dell XPS"],
        difficulty="medium",
    ),
    "reddit_001": WebArenaTask(
        task_id="reddit_001",
        category="reddit",
        intent="Find the most upvoted post in r/python this week and report its title.",
        start_url="http://reddit.webarena.example/r/python",
        reference_answers=["python", "release", "tutorial"],
        difficulty="medium",
    ),
    "reddit_002": WebArenaTask(
        task_id="reddit_002",
        category="reddit",
        intent="Post a comment on the pinned announcement in r/webdev thanking the moderators.",
        start_url="http://reddit.webarena.example/r/webdev",
        reference_answers=["comment posted", "your comment"],
        difficulty="hard",
    ),
    "gitlab_001": WebArenaTask(
        task_id="gitlab_001",
        category="gitlab",
        intent="Open an issue titled 'CI pipeline fails on Python 3.14' in the aurelius repo.",
        start_url="http://gitlab.webarena.example/aurelius",
        reference_answers=["issue created", "issue #"],
        difficulty="hard",
    ),
}

# ---------------------------------------------------------------------------
# Scorer registry
# ---------------------------------------------------------------------------

WEBARENA_SCORER_REGISTRY: dict[str, type[WebArenaScorer]] = {
    "default": WebArenaScorer,
}
