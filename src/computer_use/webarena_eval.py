"""WebArena-style task completion eval harness for the Aurelius computer_use surface.

Inspired by WebArena benchmark (Zhou et al. 2307.13854, Apache-2.0), OSWorld (Xie et al. 2024),
clean-room reimplementation.

No playwright, selenium, or external browser automation imports.
All browser interactions route through the abstract BrowserDriver interface.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.computer_use.browser_driver import BrowserDriver, BrowserState
    from src.computer_use.gui_action import GUIAction


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class WebArenaError(Exception):
    """Raised when a WebArena harness operation is invalid."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WebTask:
    """A single WebArena-style evaluation task.

    Attributes
    ----------
    task_id:
        Unique identifier for the task (e.g. ``"navigate_to_url"``).
    description:
        Human-readable description of what the agent must accomplish.
    start_url:
        The URL the browser should be navigated to before the agent starts.
    success_criteria:
        List of strings; at least one must appear in the final URL or page HTML
        for the task to be considered successful.
    max_steps:
        Maximum number of agent steps allowed (default 30, per WebArena paper).
    tags:
        Optional list of category tags for grouping/scoring (e.g. ``["navigation"]``).
    """

    task_id: str
    description: str
    start_url: str
    success_criteria: list[str]
    max_steps: int = 30
    tags: list[str] = field(default_factory=list)


@dataclass
class TaskResult:
    """The outcome of running a single :class:`WebTask`.

    Attributes
    ----------
    task_id:
        Matches :attr:`WebTask.task_id`.
    success:
        Whether the task was judged successful by :class:`SuccessEvaluator`.
    steps_taken:
        Number of agent steps executed (0 if agent returned ``None`` immediately).
    final_url:
        URL reported by the driver at task end, or ``None`` if unavailable.
    final_state_html:
        HTML snapshot at task end, or ``None`` if unavailable.
    error:
        Exception message if the run raised an unhandled error, else ``None``.
    trajectory_id:
        Optional reference to a recorded :class:`~src.computer_use.trajectory_replay.Trajectory`.
    """

    task_id: str
    success: bool
    steps_taken: int
    final_url: str | None
    final_state_html: str | None
    error: str | None = None
    trajectory_id: str | None = None


# ---------------------------------------------------------------------------
# Success evaluator
# ---------------------------------------------------------------------------

class SuccessEvaluator:
    """Evaluates whether a :class:`TaskResult` satisfies a :class:`WebTask`.

    Two evaluation strategies are supported (tried in order):

    1. **URL match** — checks whether any ``success_criterion`` string appears
       as a substring of ``result.final_url``.
    2. **Content match** — checks whether any ``success_criterion`` string
       appears as a substring of ``result.final_state_html``.

    This mirrors the evaluation logic in the WebArena paper (§4, Zhou et al. 2023).
    """

    # ------------------------------------------------------------------
    # Individual evaluators
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_url_match(result: TaskResult, task: WebTask) -> bool:
        """Return ``True`` if any success criterion appears in *result.final_url*.

        Parameters
        ----------
        result:
            The task result to inspect.
        task:
            The task whose criteria are checked.

        Returns
        -------
        bool
            ``False`` when *result.final_url* is ``None``.
        """
        if result.final_url is None:
            return False
        return any(criterion in result.final_url for criterion in task.success_criteria)

    @staticmethod
    def evaluate_content_match(result: TaskResult, task: WebTask) -> bool:
        """Return ``True`` if any success criterion appears in *result.final_state_html*.

        Parameters
        ----------
        result:
            The task result to inspect.
        task:
            The task whose criteria are checked.

        Returns
        -------
        bool
            ``False`` when *result.final_state_html* is ``None``.
        """
        if result.final_state_html is None:
            return False
        return any(
            criterion in result.final_state_html for criterion in task.success_criteria
        )

    # ------------------------------------------------------------------
    # Combined evaluator
    # ------------------------------------------------------------------

    def evaluate(self, result: TaskResult, task: WebTask) -> tuple[bool, str]:
        """Try URL match then content match; return ``(success, reason)``.

        Parameters
        ----------
        result:
            The task result to evaluate.
        task:
            The task being evaluated.

        Returns
        -------
        tuple[bool, str]
            A pair ``(success, reason_str)`` where *reason_str* explains which
            signal triggered the verdict (``"url_match"``, ``"content_match"``,
            or ``"no_match"``).
        """
        if self.evaluate_url_match(result, task):
            return True, "url_match"
        if self.evaluate_content_match(result, task):
            return True, "content_match"
        return False, "no_match"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class WebArenaHarness:
    """Registry and runner for WebArena-style evaluation tasks.

    Usage::

        harness = WebArenaHarness()
        harness.register_task(WebTask(...))
        result = harness.run_task("my_task", driver, agent_fn)
        scores = harness.score([result], harness.tasks)
    """

    def __init__(self) -> None:
        self.tasks: dict[str, WebTask] = {}
        self._evaluator: SuccessEvaluator = SuccessEvaluator()

    # ------------------------------------------------------------------
    # Task registry
    # ------------------------------------------------------------------

    def register_task(self, task: WebTask) -> None:
        """Add *task* to the registry.

        Parameters
        ----------
        task:
            A :class:`WebTask` to register.

        Raises
        ------
        WebArenaError
            If a task with the same ``task_id`` is already registered.
        """
        if task.task_id in self.tasks:
            raise WebArenaError(
                f"Task '{task.task_id}' is already registered in this harness. "
                "Use a unique task_id or remove the existing entry first."
            )
        self.tasks[task.task_id] = task

    # ------------------------------------------------------------------
    # Task runner
    # ------------------------------------------------------------------

    def run_task(
        self,
        task_id: str,
        driver: "BrowserDriver",
        agent_fn: "Callable[[BrowserState, WebTask], GUIAction | None]",
    ) -> TaskResult:
        """Run a registered task end-to-end and return a :class:`TaskResult`.

        The loop:

        1. Navigates *driver* to ``task.start_url``.
        2. Calls ``agent_fn(state, task)`` each step.
        3. If ``agent_fn`` returns ``None``, stops early.
        4. Executes the returned :class:`~src.computer_use.gui_action.GUIAction`
           via *driver* (CLICK or TYPE_TEXT; others are treated as no-ops).
        5. Checks success after each step; stops if success is detected.
        6. Catches all exceptions and returns a :class:`TaskResult` with
           ``error`` set — the harness never crashes the caller.

        Parameters
        ----------
        task_id:
            The ``task_id`` of a previously registered :class:`WebTask`.
        driver:
            A concrete :class:`~src.computer_use.browser_driver.BrowserDriver`
            instance (e.g. :class:`~src.computer_use.browser_driver.StubBrowserDriver`).
        agent_fn:
            A callable ``(BrowserState, WebTask) -> GUIAction | None``.
            Return ``None`` to signal that the agent is done.

        Returns
        -------
        TaskResult
            Always returns a result; never raises.
        """
        # Lazy imports to avoid circular import at module load time.
        from src.computer_use.gui_action import ActionType  # noqa: PLC0415

        try:
            task = self.tasks[task_id]
        except KeyError:
            return TaskResult(
                task_id=task_id,
                success=False,
                steps_taken=0,
                final_url=None,
                final_state_html=None,
                error=f"Task '{task_id}' is not registered in this harness.",
            )

        steps_taken = 0
        try:
            state = driver.navigate(task.start_url)

            for _ in range(task.max_steps):
                action = agent_fn(state, task)
                if action is None:
                    break

                # Dispatch the action to the driver.
                if action.action_type == ActionType.CLICK:
                    selector = action.target_selector or ""
                    state = driver.click(selector)
                elif action.action_type == ActionType.TYPE:
                    selector = action.target_selector or ""
                    text = action.value or ""
                    state = driver.type_text(selector, text)
                else:
                    # SCROLL, KEY, WAIT, SCREENSHOT → no-op (driver stays same state)
                    state = driver.get_state()

                steps_taken += 1

                # Early-exit on success.
                provisional = TaskResult(
                    task_id=task_id,
                    success=False,
                    steps_taken=steps_taken,
                    final_url=state.url,
                    final_state_html=state.html_snapshot,
                )
                success, _ = self._evaluator.evaluate(provisional, task)
                if success:
                    return TaskResult(
                        task_id=task_id,
                        success=True,
                        steps_taken=steps_taken,
                        final_url=state.url,
                        final_state_html=state.html_snapshot,
                    )

            # Loop exhausted or agent returned None — do a final evaluation.
            final_state = driver.get_state()
            provisional = TaskResult(
                task_id=task_id,
                success=False,
                steps_taken=steps_taken,
                final_url=final_state.url,
                final_state_html=final_state.html_snapshot,
            )
            success, _ = self._evaluator.evaluate(provisional, task)
            return TaskResult(
                task_id=task_id,
                success=success,
                steps_taken=steps_taken,
                final_url=final_state.url,
                final_state_html=final_state.html_snapshot,
            )

        except Exception as exc:  # noqa: BLE001
            # Capture the driver's last known state if possible.
            try:
                final_state = driver.get_state()
                final_url = final_state.url
                final_html = final_state.html_snapshot
            except Exception:  # noqa: BLE001
                final_url = None
                final_html = None

            return TaskResult(
                task_id=task_id,
                success=False,
                steps_taken=steps_taken,
                final_url=final_url,
                final_state_html=final_html,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Scorer
    # ------------------------------------------------------------------

    @staticmethod
    def score(
        results: list[TaskResult],
        tasks: dict[str, WebTask],
    ) -> dict:
        """Aggregate :class:`TaskResult` objects into a score report.

        Parameters
        ----------
        results:
            List of :class:`TaskResult` instances to score.
        tasks:
            Mapping of ``task_id → WebTask``; used to retrieve tag metadata.

        Returns
        -------
        dict
            A dictionary with the following keys:

            ``n_tasks`` (int)
                Total number of results evaluated.
            ``n_success`` (int)
                Number of successful results.
            ``success_rate`` (float)
                ``n_success / n_tasks``, or ``0.0`` if *results* is empty.
            ``by_tag`` (dict[str, float])
                Per-tag success rate, computed over all results whose task
                carries that tag.  Tasks with no tags are excluded from
                ``by_tag`` but still count toward the global rate.
        """
        n_tasks = len(results)
        n_success = sum(1 for r in results if r.success)
        success_rate = n_success / n_tasks if n_tasks > 0 else 0.0

        # Collect per-tag counts.
        tag_success: dict[str, int] = defaultdict(int)
        tag_total: dict[str, int] = defaultdict(int)
        for result in results:
            task = tasks.get(result.task_id)
            if task is None:
                continue
            for tag in task.tags:
                tag_total[tag] += 1
                if result.success:
                    tag_success[tag] += 1

        by_tag: dict[str, float] = {
            tag: tag_success[tag] / tag_total[tag]
            for tag in tag_total
        }

        return {
            "n_tasks": n_tasks,
            "n_success": n_success,
            "success_rate": success_rate,
            "by_tag": by_tag,
        }


# ---------------------------------------------------------------------------
# Default stub tasks (WebArena-inspired)
# ---------------------------------------------------------------------------

WEBARENA_DEFAULT_TASKS: list[WebTask] = [
    WebTask(
        task_id="navigate_to_url",
        description=(
            "Navigate the browser to a specific target URL and confirm the page loads."
        ),
        start_url="https://example.com",
        success_criteria=["example.com/target"],
        max_steps=5,
        tags=["navigation"],
    ),
    WebTask(
        task_id="find_element",
        description=(
            "Locate a specific DOM element on the current page by its label or role."
        ),
        start_url="https://example.com/search",
        success_criteria=["search", "results"],
        max_steps=10,
        tags=["navigation", "interaction"],
    ),
    WebTask(
        task_id="fill_form",
        description=(
            "Fill out an HTML form with the provided field values and submit it."
        ),
        start_url="https://example.com/form",
        success_criteria=["success", "thank-you", "submitted"],
        max_steps=15,
        tags=["interaction", "form"],
    ),
]


# ---------------------------------------------------------------------------
# Global harness registry
# ---------------------------------------------------------------------------

WEBARENA_HARNESS_REGISTRY: dict[str, WebArenaHarness] = {}
