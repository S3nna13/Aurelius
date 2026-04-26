"""τ-bench scoring harness (Sierra Research, 2024).

tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains.
Measures an agent's ability to complete multi-step tasks by calling tools and
interacting with a simulated user/environment over multi-turn trajectories.

Core metric: task success rate. Additional metrics: avg turns, tool F1
accuracy, efficiency score (success / avg turns), and per-domain breakdown.

Pure stdlib + optional torch for tensor ops. No scipy, sklearn, HuggingFace,
or other external ML libs. All inputs treated as untrusted; no crash on
malformed data.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TauBenchTask:
    """Specification for a single τ-bench task."""

    task_id: str
    instruction: str  # user instruction given to agent
    tools: list[dict[str, Any]]  # tool schemas available to agent
    expected_actions: list[str]  # sequence of tool calls expected
    success_criteria: dict[str, Any]  # what counts as success
    domain: str = "coding"  # e.g. "retail", "airline", "coding"
    max_turns: int = 30

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("task_id must be a non-empty string")
        if self.max_turns < 1:
            raise ValueError(f"max_turns must be >= 1; got {self.max_turns}")


@dataclass
class TauBenchTrajectory:
    """Recorded execution of an agent on a τ-bench task."""

    task_id: str
    turns: list[dict[str, Any]]  # list of {role, content, tool_calls, tool_results}
    final_state: dict[str, Any]  # final environment state
    success: bool
    num_turns: int

    def __post_init__(self) -> None:
        if self.num_turns < 0:
            raise ValueError(f"num_turns must be >= 0; got {self.num_turns}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_\-\.]+$")


def _safe_str(v: Any) -> str:
    """Coerce a value to str without raising."""
    try:
        return str(v)
    except Exception:
        return ""


def _extract_tool_calls_from_turns(turns: list[dict[str, Any]]) -> list[str]:
    """Extract ordered list of tool names called across all turns.

    Each turn may have a 'tool_calls' key whose value is either:
    - A list of str (tool names).
    - A list of dict with a 'name' or 'function.name' key.
    - A single str.
    Missing or malformed entries are silently skipped.
    """
    names: list[str] = []
    if not turns:
        return names
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        raw = turn.get("tool_calls")
        if raw is None:
            continue
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, str) and item:
                names.append(item)
            elif isinstance(item, dict):
                # OpenAI-style: {name: ..., arguments: ...} or
                # {function: {name: ...}}
                name = item.get("name") or (item.get("function", {}) or {}).get("name")
                if name:
                    names.append(_safe_str(name))
    return names


def _state_matches(criteria_state: dict[str, Any], final_state: dict[str, Any]) -> bool:
    """Check if final_state satisfies all required_state key/value pairs."""
    if not isinstance(criteria_state, dict) or not isinstance(final_state, dict):
        return False
    for k, v in criteria_state.items():
        if final_state.get(k) != v:
            return False
    return True


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------


class TauBenchScorer:
    """Evaluates agent trajectories against τ-bench tasks.

    Metrics
    -------
    success_rate      : fraction of tasks where the agent achieved its goal.
    avg_turns         : mean turns per task (over the full batch).
    tool_accuracy     : macro-average F1 over tool call sequences.
    efficiency_score  : success_rate / avg_turns * max_turns_normaliser
                        (higher = more efficient; 0 if avg_turns == 0).
    domain_breakdown  : per-domain success rates and average turns.
    """

    def score_trajectory(
        self,
        task: TauBenchTask,
        trajectory: TauBenchTrajectory,
    ) -> dict[str, Any]:
        """Score a single trajectory against a task.

        Returns
        -------
        dict with keys:
          success        : bool
          turns          : int
          tool_accuracy  : float  (0.0–1.0)
          partial_credit : float  (0.0–1.0)
        """
        try:
            success = self.check_success(task, trajectory)
        except Exception:
            success = False

        try:
            actual_tools = _extract_tool_calls_from_turns(
                trajectory.turns if trajectory.turns else []
            )
            ta = self.compute_tool_accuracy(task.expected_actions, actual_tools)
        except Exception:
            ta = 0.0

        try:
            pc = self.partial_credit(task, trajectory)
        except Exception:
            pc = 0.0

        turns = max(0, trajectory.num_turns if isinstance(trajectory.num_turns, int) else 0)

        return {
            "success": success,
            "turns": turns,
            "tool_accuracy": ta,
            "partial_credit": pc,
        }

    def score_batch(
        self,
        tasks: list[TauBenchTask],
        trajectories: list[TauBenchTrajectory],
    ) -> dict[str, Any]:
        """Aggregate metrics over a batch of (task, trajectory) pairs.

        Returns
        -------
        dict with keys:
          success_rate     : float
          avg_turns        : float
          tool_accuracy    : float
          efficiency_score : float
          domain_breakdown : dict[domain -> {success_rate, avg_turns, n}]
          per_task         : list[dict]  (individual score_trajectory results)
          n_tasks          : int
        """
        if not tasks or not trajectories:
            return {
                "success_rate": 0.0,
                "avg_turns": 0.0,
                "tool_accuracy": 0.0,
                "efficiency_score": 0.0,
                "domain_breakdown": {},
                "per_task": [],
                "n_tasks": 0,
            }

        n = min(len(tasks), len(trajectories))
        per_task = []
        domain_successes: dict[str, list[bool]] = defaultdict(list)
        domain_turns: dict[str, list[int]] = defaultdict(list)

        for i in range(n):
            try:
                result = self.score_trajectory(tasks[i], trajectories[i])
            except Exception:
                result = {
                    "success": False,
                    "turns": 0,
                    "tool_accuracy": 0.0,
                    "partial_credit": 0.0,
                }
            per_task.append(result)
            domain = _safe_str(getattr(tasks[i], "domain", "unknown"))
            domain_successes[domain].append(bool(result["success"]))
            domain_turns[domain].append(int(result["turns"]))

        successes = [r["success"] for r in per_task]
        turns_list = [r["turns"] for r in per_task]
        tool_acc_list = [r["tool_accuracy"] for r in per_task]

        success_rate = sum(successes) / n
        avg_turns = sum(turns_list) / n if n > 0 else 0.0
        tool_accuracy = sum(tool_acc_list) / n if n > 0 else 0.0

        # Efficiency: reward agents that succeed with fewer turns.
        # Normalised so that completing every task in 1 turn → 1.0.
        if avg_turns > 0:
            efficiency_score = success_rate / avg_turns
        else:
            efficiency_score = 0.0

        domain_breakdown: dict[str, dict[str, Any]] = {}
        for dom in domain_successes:
            d_succ = domain_successes[dom]
            d_turns = domain_turns[dom]
            domain_breakdown[dom] = {
                "success_rate": sum(d_succ) / len(d_succ) if d_succ else 0.0,
                "avg_turns": sum(d_turns) / len(d_turns) if d_turns else 0.0,
                "n": len(d_succ),
            }

        return {
            "success_rate": success_rate,
            "avg_turns": avg_turns,
            "tool_accuracy": tool_accuracy,
            "efficiency_score": efficiency_score,
            "domain_breakdown": domain_breakdown,
            "per_task": per_task,
            "n_tasks": n,
        }

    def check_success(
        self,
        task: TauBenchTask,
        trajectory: TauBenchTrajectory,
    ) -> bool:
        """Check if a trajectory satisfies the task's success_criteria.

        Success criteria keys (all optional; OR logic between methods):
          required_tool_calls : list[str]  — all of these must have been called
          required_state      : dict       — key/value pairs that must be in
                                            trajectory.final_state
          exact_tool_sequence : list[str]  — tool calls in exact order
          custom_fn           : not supported at runtime (ignored safely)

        If success_criteria is empty → fall back to trajectory.success flag.
        """
        if not isinstance(task, TauBenchTask) or not isinstance(trajectory, TauBenchTrajectory):
            return False

        criteria = task.success_criteria if isinstance(task.success_criteria, dict) else {}
        if not criteria:
            return bool(trajectory.success)

        actual_tools = _extract_tool_calls_from_turns(
            trajectory.turns if isinstance(trajectory.turns, list) else []
        )
        actual_set = set(actual_tools)
        final_state = trajectory.final_state if isinstance(trajectory.final_state, dict) else {}

        checks: list[bool] = []

        # 1. required_tool_calls — all tools in set must appear at least once
        required_calls = criteria.get("required_tool_calls")
        if required_calls is not None and isinstance(required_calls, list):
            checks.append(all(_safe_str(t) in actual_set for t in required_calls))

        # 2. required_state — final_state must contain these key/value pairs
        required_state = criteria.get("required_state")
        if required_state is not None and isinstance(required_state, dict):
            checks.append(_state_matches(required_state, final_state))

        # 3. exact_tool_sequence — tool calls must match sequence exactly
        exact_seq = criteria.get("exact_tool_sequence")
        if exact_seq is not None and isinstance(exact_seq, list):
            expected_str = [_safe_str(t) for t in exact_seq]
            checks.append(actual_tools == expected_str)

        if not checks:
            # Unknown criteria keys only → fall back to trajectory flag
            return bool(trajectory.success)

        return all(checks)

    def compute_tool_accuracy(
        self,
        expected: list[str],
        actual: list[str],
    ) -> float:
        """F1-style tool call accuracy (multiset precision / recall / F1).

        Treats expected and actual as bags-of-words over tool names.
        Returns the macro F1 in [0.0, 1.0].

        Edge cases:
          both empty  → 1.0 (perfect match, no tools needed and none called)
          one empty   → 0.0
        """
        if not isinstance(expected, list):
            expected = []
        if not isinstance(actual, list):
            actual = []

        # Normalise to str
        expected = [_safe_str(t) for t in expected if t]
        actual = [_safe_str(t) for t in actual if t]

        if not expected and not actual:
            return 1.0
        if not expected or not actual:
            return 0.0

        # Multiset intersection counts
        exp_counts: dict[str, int] = defaultdict(int)
        act_counts: dict[str, int] = defaultdict(int)
        for t in expected:
            exp_counts[t] += 1
        for t in actual:
            act_counts[t] += 1

        tp = sum(min(exp_counts[t], act_counts[t]) for t in exp_counts)

        precision = tp / len(actual)
        recall = tp / len(expected)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def partial_credit(
        self,
        task: TauBenchTask,
        trajectory: TauBenchTrajectory,
    ) -> float:
        """Compute partial credit in [0.0, 1.0] based on ordered progress.

        Scores the longest prefix of expected_actions that appears (in order,
        not necessarily contiguously) in the actual tool call sequence.
        Full credit (1.0) is given when all expected actions are completed.

        Returns 0.0 if expected_actions is empty or trajectory has no turns.
        """
        expected = task.expected_actions if isinstance(task.expected_actions, list) else []
        expected = [_safe_str(t) for t in expected if t]

        if not expected:
            # No expected actions defined: reward success, penalise failure
            return 1.0 if self.check_success(task, trajectory) else 0.0

        turns = trajectory.turns if isinstance(trajectory.turns, list) else []
        actual = _extract_tool_calls_from_turns(turns)

        # Longest common subsequence length (LCS) of expected vs actual
        # using greedy prefix matching (ordered subsequence, not contiguous)
        exp_idx = 0
        for tool in actual:
            if exp_idx >= len(expected):
                break
            if tool == expected[exp_idx]:
                exp_idx += 1

        return exp_idx / len(expected)


# ---------------------------------------------------------------------------
# Built-in synthetic dataset
# ---------------------------------------------------------------------------

_CODING_TASKS_TEMPLATE = [
    {
        "instruction": "Search for the function `parse_config` in the codebase and report which files define it.",  # noqa: E501
        "tools": [
            {"name": "search_code", "description": "Search codebase for a pattern"},
            {"name": "read_file", "description": "Read a file by path"},
        ],
        "expected_actions": ["search_code", "read_file"],
        "success_criteria": {
            "required_tool_calls": ["search_code"],
            "required_state": {"reported": True},
        },
        "domain": "coding",
    },
    {
        "instruction": "Run the test suite for the `auth` module and fix any failing test.",
        "tools": [
            {"name": "run_tests", "description": "Execute pytest on a module"},
            {"name": "edit_file", "description": "Edit a source file"},
        ],
        "expected_actions": ["run_tests", "edit_file", "run_tests"],
        "success_criteria": {
            "required_tool_calls": ["run_tests", "edit_file"],
            "required_state": {"all_tests_pass": True},
        },
        "domain": "coding",
    },
    {
        "instruction": "Create a new Python module `utils/retry.py` with exponential backoff.",
        "tools": [
            {"name": "create_file", "description": "Create a new file"},
            {"name": "edit_file", "description": "Edit a source file"},
        ],
        "expected_actions": ["create_file"],
        "success_criteria": {
            "required_tool_calls": ["create_file"],
            "required_state": {"file_created": True},
        },
        "domain": "coding",
    },
    {
        "instruction": "Refactor the `DataLoader` class to use lazy evaluation.",
        "tools": [
            {"name": "read_file", "description": "Read a file"},
            {"name": "edit_file", "description": "Edit a source file"},
            {"name": "run_tests", "description": "Run tests"},
        ],
        "expected_actions": ["read_file", "edit_file", "run_tests"],
        "success_criteria": {
            "required_tool_calls": ["edit_file"],
        },
        "domain": "coding",
    },
    {
        "instruction": "List all TODO comments in the repository.",
        "tools": [
            {"name": "search_code", "description": "Search for a pattern"},
        ],
        "expected_actions": ["search_code"],
        "success_criteria": {
            "required_tool_calls": ["search_code"],
        },
        "domain": "coding",
    },
]

_RETAIL_TASKS_TEMPLATE = [
    {
        "instruction": "Look up order #12345 and check its shipping status.",
        "tools": [
            {"name": "lookup_order", "description": "Look up an order by ID"},
            {"name": "get_shipping_status", "description": "Get shipping status"},
        ],
        "expected_actions": ["lookup_order", "get_shipping_status"],
        "success_criteria": {
            "required_tool_calls": ["lookup_order", "get_shipping_status"],
        },
        "domain": "retail",
    },
    {
        "instruction": "Process a return for order #98765 and issue a refund.",
        "tools": [
            {"name": "lookup_order", "description": "Look up an order"},
            {"name": "initiate_return", "description": "Start a return"},
            {"name": "issue_refund", "description": "Issue a refund"},
        ],
        "expected_actions": ["lookup_order", "initiate_return", "issue_refund"],
        "success_criteria": {
            "required_tool_calls": ["initiate_return", "issue_refund"],
        },
        "domain": "retail",
    },
    {
        "instruction": "Find the top 3 best-selling products this month.",
        "tools": [
            {"name": "query_sales_data", "description": "Query sales database"},
        ],
        "expected_actions": ["query_sales_data"],
        "success_criteria": {
            "required_tool_calls": ["query_sales_data"],
        },
        "domain": "retail",
    },
]

_AIRLINE_TASKS_TEMPLATE = [
    {
        "instruction": "Book a round-trip flight from SFO to JFK for next Monday.",
        "tools": [
            {"name": "search_flights", "description": "Search available flights"},
            {"name": "book_flight", "description": "Book a selected flight"},
        ],
        "expected_actions": ["search_flights", "book_flight"],
        "success_criteria": {
            "required_tool_calls": ["search_flights", "book_flight"],
        },
        "domain": "airline",
    },
    {
        "instruction": "Cancel reservation PNR-ABCDEF and request a refund.",
        "tools": [
            {"name": "lookup_reservation", "description": "Retrieve a reservation"},
            {"name": "cancel_reservation", "description": "Cancel a reservation"},
            {"name": "request_refund", "description": "Request refund"},
        ],
        "expected_actions": ["lookup_reservation", "cancel_reservation", "request_refund"],
        "success_criteria": {
            "required_tool_calls": ["cancel_reservation"],
        },
        "domain": "airline",
    },
]

_DOMAIN_TEMPLATES = {
    "coding": _CODING_TASKS_TEMPLATE,
    "retail": _RETAIL_TASKS_TEMPLATE,
    "airline": _AIRLINE_TASKS_TEMPLATE,
}


class TauBenchDataset:
    """Minimal built-in dataset with synthetic τ-bench-style tasks.

    Provides deterministic tasks for unit / integration testing without
    requiring network access or external files.
    """

    @staticmethod
    def get_sample_tasks(domain: str = "coding", n: int = 10) -> list[TauBenchTask]:
        """Return up to *n* synthetic TauBenchTask objects for *domain*.

        If *domain* is unknown, returns coding tasks. Repeats templates
        cyclically to satisfy *n*.
        """
        templates = _DOMAIN_TEMPLATES.get(domain, _CODING_TASKS_TEMPLATE)
        if not templates:
            templates = _CODING_TASKS_TEMPLATE

        tasks: list[TauBenchTask] = []
        for i in range(max(0, n)):
            tmpl = templates[i % len(templates)]
            tasks.append(
                TauBenchTask(
                    task_id=f"{domain}-{i:04d}",
                    instruction=tmpl["instruction"],
                    tools=list(tmpl["tools"]),
                    expected_actions=list(tmpl["expected_actions"]),
                    success_criteria=dict(tmpl["success_criteria"]),
                    domain=domain,
                    max_turns=30,
                )
            )
        return tasks

    @staticmethod
    def make_successful_trajectory(task: TauBenchTask) -> TauBenchTrajectory:
        """Construct a synthetic trajectory that satisfies a task's criteria."""
        turns = []
        for action in task.expected_actions:
            turns.append(
                {
                    "role": "assistant",
                    "content": f"Calling {action}",
                    "tool_calls": [action],
                    "tool_results": [{"status": "ok"}],
                }
            )
        # Build final_state from required_state if present
        final_state: dict[str, Any] = {}
        if "required_state" in task.success_criteria:
            final_state.update(task.success_criteria["required_state"])

        return TauBenchTrajectory(
            task_id=task.task_id,
            turns=turns,
            final_state=final_state,
            success=True,
            num_turns=len(turns),
        )
