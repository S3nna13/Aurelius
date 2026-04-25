"""Federated evaluation protocol.

Coordinates an evaluation task across multiple clients, each of whom
evaluates the shared model on their local data. Results are weighted
by sample count and summarized with a population standard deviation.
"""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field


def _new_task_id() -> str:
    return uuid.uuid4().hex[:8]


@dataclass(frozen=True)
class EvalTask:
    """A federated evaluation task."""

    dataset_name: str
    metric_name: str
    split: str = "test"
    max_samples: int = 100
    task_id: str = field(default_factory=_new_task_id)


@dataclass(frozen=True)
class ClientEvalResult:
    """One client's report for a federated evaluation task."""

    client_id: str
    task_id: str
    metric_value: float
    num_samples: int
    error: str = ""


@dataclass(frozen=True)
class FederatedEvalResult:
    """Aggregated federated evaluation result for a task."""

    task_id: str
    aggregated_metric: float
    client_results: list[ClientEvalResult]
    num_clients: int
    std_dev: float


class FederatedEvaluator:
    """Coordinates federated evaluation across registered clients."""

    def __init__(self) -> None:
        self._clients: list[str] = []
        self._tasks: dict[str, EvalTask] = {}
        self._results: dict[str, dict[str, ClientEvalResult]] = {}
        self._completed: list[str] = []

    def register_client(self, client_id: str) -> None:
        if client_id not in self._clients:
            self._clients.append(client_id)

    def submit_task(self, task: EvalTask) -> None:
        self._tasks[task.task_id] = task
        self._results.setdefault(task.task_id, {})

    def receive_result(self, result: ClientEvalResult) -> None:
        self._results.setdefault(result.task_id, {})
        self._results[result.task_id][result.client_id] = result

    def pending_clients(self, task_id: str) -> list[str]:
        submitted = set(self._results.get(task_id, {}).keys())
        return [c for c in self._clients if c not in submitted]

    def completed_tasks(self) -> list[str]:
        return list(self._completed)

    def aggregate(self, task_id: str) -> FederatedEvalResult | None:
        if task_id not in self._tasks:
            return None
        results_map = self._results.get(task_id, {})
        if not self._clients:
            return None
        if any(c not in results_map for c in self._clients):
            return None

        client_results = [results_map[c] for c in self._clients]
        total_samples = sum(r.num_samples for r in client_results)
        if total_samples > 0:
            aggregated = (
                sum(r.metric_value * r.num_samples for r in client_results)
                / total_samples
            )
        else:
            aggregated = sum(r.metric_value for r in client_results) / len(
                client_results
            )

        values = [r.metric_value for r in client_results]
        if len(values) >= 2:
            std_dev = statistics.pstdev(values)
        else:
            std_dev = 0.0

        if task_id not in self._completed:
            self._completed.append(task_id)

        return FederatedEvalResult(
            task_id=task_id,
            aggregated_metric=aggregated,
            client_results=client_results,
            num_clients=len(client_results),
            std_dev=std_dev,
        )


FEDERATED_EVALUATOR_REGISTRY = {"default": FederatedEvaluator}
