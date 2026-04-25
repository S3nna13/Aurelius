from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalTask:
    task_id: str
    prompt: str
    reference: str
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalMetrics:
    task_id: str
    prediction: str
    exact_match: bool
    token_f1: float
    char_overlap: float
    passed: bool


class EvalHarness:
    """Pluggable evaluation harness for text generation tasks."""

    def __init__(self, tasks: list[EvalTask] | None = None) -> None:
        self._tasks: list[EvalTask] = list(tasks) if tasks is not None else []

    def add_task(self, task: EvalTask) -> None:
        self._tasks.append(task)

    def load_tasks(self, tasks: list[EvalTask]) -> None:
        self._tasks = list(tasks)

    def _token_f1(self, pred: str, ref: str) -> float:
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        if not pred_tokens or not ref_tokens:
            return 1.0 if pred_tokens == ref_tokens else 0.0
        pred_set = {}
        for t in pred_tokens:
            pred_set[t] = pred_set.get(t, 0) + 1
        ref_set = {}
        for t in ref_tokens:
            ref_set[t] = ref_set.get(t, 0) + 1
        overlap = sum(min(pred_set.get(t, 0), ref_set[t]) for t in ref_set)
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred_tokens)
        recall = overlap / len(ref_tokens)
        return 2.0 * precision * recall / (precision + recall)

    def _char_overlap(self, pred: str, ref: str) -> float:
        if not pred and not ref:
            return 1.0
        if not pred or not ref:
            return 0.0
        pred_chars: dict[str, int] = {}
        for c in pred:
            pred_chars[c] = pred_chars.get(c, 0) + 1
        ref_chars: dict[str, int] = {}
        for c in ref:
            ref_chars[c] = ref_chars.get(c, 0) + 1
        overlap = sum(min(pred_chars.get(c, 0), ref_chars[c]) for c in ref_chars)
        return overlap / max(len(pred), len(ref))

    def evaluate_one(self, task: EvalTask, prediction: str) -> EvalMetrics:
        exact = prediction == task.reference
        tf1 = self._token_f1(prediction, task.reference)
        co = self._char_overlap(prediction, task.reference)
        return EvalMetrics(
            task_id=task.task_id,
            prediction=prediction,
            exact_match=exact,
            token_f1=tf1,
            char_overlap=co,
            passed=exact,
        )

    def evaluate_all(self, predictions: list[str]) -> list[EvalMetrics]:
        return [
            self.evaluate_one(task, pred)
            for task, pred in zip(self._tasks, predictions)
        ]

    def aggregate(self, metrics: list[EvalMetrics]) -> dict:
        n = len(metrics)
        if n == 0:
            return {
                "exact_match_rate": 0.0,
                "mean_token_f1": 0.0,
                "mean_char_overlap": 0.0,
                "pass_rate": 0.0,
                "n_tasks": 0,
            }
        return {
            "exact_match_rate": sum(m.exact_match for m in metrics) / n,
            "mean_token_f1": sum(m.token_f1 for m in metrics) / n,
            "mean_char_overlap": sum(m.char_overlap for m in metrics) / n,
            "pass_rate": sum(m.passed for m in metrics) / n,
            "n_tasks": n,
        }

    def filter_failed(self, metrics: list[EvalMetrics]) -> list[EvalMetrics]:
        return [m for m in metrics if not m.passed]
