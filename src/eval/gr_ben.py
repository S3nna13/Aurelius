"""GR-Ben: General Reasoning Benchmark for Process Reward Models.

Comprehensive benchmark for evaluating PRMs across science, logic, and
math domains. Identifies that PRMs excel at math but struggle with
knowledge-based errors while LLMs struggle with computational errors.

Paper: arXiv:2605.01203 — Sun et al.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class PRMInterface(Protocol):
    """Protocol for any process reward model being evaluated."""

    def score_step(self, prompt: str, steps: list[str]) -> list[float]:
        """Score each reasoning step. Returns list of scores."""
        ...


@dataclass
class GRBenConfig:
    domains: list[str] | None = None
    subdomains: list[str] | None = None
    evaluation_metrics: tuple[str, ...] = ("auroc", "accuracy", "f1")


class GRBenEvaluator:
    """Benchmark evaluator for PRMs on general reasoning tasks."""

    def __init__(self, prm: PRMInterface, config: GRBenConfig | None = None) -> None:
        self.prm = prm
        self.cfg = config or GRBenConfig()

    def evaluate_domain(self, domain: str, test_data: list[dict]) -> dict:
        """Evaluate PRM on a specific reasoning domain.

        Returns metrics including AUROC for error detection,
        per-step accuracy, and domain-specific breakdowns.
        """
        total_correct = 0
        total_errors = 0
        all_scores = []
        all_labels = []

        for item in test_data or []:
            prompt = item.get("prompt")
            steps = item.get("steps", [])
            labels = item.get("step_correctness", [])
            if not prompt or not steps or not labels:
                continue
            scores = self.prm.score_step(prompt, steps)

            for score, label in zip(scores, labels):
                all_scores.append(score)
                all_labels.append(label)
                if label == 1:
                    total_errors += 1
                else:
                    total_correct += 1

        if not all_scores:
            return {
                "domain": domain,
                "auroc": float("nan"),
                "accuracy": float("nan"),
                "f1": float("nan"),
                "n_samples": 0,
            }

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        predictions = [s > 0.5 for s in all_scores]
        auroc = roc_auc_score(all_labels, all_scores) if len(set(all_labels)) > 1 else float("nan")
        acc = accuracy_score(all_labels, predictions)
        f1 = f1_score(all_labels, predictions, zero_division=0)

        return {
            "domain": domain,
            "auroc": auroc,
            "accuracy": acc,
            "f1": f1,
            "n_samples": len(all_scores),
        }

    def run_full_benchmark(self, benchmark_data: dict[str, list]) -> dict:
        """Run GR-Ben across all domains and subdomains."""
        results = {}
        for domain, data in benchmark_data.items():
            results[domain] = self.evaluate_domain(domain, data)
        return results


@dataclass
class PRMEvaluationResult:
    domain: str
    subdomain: str
    auroc: float
    accuracy: float
    knowledge_based_error_detection: float
    computational_error_detection: float
    recommendations: list[str]


__all__ = ["GRBenEvaluator", "GRBenConfig", "PRMInterface", "PRMEvaluationResult"]
