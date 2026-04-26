"""A/B model comparison: statistical significance testing, effect size, win/loss matrix."""

import math
from dataclasses import dataclass
from enum import StrEnum


class ComparisonMetric(StrEnum):
    ACCURACY = "accuracy"
    BLEU = "bleu"
    ROUGE_L = "rouge_l"
    EXACT_MATCH = "exact_match"
    PREFERENCE = "preference"


@dataclass
class ModelComparison:
    model_a: str
    model_b: str
    metric: ComparisonMetric
    score_a: float
    score_b: float
    p_value: float
    significant: bool
    winner: str | None


def _mean(values: list) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


class ABComparison:
    def __init__(self, significance_threshold: float = 0.05):
        self._threshold = significance_threshold

    def compare(
        self,
        model_a: str,
        scores_a: list,
        model_b: str,
        scores_b: list,
        metric: ComparisonMetric = ComparisonMetric.ACCURACY,
    ) -> ModelComparison:
        mean_a = _mean(scores_a)
        mean_b = _mean(scores_b)
        std_a = _std(scores_a)
        std_b = _std(scores_b)
        n = max(len(scores_a), 1)
        t = (mean_a - mean_b) / math.sqrt((std_a**2 / n + std_b**2 / n) + 1e-8)
        p_value = 1.0 / (1.0 + abs(t))
        significant = p_value < self._threshold
        if significant and mean_a > mean_b:
            winner = model_a
        elif significant and mean_b > mean_a:
            winner = model_b
        else:
            winner = None
        return ModelComparison(
            model_a=model_a,
            model_b=model_b,
            metric=metric,
            score_a=mean_a,
            score_b=mean_b,
            p_value=p_value,
            significant=significant,
            winner=winner,
        )

    def effect_size_cohens_d(self, scores_a: list, scores_b: list) -> float:
        mean_a = _mean(scores_a)
        mean_b = _mean(scores_b)
        std_a = _std(scores_a)
        std_b = _std(scores_b)
        pooled_variance = (std_a**2 + std_b**2) / 2.0
        pooled_std = math.sqrt(pooled_variance)
        if pooled_std == 0.0:
            return 0.0
        return (mean_a - mean_b) / pooled_std

    def win_loss_matrix(self, models: list, scores: dict) -> dict:
        matrix = {a: {b: 0 for b in models} for a in models}
        for i, model_a in enumerate(models):
            for j, model_b in enumerate(models):
                if i == j:
                    matrix[model_a][model_b] = 0
                    continue
                sa = scores.get(model_a, [])
                sb = scores.get(model_b, [])
                count = sum(1 for idx in range(min(len(sa), len(sb))) if sa[idx] > sb[idx])
                matrix[model_a][model_b] = count
        return matrix
