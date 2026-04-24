"""Knowledge distillation training support (pure Python)."""
from __future__ import annotations

import math
from dataclasses import dataclass


_EPS = 1e-10


@dataclass
class DistillConfig:
    temperature: float = 4.0
    alpha: float = 0.5
    hard_label_weight: float = 0.5
    soft_label_weight: float = 0.5


@dataclass(frozen=True)
class DistillResult:
    hard_loss: float
    soft_loss: float
    total_loss: float
    alpha: float


class DistillationTrainer:
    """Compute distillation losses combining hard and soft targets."""

    def __init__(self, config: DistillConfig | None = None) -> None:
        self.config = config or DistillConfig()

    def soft_targets(
        self, logits: list[float], temperature: float
    ) -> list[float]:
        """Temperature-scaled softmax."""
        if not logits:
            return []
        T = max(temperature, _EPS)
        scaled = [x / T for x in logits]
        m = max(scaled)
        exps = [math.exp(x - m) for x in scaled]
        s = sum(exps)
        if s <= 0.0:
            return [1.0 / len(logits)] * len(logits)
        return [e / s for e in exps]

    def kl_divergence(self, p: list[float], q: list[float]) -> float:
        """KL(p || q), safe against zeros."""
        if len(p) != len(q):
            raise ValueError("p and q length mismatch")
        total = 0.0
        for pi, qi in zip(p, q):
            if pi <= 0.0:
                continue
            qi_safe = max(qi, _EPS)
            total += pi * math.log(pi / qi_safe)
        return total

    def _softmax(self, logits: list[float]) -> list[float]:
        return self.soft_targets(logits, 1.0)

    def _cross_entropy(
        self, logits: list[float], label: int
    ) -> float:
        probs = self._softmax(logits)
        if label < 0 or label >= len(probs):
            raise ValueError("label out of range")
        return -math.log(max(probs[label], _EPS))

    def distill_step(
        self,
        student_logits: list[float],
        teacher_logits: list[float],
        hard_labels: list[int],
    ) -> DistillResult:
        T = self.config.temperature
        alpha = self.config.alpha

        teacher_soft = self.soft_targets(teacher_logits, T)
        student_soft = self.soft_targets(student_logits, T)
        soft_loss = (T * T) * self.kl_divergence(teacher_soft, student_soft)

        if hard_labels:
            hard_losses = [
                self._cross_entropy(student_logits, lbl) for lbl in hard_labels
            ]
            hard_loss = sum(hard_losses) / len(hard_losses)
        else:
            hard_loss = 0.0

        total = alpha * hard_loss + (1.0 - alpha) * soft_loss
        return DistillResult(
            hard_loss=hard_loss,
            soft_loss=soft_loss,
            total_loss=total,
            alpha=alpha,
        )

    def optimal_temperature(self, n_classes: int) -> float:
        if n_classes <= 0:
            return 2.0
        t = math.sqrt(float(n_classes))
        return max(2.0, min(10.0, t))


DISTILLATION_TRAINER_REGISTRY: dict[str, type[DistillationTrainer]] = {
    "default": DistillationTrainer,
}
