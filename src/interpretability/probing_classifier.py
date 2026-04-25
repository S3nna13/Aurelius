"""Probing classifier: linear probe, logistic regression, probing accuracy."""

import math
from dataclasses import dataclass
from enum import Enum


class ProbeTask(str, Enum):
    POS_TAGGING = "pos_tagging"
    SENTIMENT = "sentiment"
    SYNTAX_DEPTH = "syntax_depth"
    ENTITY_TYPE = "entity_type"
    COREFERENCE = "coreference"


@dataclass
class ProbeResult:
    task: ProbeTask
    layer: int
    accuracy: float
    n_samples: int
    chance_level: float


class LinearProbe:
    def __init__(self, n_features: int, n_classes: int) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        # weights shape: (n_classes, n_features), bias shape: (n_classes,)
        self.weights: list[list[float]] = [[0.0] * n_features for _ in range(n_classes)]
        self.bias: list[float] = [0.0] * n_classes

    def _softmax(self, logits: list[float]) -> list[float]:
        max_l = max(logits)
        exps = [math.exp(l - max_l) for l in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def _logits(self, x: list[float]) -> list[float]:
        result = []
        for c in range(self.n_classes):
            val = self.bias[c]
            for j in range(self.n_features):
                val += self.weights[c][j] * x[j]
            result.append(val)
        return result

    def fit(
        self,
        X: list[list[float]],
        y: list[int],
        lr: float = 0.01,
        n_epochs: int = 100,
    ) -> None:
        n = len(X)
        for _ in range(n_epochs):
            # accumulate gradients
            dw = [[0.0] * self.n_features for _ in range(self.n_classes)]
            db = [0.0] * self.n_classes
            for i in range(n):
                logits = self._logits(X[i])
                probs = self._softmax(logits)
                for c in range(self.n_classes):
                    grad = probs[c] - (1.0 if y[i] == c else 0.0)
                    db[c] += grad
                    for j in range(self.n_features):
                        dw[c][j] += grad * X[i][j]
            # update
            for c in range(self.n_classes):
                self.bias[c] -= lr * db[c] / n
                for j in range(self.n_features):
                    self.weights[c][j] -= lr * dw[c][j] / n

    def predict(self, X: list[list[float]]) -> list[int]:
        preds = []
        for x in X:
            logits = self._logits(x)
            preds.append(logits.index(max(logits)))
        return preds

    def accuracy(self, X: list[list[float]], y: list[int]) -> float:
        preds = self.predict(X)
        if not y:
            return 0.0
        correct = sum(p == t for p, t in zip(preds, y))
        return correct / len(y)


class ProbingClassifier:
    def __init__(self) -> None:
        pass

    def probe_layer(
        self,
        task: ProbeTask,
        layer_idx: int,
        activations: list[list[float]],
        labels: list[int],
    ) -> ProbeResult:
        n_features = len(activations[0]) if activations else 1
        unique_labels = set(labels)
        n_classes = len(unique_labels) if unique_labels else 1
        probe = LinearProbe(n_features, n_classes)
        probe.fit(activations, labels)
        acc = probe.accuracy(activations, labels)
        chance = 1.0 / n_classes
        return ProbeResult(
            task=task,
            layer=layer_idx,
            accuracy=acc,
            n_samples=len(labels),
            chance_level=chance,
        )

    def probe_all_layers(
        self,
        task: ProbeTask,
        layer_activations: list[list[list[float]]],
        labels: list[int],
    ) -> list[ProbeResult]:
        results = []
        for idx, acts in enumerate(layer_activations):
            results.append(self.probe_layer(task, idx, acts, labels))
        return results

    def best_layer(self, results: list[ProbeResult]) -> "ProbeResult | None":
        if not results:
            return None
        return max(results, key=lambda r: r.accuracy)
