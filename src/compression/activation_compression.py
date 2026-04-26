"""Activation compression: quantization, sparsification, checkpointing."""

from __future__ import annotations

from enum import StrEnum


class SparsificationMethod(StrEnum):
    TOPK = "topk"
    THRESHOLD = "threshold"
    RANDOM_DROPOUT = "random_dropout"


class ActivationCompressor:
    def __init__(self, sparsity_ratio: float = 0.5) -> None:
        self.default_sparsity = sparsity_ratio

    def topk_sparsify(self, activations: list[float], k: int) -> list[float]:
        if not activations:
            return []
        k = max(0, min(k, len(activations)))
        indexed = sorted(enumerate(activations), key=lambda x: abs(x[1]), reverse=True)
        keep_indices = set(idx for idx, _ in indexed[:k])
        return [v if i in keep_indices else 0.0 for i, v in enumerate(activations)]

    def threshold_sparsify(self, activations: list[float], threshold: float) -> list[float]:
        return [v if abs(v) >= threshold else 0.0 for v in activations]

    def quantize_fp8(self, values: list[float], max_val: float = 448.0) -> list[float]:
        step = 0.0625  # 1/16
        result = []
        for v in values:
            clamped = max(-max_val, min(max_val, v))
            quantized = round(clamped / step) * step
            result.append(quantized)
        return result

    def sparsity_ratio(self, activations: list[float]) -> float:
        if not activations:
            return 0.0
        zero_count = sum(1 for v in activations if abs(v) < 1e-8)
        return zero_count / len(activations)

    def compress(
        self,
        activations: list[float],
        method: SparsificationMethod = SparsificationMethod.TOPK,
        k: int = None,
        threshold: float = 0.5,
    ) -> list[float]:
        if method == SparsificationMethod.TOPK:
            if k is None:
                k = max(1, int(len(activations) * (1 - self.default_sparsity)))
            return self.topk_sparsify(activations, k)
        elif method == SparsificationMethod.THRESHOLD:
            return self.threshold_sparsify(activations, threshold)
        elif method == SparsificationMethod.RANDOM_DROPOUT:
            import random

            rng = random.Random(42)
            return [v if rng.random() > self.default_sparsity else 0.0 for v in activations]
        return list(activations)

    def memory_savings_estimate(self, original_size: int, sparsity: float) -> float:
        return (original_size * (1 - sparsity)) / original_size if original_size > 0 else 0.0


ACTIVATION_COMPRESSOR = ActivationCompressor()
