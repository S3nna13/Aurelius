"""Aurelius cross-validation: k-fold cross-validation utilities."""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class FoldResult:
    fold_idx: int
    train_size: int
    val_size: int
    score: float


@dataclass(frozen=True)
class CVResult:
    k: int
    fold_results: list[FoldResult]
    mean_score: float
    std_score: float


class CrossValidator:
    """Provides k-fold cross-validation splits and evaluation utilities."""

    def __init__(self, k: int = 5, shuffle: bool = True, seed: int = 42) -> None:
        self.k = k
        self.shuffle = shuffle
        self.seed = seed

    def split(self, n_samples: int) -> list[tuple[list[int], list[int]]]:
        """Return k (train_indices, val_indices) splits over range(n_samples).

        If shuffle=True, permute indices using random.Random(seed) before folding.
        Each fold uses 1/k of data as validation set.
        """
        indices = list(range(n_samples))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        folds: list[tuple[list[int], list[int]]] = []
        fold_size = n_samples // self.k

        for fold_idx in range(self.k):
            val_start = fold_idx * fold_size
            # Last fold absorbs any remainder
            if fold_idx == self.k - 1:
                val_end = n_samples
            else:
                val_end = val_start + fold_size

            val_indices = indices[val_start:val_end]
            train_indices = indices[:val_start] + indices[val_end:]
            folds.append((train_indices, val_indices))

        return folds

    def evaluate(
        self,
        data: list,
        score_fn: Callable[[list, list], float],
    ) -> CVResult:
        """Run k-fold cross-validation and return aggregated CVResult.

        score_fn(train_data, val_data) -> float is called once per fold.
        std_score uses population standard deviation (statistics.pstdev).
        """
        splits = self.split(len(data))
        fold_results: list[FoldResult] = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]
            score = score_fn(train_data, val_data)
            fold_results.append(
                FoldResult(
                    fold_idx=fold_idx,
                    train_size=len(train_data),
                    val_size=len(val_data),
                    score=score,
                )
            )

        scores = [fr.score for fr in fold_results]
        mean_score = statistics.mean(scores) if scores else 0.0
        std_score = statistics.pstdev(scores) if scores else 0.0

        return CVResult(
            k=self.k,
            fold_results=fold_results,
            mean_score=mean_score,
            std_score=std_score,
        )

    def stratified_split(
        self,
        labels: list[int],
        k: int | None = None,
    ) -> list[tuple[list[int], list[int]]]:
        """Return k (train_indices, val_indices) splits preserving class distribution.

        Each fold has proportional class counts. If k is None, uses self.k.
        If shuffle=True, shuffles indices within each class using self.seed.
        """
        num_folds = k if k is not None else self.k
        n_samples = len(labels)

        # Group indices by class label
        class_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            class_indices.setdefault(label, []).append(idx)

        if self.shuffle:
            rng = random.Random(self.seed)
            for cls_idx_list in class_indices.values():
                rng.shuffle(cls_idx_list)

        # Build per-fold val index lists by distributing each class round-robin
        fold_val_indices: list[list[int]] = [[] for _ in range(num_folds)]
        for cls_idx_list in class_indices.values():
            cls_len = len(cls_idx_list)
            fold_size = cls_len // num_folds
            remainder = cls_len % num_folds
            start = 0
            for fold_idx in range(num_folds):
                # Distribute remainder among the first `remainder` folds
                extra = 1 if fold_idx < remainder else 0
                end = start + fold_size + extra
                fold_val_indices[fold_idx].extend(cls_idx_list[start:end])
                start = end

        # Build final splits
        all_indices = set(range(n_samples))
        folds: list[tuple[list[int], list[int]]] = []
        for fold_idx in range(num_folds):
            val_set = set(fold_val_indices[fold_idx])
            train_indices = sorted(all_indices - val_set)
            val_indices = sorted(val_set)
            folds.append((train_indices, val_indices))

        return folds


CROSS_VALIDATOR_REGISTRY = {"default": CrossValidator}
