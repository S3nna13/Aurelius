"""Hybrid fusion v2: normalized score fusion, dynamic weight tuning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FusionStrategy(StrEnum):
    RRF = "rrf"
    LINEAR = "linear"
    CONVEX = "convex"


@dataclass
class FusionResult:
    doc_id: str
    fused_score: float
    sparse_score: float = 0.0
    dense_score: float = 0.0


def _normalize(scores: list[tuple[str, float]]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1]. Returns {doc_id: normalized_score}."""
    if not scores:
        return {}
    values = [s for _, s in scores]
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return {doc_id: 1.0 for doc_id, _ in scores}
    return {doc_id: (s - min_v) / (max_v - min_v) for doc_id, s in scores}


class HybridFusionV2:
    def __init__(
        self,
        strategy: FusionStrategy = FusionStrategy.CONVEX,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self.strategy = strategy
        self.alpha = alpha
        self.rrf_k = rrf_k

    def fuse(
        self,
        sparse_results: list[tuple[str, float]],
        dense_results: list[tuple[str, float]],
    ) -> list[FusionResult]:
        # Collect all doc ids
        all_ids: set[str] = set()
        for doc_id, _ in sparse_results:
            all_ids.add(doc_id)
        for doc_id, _ in dense_results:
            all_ids.add(doc_id)

        sparse_norm = _normalize(sparse_results)
        dense_norm = _normalize(dense_results)

        # Build rank maps (1-based positions) for RRF
        sparse_rank: dict[str, int] = {
            doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sparse_results)
        }
        dense_rank: dict[str, int] = {
            doc_id: rank + 1 for rank, (doc_id, _) in enumerate(dense_results)
        }

        results: list[FusionResult] = []
        k = self.rrf_k
        alpha = max(0.0, min(1.0, self.alpha))

        for doc_id in all_ids:
            sp = sparse_norm.get(doc_id, 0.0)
            dn = dense_norm.get(doc_id, 0.0)

            if self.strategy == FusionStrategy.RRF:
                rrf_sparse = 1.0 / (k + sparse_rank[doc_id]) if doc_id in sparse_rank else 0.0
                rrf_dense = 1.0 / (k + dense_rank[doc_id]) if doc_id in dense_rank else 0.0
                fused = rrf_sparse + rrf_dense
            elif self.strategy == FusionStrategy.LINEAR:
                fused = self.alpha * sp + (1.0 - self.alpha) * dn
            else:  # CONVEX
                fused = alpha * sp + (1.0 - alpha) * dn

            results.append(
                FusionResult(
                    doc_id=doc_id,
                    fused_score=fused,
                    sparse_score=sp,
                    dense_score=dn,
                )
            )

        results.sort(key=lambda r: r.fused_score, reverse=True)
        return results

    def tune_alpha(
        self,
        relevant_ids: set[str],
        results_at_alpha: dict[float, list[FusionResult]],
    ) -> float:
        """Return alpha with best precision@5 (fraction of top-5 that are relevant)."""
        best_alpha = 0.5
        best_precision = -1.0

        for alpha, results in results_at_alpha.items():
            top5 = [r.doc_id for r in results[:5]]
            precision = sum(1 for d in top5 if d in relevant_ids) / max(1, len(top5))
            if precision > best_precision:
                best_precision = precision
                best_alpha = alpha

        return best_alpha


HYBRID_FUSION_REGISTRY: dict[str, HybridFusionV2] = {
    "rrf": HybridFusionV2(FusionStrategy.RRF),
    "linear": HybridFusionV2(FusionStrategy.LINEAR),
    "convex": HybridFusionV2(FusionStrategy.CONVEX),
}
