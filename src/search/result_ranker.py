"""Result ranker: RRF fusion, score normalization, deduplication."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RankedResult:
    result_id: str
    score: float
    rank: int
    source: str = ""


class ResultRanker:
    """Ranks and fuses result lists using Reciprocal Rank Fusion and utilities."""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k

    def reciprocal_rank_fusion(
        self, rankings: list[list[str]]
    ) -> list[tuple[str, float]]:
        """Fuse multiple ranked lists using RRF. Returns sorted (doc_id, score) desc."""
        if not rankings:
            return []

        scores: dict[str, float] = {}
        for ranked_list in rankings:
            for rank_i, doc_id in enumerate(ranked_list, start=1):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank_i)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    def normalize_scores(self, results: list[RankedResult]) -> list[RankedResult]:
        """Min-max normalize scores to [0, 1]. If all equal, set all to 1.0."""
        if not results:
            return results

        min_score = min(r.score for r in results)
        max_score = max(r.score for r in results)

        if max_score == min_score:
            return [
                RankedResult(
                    result_id=r.result_id, score=1.0, rank=r.rank, source=r.source
                )
                for r in results
            ]

        spread = max_score - min_score
        return [
            RankedResult(
                result_id=r.result_id,
                score=(r.score - min_score) / spread,
                rank=r.rank,
                source=r.source,
            )
            for r in results
        ]

    def deduplicate(
        self, results: list[RankedResult], key: str = "result_id"
    ) -> list[RankedResult]:
        """Keep first occurrence of each unique result_id (stable order)."""
        seen: set[str] = set()
        out: list[RankedResult] = []
        for r in results:
            rid = getattr(r, key)
            if rid not in seen:
                seen.add(rid)
                out.append(r)
        return out

    def to_ranked(
        self, items: list[str], scores: list[float], source: str = ""
    ) -> list[RankedResult]:
        """Zip items and scores into RankedResult list with rank=index+1."""
        return [
            RankedResult(result_id=item, score=score, rank=idx + 1, source=source)
            for idx, (item, score) in enumerate(zip(items, scores))
        ]


RESULT_RANKER = ResultRanker()
