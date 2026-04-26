"""Aurelius — Multi-annotator Preference Aggregation (CoVal-inspired).

Converts conflicting human rankings (e.g. "B>A>C=D") into a single DPO
training signal using Borda count, Bradley-Terry MLE, or majority vote.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AggregationConfig:
    min_confidence: float = 0.6  # minimum majority fraction to include pair
    min_annotators: int = 3  # minimum annotators required per prompt
    tie_threshold: float = 0.05  # Borda score gap below which = tie
    aggregation_method: str = "borda"  # "borda" | "majority" | "bradley_terry"


# ---------------------------------------------------------------------------
# Ranking string parser
# ---------------------------------------------------------------------------


def parse_ranking(ranking_str: str) -> list[list[str]]:
    """Parse "B>A>C=D" → [["B"], ["A"], ["C", "D"]].

    Each inner list is a group of tied candidates at that rank position.
    Empty string → []. Single item → [["A"]].
    """
    ranking_str = ranking_str.strip()
    if not ranking_str:
        return []

    groups: list[list[str]] = []
    # Split by '>' to get rank tiers
    for tier in ranking_str.split(">"):
        # Within a tier, candidates are separated by '='
        candidates = [c.strip() for c in tier.split("=") if c.strip()]
        if candidates:
            groups.append(candidates)
    return groups


def ranking_to_pairwise(ranking: list[list[str]]) -> list[tuple[str, str, float]]:
    """Convert ranking groups to (winner, loser, weight) triples.

    Returns only (winner, loser, 1.0) pairs — ties (same group) are excluded.
    """
    pairs: list[tuple[str, str, float]] = []
    for i, higher_group in enumerate(ranking):
        for lower_group in ranking[i + 1 :]:
            for winner in higher_group:
                for loser in lower_group:
                    pairs.append((winner, loser, 1.0))
    return pairs


# ---------------------------------------------------------------------------
# Borda Count
# ---------------------------------------------------------------------------


def borda_scores(
    rankings: list[list[list[str]]],
    candidates: list[str],
) -> dict[str, float]:
    """Compute normalized Borda count scores across multiple annotator rankings.

    Position rank_pos gets (n_candidates - rank_pos) points where rank_pos
    starts at 0. Tied candidates share the average of their positions' points.
    Missing candidates get 0 points. Scores are divided by max possible score.
    """
    n = len(candidates)
    if n == 0:
        return {}

    cumulative: dict[str, float] = {c: 0.0 for c in candidates}

    for ranking in rankings:
        # Expand groups to assign positions
        pos = 0
        for group in ranking:
            group_size = len(group)
            # Positions this group occupies: pos, pos+1, ..., pos+group_size-1
            # Points for each position: (n - 1 - pos_index)
            points_list = [n - 1 - (pos + k) for k in range(group_size)]
            avg_points = sum(points_list) / group_size
            for candidate in group:
                if candidate in cumulative:
                    cumulative[candidate] += avg_points
            pos += group_size

    # Normalize by maximum possible score = (n - 1) * len(rankings)
    max_score = (n - 1) * len(rankings) if rankings else 1.0
    if max_score == 0:
        return {c: 0.0 for c in candidates}

    return {c: cumulative[c] / max_score for c in candidates}


# ---------------------------------------------------------------------------
# Bradley-Terry Model
# ---------------------------------------------------------------------------


def bradley_terry_scores(
    pairwise_wins: dict[tuple[str, str], int],
    candidates: list[str],
    n_iter: int = 100,
    lr: float = 0.1,
) -> dict[str, float]:
    """Iterative Bradley-Terry MLE.

    p(i beats j) = s_i / (s_i + s_j)
    Update: s_i ∝ wins_i / sum_j(total_ij / (s_i + s_j))
    Returns normalized strengths (sum to 1).
    """
    n = len(candidates)
    if n == 0:
        return {}

    idx = {c: i for i, c in enumerate(candidates)}

    # Initialize strengths uniformly
    strengths = [1.0] * n

    # Build wins and totals matrices
    wins = [[0.0] * n for _ in range(n)]
    totals = [[0.0] * n for _ in range(n)]
    for (winner, loser), count in pairwise_wins.items():
        if winner in idx and loser in idx:
            i, j = idx[winner], idx[loser]
            wins[i][j] += count
            totals[i][j] += count
            totals[j][i] += count

    for _ in range(n_iter):
        new_strengths = []
        for i in range(n):
            numerator = sum(wins[i][j] for j in range(n) if j != i)
            denominator = sum(
                totals[i][j] / (strengths[i] + strengths[j])
                for j in range(n)
                if j != i and totals[i][j] > 0
            )
            if denominator == 0:
                new_strengths.append(strengths[i])
            else:
                new_strengths.append(numerator / denominator if numerator > 0 else 1e-8)
        # Normalize to prevent drift
        total = sum(new_strengths)
        if total > 0:
            strengths = [s / total for s in new_strengths]
        else:
            strengths = [1.0 / n] * n

    total = sum(strengths)
    if total == 0:
        return {c: 1.0 / n for c in candidates}
    return {candidates[i]: strengths[i] / total for i in range(n)}


# ---------------------------------------------------------------------------
# Majority Vote
# ---------------------------------------------------------------------------


def majority_vote_scores(
    rankings: list[list[list[str]]],
    candidates: list[str],
) -> dict[str, float]:
    """For each pair (i, j): count how many annotators rank i above j.

    Score(i) = mean(pairwise win fractions against all others).
    Returns dict of win rates in [0, 1].
    """
    n = len(candidates)
    if n == 0:
        return {}

    idx = {c: i for i, c in enumerate(candidates)}

    # wins[i][j] = number of annotators who rank candidate i above candidate j
    wins = [[0] * n for _ in range(n)]
    comparisons = [[0] * n for _ in range(n)]

    for ranking in rankings:
        # Build rank position map for this annotator
        rank_pos: dict[str, int] = {}
        pos = 0
        for group in ranking:
            for candidate in group:
                rank_pos[candidate] = pos
            pos += len(group)

        for ci in candidates:
            for cj in candidates:
                if ci == cj:
                    continue
                i, j = idx[ci], idx[cj]
                if ci in rank_pos and cj in rank_pos:
                    comparisons[i][j] += 1
                    if rank_pos[ci] < rank_pos[cj]:
                        wins[i][j] += 1

    scores: dict[str, float] = {}
    for ci in candidates:
        i = idx[ci]
        win_rates = []
        for cj in candidates:
            if ci == cj:
                continue
            j = idx[cj]
            if comparisons[i][j] > 0:
                win_rates.append(wins[i][j] / comparisons[i][j])
        scores[ci] = sum(win_rates) / len(win_rates) if win_rates else 0.0

    return scores


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class PreferenceAggregator:
    def __init__(self, cfg: AggregationConfig) -> None:
        self.cfg = cfg

    def aggregate(
        self,
        candidates: list[str],
        rankings: list[list[list[str]]],
    ) -> dict[str, float]:
        """Dispatch to configured method and return candidate → score dict."""
        method = self.cfg.aggregation_method

        if method == "borda":
            return borda_scores(rankings, candidates)
        elif method == "majority":
            return majority_vote_scores(rankings, candidates)
        elif method == "bradley_terry":
            # Build pairwise wins from all rankings
            pairwise_wins: dict[tuple[str, str], int] = {}
            for ranking in rankings:
                for winner, loser, _ in ranking_to_pairwise(ranking):
                    key = (winner, loser)
                    pairwise_wins[key] = pairwise_wins.get(key, 0) + 1
            return bradley_terry_scores(pairwise_wins, candidates)
        else:
            raise ValueError(f"Unknown aggregation_method: {method!r}")

    def get_best_pair(
        self,
        candidates: list[str],
        scores: dict[str, float],
        confidence: float | None = None,
    ) -> tuple[str, str] | None:
        """Return (best_candidate, worst_candidate) if score gap > tie_threshold.

        Returns None if all candidates are within tie_threshold of each other.
        """
        if not candidates or len(candidates) < 2:
            return None

        # Filter to candidates that have scores
        scored = [(c, scores.get(c, 0.0)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)

        best = scored[0]
        worst = scored[-1]

        if abs(best[1] - worst[1]) <= self.cfg.tie_threshold:
            return None

        return (best[0], worst[0])

    def to_dpo_pairs(
        self,
        responses: dict[str, str],
        rankings: list[list[list[str]]],
    ) -> list[tuple[str, str]]:
        """Aggregate rankings, find best vs worst, return [(chosen_text, rejected_text)].

        Returns [] if not enough confidence or too few annotators.
        """
        if len(rankings) < self.cfg.min_annotators:
            return []

        candidates = list(responses.keys())
        if len(candidates) < 2:
            return []

        scores = self.aggregate(candidates, rankings)
        pair = self.get_best_pair(candidates, scores)

        if pair is None:
            return []

        best_key, worst_key = pair
        chosen = responses.get(best_key)
        rejected = responses.get(worst_key)

        if chosen is None or rejected is None:
            return []

        return [(chosen, rejected)]


# ---------------------------------------------------------------------------
# Confidence metrics
# ---------------------------------------------------------------------------


def annotator_agreement(
    rankings: list[list[list[str]]],
    candidates: list[str],
) -> float:
    """Compute Kendall's W (coefficient of concordance) across annotators.

    W = 0 means no agreement, W = 1 means perfect agreement.
    Returns float in [0, 1].
    """
    k = len(rankings)  # number of annotators
    n = len(candidates)  # number of candidates

    if k == 0 or n == 0:
        return 0.0
    if k == 1 or n == 1:
        return 1.0

    idx = {c: i for i, c in enumerate(candidates)}

    # Build rank matrix: R[annotator][candidate_index] = rank (1-based)
    # Tied items get average rank
    R: list[list[float]] = []
    for ranking in rankings:
        rank_row = [0.0] * n
        pos = 1  # 1-based rank position
        for group in ranking:
            group_size = len(group)
            # Average rank for tied items
            avg_rank = (pos + pos + group_size - 1) / 2.0
            for candidate in group:
                if candidate in idx:
                    rank_row[idx[candidate]] = avg_rank
            pos += group_size

        # Candidates not mentioned get rank n+1 (worst)
        for i in range(n):
            if rank_row[i] == 0.0:
                rank_row[i] = float(n + 1)

        R.append(rank_row)

    # Compute column sums Rj = sum over annotators of rank of candidate j
    col_sums = [sum(R[i][j] for i in range(k)) for j in range(n)]

    # Mean of column sums
    mean_col_sum = sum(col_sums) / n

    # S = sum of squared deviations from mean
    S = sum((rj - mean_col_sum) ** 2 for rj in col_sums)

    # Kendall's W = 12 * S / (k^2 * (n^3 - n))
    denominator = k * k * (n**3 - n)
    if denominator == 0:
        return 1.0

    W = 12.0 * S / denominator

    # Clamp to [0, 1] due to floating point
    return max(0.0, min(1.0, W))
