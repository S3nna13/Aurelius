"""Best-of-N inference-time reranker.

Reference:
    Stiennon et al. 2020, "Learning to summarize from human feedback".
    Cobbe et al. 2021, "Training Verifiers to Solve Math Word Problems".

Distinct from :mod:`src.alignment.bond` (BOND is training-time distillation of
Best-of-N; this module does inference-time reranking only).

Given a prompt, sample ``N`` candidate responses from a ``generate_fn``, score
each with a ``reward_fn``, and return either the top-scoring candidate or an
aggregated answer (e.g. majority answer weighted by reward). Ties are broken
deterministically by generation index.

Pure stdlib.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, List, Optional

__all__ = ["BoNCandidate", "BestOfNReranker"]

_LOGGER = logging.getLogger(__name__)

_VALID_AGGREGATIONS = ("max", "weighted_vote", "avg_reward")


@dataclass
class BoNCandidate:
    """A single Best-of-N candidate response with its reward and rank.

    Attributes:
        response: The generated response string.
        reward: Scalar reward assigned by ``reward_fn``. ``-inf`` if scoring
            failed.
        rank: 0-based rank after sorting by reward descending (0 is best).
    """

    response: str
    reward: float
    rank: int


class BestOfNReranker:
    """Inference-time Best-of-N reranker.

    Args:
        generate_fn: Callable ``prompt -> response``. Called ``n`` times per
            rerank. Exceptions are logged and the candidate is skipped.
        reward_fn: Callable ``(prompt, response) -> float``. Exceptions cause
            the candidate to receive ``-inf`` reward and to be ranked last.
        n: Number of candidates to sample. Must be ``>= 1``.
        aggregation: One of ``"max"``, ``"weighted_vote"``, ``"avg_reward"``.
            Used as the default strategy for methods that consult it. Unknown
            values raise :class:`ValueError` eagerly at construction time.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        reward_fn: Callable[[str, str], float],
        n: int = 8,
        aggregation: str = "max",
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {_VALID_AGGREGATIONS}, "
                f"got {aggregation!r}"
            )
        self.generate_fn = generate_fn
        self.reward_fn = reward_fn
        self.n = n
        self.aggregation = aggregation

    # ------------------------------------------------------------------ core

    def _sample_and_score(self, prompt: str) -> List[BoNCandidate]:
        """Sample ``n`` responses and score each. Preserves generation index
        for deterministic tie-breaking.

        Returns a list of ``BoNCandidate`` with a provisional ``rank`` equal to
        their generation index; callers are expected to re-rank.
        """
        raw: List[tuple[int, str, float]] = []
        for i in range(self.n):
            try:
                response = self.generate_fn(prompt)
            except Exception as exc:  # noqa: BLE001 - spec: skip + log
                _LOGGER.warning(
                    "generate_fn raised on candidate %d: %r; skipping", i, exc
                )
                continue
            try:
                reward = float(self.reward_fn(prompt, response))
                if math.isnan(reward):
                    # NaN cannot be ordered sensibly; treat as failure.
                    raise ValueError("reward_fn returned NaN")
            except Exception as exc:  # noqa: BLE001 - spec: -inf + log
                _LOGGER.warning(
                    "reward_fn raised on candidate %d: %r; assigning -inf",
                    i,
                    exc,
                )
                reward = float("-inf")
            raw.append((i, response, reward))
        # Sort by reward desc, then by generation index asc (deterministic).
        # For the reward key we use ``_reward_sort_key`` which returns a
        # float that, in ascending order, places higher rewards first and
        # ``-inf`` last.
        raw.sort(key=lambda t: (_reward_sort_key(t[2]), t[0]))
        ranked = [
            BoNCandidate(response=resp, reward=rew, rank=rank)
            for rank, (_, resp, rew) in enumerate(raw)
        ]
        return ranked

    # ------------------------------------------------------------------ api

    def rerank(self, prompt: str) -> List[BoNCandidate]:
        """Return all ``n`` candidates sorted by reward descending.

        Candidates whose ``generate_fn`` raised are omitted. Candidates whose
        ``reward_fn`` raised are included with ``reward=-inf`` and ranked last.
        Ties are broken by generation index (ascending).
        """
        return self._sample_and_score(prompt)

    def best(self, prompt: str) -> BoNCandidate:
        """Return the single top-ranked candidate (rank 0).

        Raises:
            RuntimeError: if every ``generate_fn`` call failed and no candidate
                was produced.
        """
        ranked = self._sample_and_score(prompt)
        if not ranked:
            raise RuntimeError(
                "BestOfNReranker.best: all generate_fn calls failed; "
                "no candidates available."
            )
        return ranked[0]

    def weighted_vote(
        self,
        prompt: str,
        answer_extractor: Callable[[str], str],
    ) -> str:
        """Return the answer with the greatest summed reward across candidates.

        Each candidate's response is mapped to an answer string by
        ``answer_extractor``; answers are grouped and each group's total
        reward is summed. The answer with the highest sum wins. Ties are
        broken by the best (lowest) generation index contributing to the
        group.

        ``-inf`` rewards are excluded from the sum (they represent failed
        scoring and should not dominate by comparison).
        """
        # Need the raw sampling order for deterministic tie-breaking by
        # generation index, so sample directly here.
        raw: List[tuple[int, str, float]] = []
        for i in range(self.n):
            try:
                response = self.generate_fn(prompt)
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "generate_fn raised on candidate %d: %r; skipping", i, exc
                )
                continue
            try:
                reward = float(self.reward_fn(prompt, response))
                if math.isnan(reward):
                    raise ValueError("reward_fn returned NaN")
            except Exception as exc:  # noqa: BLE001
                _LOGGER.warning(
                    "reward_fn raised on candidate %d: %r; assigning -inf",
                    i,
                    exc,
                )
                reward = float("-inf")
            raw.append((i, response, reward))

        if not raw:
            raise RuntimeError(
                "BestOfNReranker.weighted_vote: all generate_fn calls failed."
            )

        # Group by extracted answer.
        sums: dict[str, float] = {}
        first_idx: dict[str, int] = {}
        best_reward: dict[str, float] = {}
        for idx, response, reward in raw:
            answer = answer_extractor(response)
            effective = reward if reward != float("-inf") else 0.0
            if answer not in sums:
                sums[answer] = effective
                first_idx[answer] = idx
                best_reward[answer] = reward
            else:
                sums[answer] += effective
                if idx < first_idx[answer]:
                    first_idx[answer] = idx
                if reward > best_reward[answer]:
                    best_reward[answer] = reward

        # Choose: highest summed reward, then highest single candidate reward,
        # then lowest first generation index. The intermediate tie-break by
        # `best_reward` ensures that when every candidate is unique (each
        # answer has sum == its single reward), the winner matches `best()`.
        winner = min(
            sums.keys(),
            key=lambda a: (-sums[a], -best_reward[a], first_idx[a]),
        )
        return winner


def _reward_sort_key(reward: float) -> float:
    """Key for ascending sort that puts higher rewards first, ``-inf`` last.

    Mapping: finite ``r`` -> ``-r``; ``-inf`` -> ``+inf`` (sorts last);
    ``+inf`` -> ``-inf`` (sorts first).
    """
    if reward == float("-inf"):
        return float("inf")
    if reward == float("inf"):
        return float("-inf")
    return -reward
