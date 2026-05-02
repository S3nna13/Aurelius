"""Multi-sample voting / self-consistency aggregation.

References:
    Wang et al., 2022, "Self-Consistency Improves Chain of Thought Reasoning in
    Language Models," arXiv:2203.11171.
    Chen et al., 2024, "Universal Self-Consistency for Large Language Model
    Generation."

Given N sampled completions, aggregate them via:
    - "majority" vote: pick the extracted answer with the most occurrences.
    - "weighted" vote: pick the extracted answer with the largest total weight.
    - "usc"        : Universal Self-Consistency. Pick the sample with the
                     highest mean similarity to all other samples (the medoid).

Pure stdlib only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class VoteResult:
    """Outcome of a multi-sample vote.

    Attributes:
        selected   : The chosen sample (full text for USC; for majority/weighted
                     it is the sample whose extracted answer won).
        votes      : Mapping of extracted-answer (or sample text for USC) to the
                     accumulated vote mass.
        strategy   : One of ``"majority"``, ``"weighted"``, ``"usc"``.
        confidence : Proportion of the total mass assigned to the selected
                     answer. In [0.0, 1.0].
    """

    selected: str
    votes: dict[str, float] = field(default_factory=dict)
    strategy: str = "majority"
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Default helpers
# ---------------------------------------------------------------------------
def _identity_extractor(text: str) -> str:
    """Default answer extractor: the identity function."""

    return text


def _char_ngrams(text: str, n: int = 3) -> set:
    """Character n-grams (as a set) for a string."""

    if len(text) <= n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _jaccard(a: str, b: str) -> float:
    """Character-trigram Jaccard similarity in [0, 1]."""

    if a == b:
        return 1.0
    ga = _char_ngrams(a)
    gb = _char_ngrams(b)
    if not ga and not gb:
        return 1.0
    inter = len(ga & gb)
    union = len(ga | gb)
    if union == 0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# Voter
# ---------------------------------------------------------------------------
class MultiSampleVoter:
    """Aggregate N sampled completions into a single answer.

    Args:
        answer_extractor : Callable mapping a raw sample to an "answer" key
                           (e.g. extract last integer for math). Defaults to
                           identity.
        strategy         : ``"majority"``, ``"weighted"``, or ``"usc"``.
        similarity_fn    : Callable (a, b) -> float in [0, 1]. Used only for
                           USC. Defaults to character-trigram Jaccard.
    """

    _VALID_STRATEGIES = ("majority", "weighted", "usc")

    def __init__(
        self,
        answer_extractor: Callable[[str], str] | None = None,
        strategy: str = "majority",
        similarity_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}; expected one of {self._VALID_STRATEGIES}."
            )
        self.answer_extractor = answer_extractor or _identity_extractor
        self.strategy = strategy
        self.similarity_fn = similarity_fn or _jaccard

    # ------------------------------------------------------------------
    def vote(
        self,
        samples: list[str],
        weights: list[float] | None = None,
    ) -> VoteResult:
        """Aggregate ``samples`` to a single answer.

        Args:
            samples : list of raw completion strings (N >= 1).
            weights : optional per-sample weights (e.g. log-probs exponentiated).
                      Length must match ``samples``. Non-negative.

        Returns:
            ``VoteResult``.
        """

        if not samples:
            raise ValueError("samples must be a non-empty list.")

        if weights is not None:
            if len(weights) != len(samples):
                raise ValueError(f"weights length {len(weights)} != samples length {len(samples)}.")
            for w in weights:
                if w < 0:
                    raise ValueError(f"weights must be non-negative; got {w}.")

        # Single-sample fast path --------------------------------------------------
        if len(samples) == 1:
            only = samples[0]
            if self.strategy == "usc":
                return VoteResult(
                    selected=only,
                    votes={only: 1.0},
                    strategy="usc",
                    confidence=1.0,
                )
            key = self.answer_extractor(only)
            return VoteResult(
                selected=key,
                votes={key: 1.0},
                strategy=self.strategy,
                confidence=1.0,
            )

        if self.strategy == "usc":
            return self._vote_usc(samples)
        if self.strategy == "weighted":
            return self._vote_weighted(samples, weights)
        return self._vote_majority(samples, weights)

    # ------------------------------------------------------------------
    def _vote_majority(
        self,
        samples: list[str],
        weights: list[float] | None,
    ) -> VoteResult:
        """Unit-weight majority vote over extracted answers. Ties: first-seen."""

        tallies: dict[str, float] = {}
        first_seen: dict[str, int] = {}
        for idx, s in enumerate(samples):
            key = self.answer_extractor(s)
            if key not in tallies:
                tallies[key] = 0.0
                first_seen[key] = idx
            tallies[key] += 1.0

        total = sum(tallies.values())
        # Sort: higher tally first, earlier first-seen index breaks ties.
        best_key = min(
            tallies.keys(),
            key=lambda k: (-tallies[k], first_seen[k]),
        )
        confidence = tallies[best_key] / total if total > 0 else 0.0
        return VoteResult(
            selected=best_key,
            votes=tallies,
            strategy="majority",
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    def _vote_weighted(
        self,
        samples: list[str],
        weights: list[float] | None,
    ) -> VoteResult:
        """Weighted vote. If weights is None, falls back to unit weights."""

        w = weights if weights is not None else [1.0] * len(samples)
        tallies: dict[str, float] = {}
        first_seen: dict[str, int] = {}
        for idx, (s, wi) in enumerate(zip(samples, w)):
            key = self.answer_extractor(s)
            if key not in tallies:
                tallies[key] = 0.0
                first_seen[key] = idx
            tallies[key] += float(wi)

        total = sum(tallies.values())
        best_key = min(
            tallies.keys(),
            key=lambda k: (-tallies[k], first_seen[k]),
        )
        confidence = tallies[best_key] / total if total > 0 else 0.0
        return VoteResult(
            selected=best_key,
            votes=tallies,
            strategy="weighted",
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    def _vote_usc(self, samples: list[str]) -> VoteResult:
        """Universal Self-Consistency: pick the medoid sample.

        For each sample i, compute mean_j sim(i, j) for j != i. Pick the
        sample with the largest mean similarity. Ties: first-seen.
        """

        n = len(samples)
        mean_sims: list[float] = []
        for i in range(n):
            total = 0.0
            count = 0
            for j in range(n):
                if i == j:
                    continue
                total += float(self.similarity_fn(samples[i], samples[j]))
                count += 1
            mean_sims.append(total / count if count else 0.0)

        # Pick argmax with first-seen tiebreak.
        best_idx = 0
        best_val = mean_sims[0]
        for i in range(1, n):
            if mean_sims[i] > best_val:
                best_val = mean_sims[i]
                best_idx = i

        # Votes dict: index-keyed scores. Key by sample text; if duplicates,
        # accumulate.
        votes: dict[str, float] = {}
        for s, m in zip(samples, mean_sims):
            votes[s] = votes.get(s, 0.0) + m
        total_mass = sum(votes.values())
        selected_text = samples[best_idx]
        confidence = votes[selected_text] / total_mass if total_mass > 0 else 0.0
        return VoteResult(
            selected=selected_text,
            votes=votes,
            strategy="usc",
            confidence=confidence,
        )


__all__ = ["VoteResult", "MultiSampleVoter"]
