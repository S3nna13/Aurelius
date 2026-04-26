"""Beam search reasoner over step-level reasoning chains.

Inspired by Tree-of-Thought beam variant (Yao et al. 2305.10601) and
Self-Consistency (Wang et al. 2203.11171); Aurelius-native. License: MIT.
"""

from __future__ import annotations

from dataclasses import dataclass

_MAX_BEAM_WIDTH = 64
_MAX_DEPTH = 128
_MAX_STEP_LEN = 8192


@dataclass
class BeamHypothesis:
    steps: list[str]
    score: float = 0.0
    is_finished: bool = False

    def __post_init__(self) -> None:
        for s in self.steps:
            if len(s) > _MAX_STEP_LEN:
                raise ValueError(f"step exceeds {_MAX_STEP_LEN} chars")

    @property
    def depth(self) -> int:
        return len(self.steps)

    def extend(self, new_step: str, step_score: float) -> BeamHypothesis:
        """Return a new hypothesis with new_step appended."""
        if len(new_step) > _MAX_STEP_LEN:
            raise ValueError(f"step exceeds {_MAX_STEP_LEN} chars")
        return BeamHypothesis(
            steps=self.steps + [new_step],
            score=self.score + step_score,
            is_finished=self.is_finished,
        )

    def normalized_score(self) -> float:
        """Length-normalized score (prevents bias toward longer hypotheses)."""
        return self.score / max(len(self.steps), 1)


class BeamSearchReasoner:
    """Beam search over reasoning step sequences."""

    def __init__(
        self,
        beam_width: int = 4,
        max_depth: int = 8,
        length_penalty: float = 0.6,
        normalize_scores: bool = True,
    ) -> None:
        if beam_width < 1 or beam_width > _MAX_BEAM_WIDTH:
            raise ValueError(f"beam_width must be in [1, {_MAX_BEAM_WIDTH}]")
        if max_depth < 1 or max_depth > _MAX_DEPTH:
            raise ValueError(f"max_depth must be in [1, {_MAX_DEPTH}]")
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.length_penalty = length_penalty
        self.normalize_scores = normalize_scores

    def initialize(self, initial_step: str, initial_score: float = 0.0) -> list[BeamHypothesis]:
        """Create initial beam with one hypothesis."""
        return [BeamHypothesis(steps=[initial_step], score=initial_score)]

    def expand(
        self, hypotheses: list[BeamHypothesis], candidates: list[list[tuple[str, float]]]
    ) -> list[BeamHypothesis]:
        """
        Expand each hypothesis with its candidate (step, score) list.
        candidates[i] is the list of (step, score) for hypotheses[i].
        Returns beam_width best new hypotheses, sorted by score desc.
        Raises ValueError if len(candidates) != len(hypotheses).
        """
        if len(candidates) != len(hypotheses):
            raise ValueError("candidates length must match hypotheses length")
        new_hyps: list[BeamHypothesis] = []
        for hyp, cands in zip(hypotheses, candidates):
            if hyp.is_finished or hyp.depth >= self.max_depth:
                new_hyps.append(hyp)
                continue
            for step, step_score in cands:
                new_hyps.append(hyp.extend(step, step_score))
        return self._prune(new_hyps)

    def _prune(self, hypotheses: list[BeamHypothesis]) -> list[BeamHypothesis]:
        """Keep top beam_width by score (normalized if enabled)."""
        key = (lambda h: h.normalized_score()) if self.normalize_scores else (lambda h: h.score)
        return sorted(hypotheses, key=key, reverse=True)[: self.beam_width]

    def best(self, hypotheses: list[BeamHypothesis]) -> BeamHypothesis:
        """Return the highest-scoring hypothesis."""
        if not hypotheses:
            raise ValueError("empty hypothesis list")
        key = (lambda h: h.normalized_score()) if self.normalize_scores else (lambda h: h.score)
        return max(hypotheses, key=key)

    def mark_finished(self, hyp: BeamHypothesis) -> BeamHypothesis:
        """Mark a hypothesis as finished (terminal step reached)."""
        return BeamHypothesis(steps=hyp.steps, score=hyp.score, is_finished=True)


BEAM_SEARCH_REASONER = BeamSearchReasoner()
