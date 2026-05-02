"""Universal Self-Consistency (Chen et al., arXiv:2311.17311).

Extends Self-Consistency (Wang et al. 2022) by using the LLM itself — or a
consistency-scoring heuristic — to select the most consistent answer rather
than relying on hard majority voting alone.

Classes
-------
SelfConsistencyVoter   : aggregate answers by frequency (majority vote).
UniversalSCScorer      : score each answer by consistency with all others.
TemperatureSampler     : sample diverse single-token outputs at varied temps.
SelfConsistencyDecoder : full pipeline — sample, score/vote, return best.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# SelfConsistencyVoter
# ---------------------------------------------------------------------------


class SelfConsistencyVoter:
    """Aggregate answers by frequency (majority vote).

    Parameters
    ----------
    answer_extractor:
        Optional callable ``str -> str`` that normalises / extracts the
        relevant portion of each answer before comparison.  When *None*,
        exact string match is used (after ``str.strip()``).
    """

    def __init__(self, answer_extractor: Callable[[str], str] | None = None) -> None:
        self.answer_extractor = answer_extractor

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, answer: str) -> str:
        if self.answer_extractor is not None:
            return self.answer_extractor(answer)
        return answer.strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def vote(self, answers: list[str]) -> tuple[str, dict[str, int]]:
        """Return (majority_answer, vote_counts).

        Parameters
        ----------
        answers:
            List of raw answer strings.

        Returns
        -------
        majority_answer:
            The answer (in its *original*, pre-extraction form) that appears
            most frequently after extraction.  Ties are broken by first
            occurrence among tied answers.
        vote_counts:
            Mapping of *extracted* answer strings to their frequency counts.

        Raises
        ------
        ValueError
            If *answers* is empty.
        """
        if not answers:
            raise ValueError("answers must not be empty")

        extracted = [self._extract(a) for a in answers]
        counts: dict[str, int] = {}
        for e in extracted:
            counts[e] = counts.get(e, 0) + 1

        max_count = max(counts.values())
        # Prefer first occurrence among ties
        winner_extracted: str | None = None
        for e in extracted:
            if counts[e] == max_count:
                winner_extracted = e
                break

        # Find first original answer that maps to the winning extracted form
        winner_original: str = ""
        for orig, ext in zip(answers, extracted):
            if ext == winner_extracted:
                winner_original = orig
                break

        return winner_original, counts

    def vote_with_confidence(self, answers: list[str]) -> tuple[str, float]:
        """Return (majority_answer, confidence).

        confidence = winner_count / len(answers), so it lies in (0, 1].

        Raises
        ------
        ValueError
            If *answers* is empty.
        """
        if not answers:
            raise ValueError("answers must not be empty")

        winner, counts = self.vote(answers)
        extracted_winner = self._extract(winner)
        confidence = counts[extracted_winner] / len(answers)
        return winner, confidence


# ---------------------------------------------------------------------------
# UniversalSCScorer
# ---------------------------------------------------------------------------


class UniversalSCScorer:
    """Score each answer by pairwise consistency with all other answers.

    The score of answer ``a_i`` is defined as the fraction of *other*
    answers ``a_j`` (j ≠ i) that *agree* with ``a_i`` (exact string
    match after strip).  When there is only one answer the score is 0.
    """

    def __init__(self) -> None:
        pass

    def score(self, answers: list[str]) -> Tensor:
        """Return a (N,) float tensor of consistency scores.

        For each answer a_i, score_i = (# of a_j == a_i for j ≠ i) / (N - 1).
        When N == 1, returns tensor([0.0]).
        """
        n = len(answers)
        if n == 0:
            return torch.zeros(0, dtype=torch.float32)
        if n == 1:
            return torch.zeros(1, dtype=torch.float32)

        stripped = [a.strip() for a in answers]
        scores = torch.zeros(n, dtype=torch.float32)
        for i in range(n):
            agree = sum(1 for j in range(n) if j != i and stripped[j] == stripped[i])
            scores[i] = agree / (n - 1)
        return scores

    def select_best(self, answers: list[str], scores: Tensor) -> str:
        """Return the answer with the highest score.

        Ties are broken by first occurrence (lowest index).

        Parameters
        ----------
        answers:
            List of answer strings (same length as *scores*).
        scores:
            (N,) float tensor as returned by :meth:`score`.
        """
        if not answers:
            raise ValueError("answers must not be empty")
        best_idx = int(torch.argmax(scores).item())
        return answers[best_idx]


# ---------------------------------------------------------------------------
# TemperatureSampler
# ---------------------------------------------------------------------------


class TemperatureSampler:
    """Sample diverse single-token outputs at varied temperatures.

    Parameters
    ----------
    model_fn:
        Callable ``(input_ids: LongTensor, temperature: float) -> int``
        that returns a sampled token id.
    temperatures:
        Cycle of temperatures to use.  ``sample_n`` uses
        ``temperatures[i % len(temperatures)]`` for the *i*-th sample.
    """

    def __init__(
        self,
        model_fn: Callable[[Tensor, float], int],
        temperatures: list[float],
    ) -> None:
        if not temperatures:
            raise ValueError("temperatures must not be empty")
        self.model_fn = model_fn
        self.temperatures = temperatures

    def sample_n(self, prompt_ids: Tensor, n: int) -> list[int]:
        """Draw *n* token samples by cycling through temperatures.

        Parameters
        ----------
        prompt_ids:
            1-D or 2-D ``LongTensor`` of prompt token ids.
        n:
            Number of samples to draw.

        Returns
        -------
        List[int]
            Length-*n* list of sampled token ids.
        """
        samples: list[int] = []
        for i in range(n):
            temp = self.temperatures[i % len(self.temperatures)]
            token_id = self.model_fn(prompt_ids, temp)
            samples.append(int(token_id))
        return samples


# ---------------------------------------------------------------------------
# SelfConsistencyDecoder
# ---------------------------------------------------------------------------


class SelfConsistencyDecoder:
    """Full Universal Self-Consistency pipeline.

    Generates *n_samples* completions via *model_fn*, then either applies
    Universal SC scoring (``use_universal=True``) or falls back to majority
    vote (``use_universal=False``).

    Parameters
    ----------
    model_fn:
        Callable ``(prompt_ids: LongTensor, temperature: float) -> LongTensor``
        that generates a *full continuation* given a prompt and temperature.
        The returned tensor is 1-D (sequence of token ids).
    n_samples:
        Number of independent completions to generate.
    use_universal:
        When *True* (default), use :class:`UniversalSCScorer` to select the
        best completion.  When *False*, fall back to
        :class:`SelfConsistencyVoter` majority vote.
    """

    def __init__(
        self,
        model_fn: Callable[[Tensor, float], Tensor],
        n_samples: int = 10,
        use_universal: bool = True,
    ) -> None:
        self.model_fn = model_fn
        self.n_samples = n_samples
        self.use_universal = use_universal
        self._voter = SelfConsistencyVoter()
        self._scorer = UniversalSCScorer()
        # Default temperature schedule for sampling diversity
        self._temperatures = [0.7, 0.8, 0.9, 1.0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tensor_to_str(self, tokens: Tensor) -> str:
        """Convert a 1-D LongTensor to a canonical string for comparison."""
        return str(tokens.tolist())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int = 50,
    ) -> tuple[Tensor, dict]:
        """Generate *n_samples* completions and select the best.

        Parameters
        ----------
        prompt_ids:
            1-D ``LongTensor`` of prompt token ids.
        max_new_tokens:
            Passed informally — the actual limit is enforced by *model_fn*.
            Kept here for API consistency / future use.

        Returns
        -------
        best_tokens:
            The best completion as a 1-D ``LongTensor``.
        stats:
            Dictionary with keys:
            - ``'n_samples'`` (int): number of completions generated.
            - ``'confidence'`` (float): winner's vote share or USC score.
            - ``'all_completions'`` (List[LongTensor]): all sampled completions.
        """
        all_completions: list[Tensor] = []
        temperatures = self._temperatures

        for i in range(self.n_samples):
            temp = temperatures[i % len(temperatures)]
            completion = self.model_fn(prompt_ids, temp)
            # Ensure 1-D LongTensor
            if not isinstance(completion, Tensor):
                completion = torch.tensor(completion, dtype=torch.long)
            completion = completion.view(-1).long()
            all_completions.append(completion)

        # Convert completions to string keys for comparison
        str_completions = [self._tensor_to_str(c) for c in all_completions]

        if self.use_universal:
            scores = self._scorer.score(str_completions)
            best_str = self._scorer.select_best(str_completions, scores)
            best_idx = str_completions.index(best_str)
            # Confidence = the USC score of the winner (fraction of agreements)
            confidence = float(scores[best_idx].item())
            # If all scores are 0 (all unique), set confidence to 1/n for single
            # choice guarantee
            if confidence == 0.0 and self.n_samples > 0:
                confidence = 1.0 / self.n_samples
        else:
            winner_str, vote_counts = self._voter.vote(str_completions)
            best_idx = str_completions.index(winner_str)
            total = sum(vote_counts.values())
            confidence = vote_counts[winner_str] / total if total > 0 else 0.0

        best_tokens = all_completions[best_idx]

        stats: dict = {
            "n_samples": self.n_samples,
            "confidence": float(confidence),
            "all_completions": all_completions,
        }
        return best_tokens, stats
