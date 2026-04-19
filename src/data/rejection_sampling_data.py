"""Rejection-sampling data synthesis for DPO / preference pairs.

Given a pool of prompts, sample N candidate responses per prompt with a
user-supplied ``generate_fn``, score each with ``reward_fn``, then build
``(chosen, rejected)`` preference pairs from the highest- and lowest-reward
responses. This is the classic offline DPO / reward-model bootstrap pattern
(Rafailov et al., 2023 -- "Direct Preference Optimization").

Pure stdlib; no foreign imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, List, Optional, Tuple

__all__ = ["PreferencePair", "RejectionSampler"]


@dataclass(frozen=True)
class PreferencePair:
    """A single (chosen, rejected) preference example with reward metadata."""

    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    margin: float


class RejectionSampler:
    """Build preference pairs from rejection sampling.

    Parameters
    ----------
    generate_fn:
        Callable ``prompt -> response`` used to draw candidates. Called
        ``n_samples`` times per prompt (or ``max_candidates`` if set and
        smaller).
    reward_fn:
        Callable ``(prompt, response) -> float`` returning a scalar reward.
        If it raises, that candidate is silently skipped.
    n_samples:
        Number of candidates to draw per prompt. Must be >= 1.
    min_margin:
        Minimum ``chosen_reward - rejected_reward`` for a pair to be kept.
        Must be >= 0.0.
    max_candidates:
        Optional upper bound on how many candidates are actually generated
        per prompt (caps ``n_samples``).
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        reward_fn: Callable[[str, str], float],
        n_samples: int = 4,
        min_margin: float = 0.0,
        max_candidates: Optional[int] = None,
    ) -> None:
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable")
        if not callable(reward_fn):
            raise TypeError("reward_fn must be callable")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if min_margin < 0.0:
            raise ValueError(f"min_margin must be >= 0.0, got {min_margin}")
        if max_candidates is not None and max_candidates < 1:
            raise ValueError(
                f"max_candidates must be >= 1 or None, got {max_candidates}"
            )

        self.generate_fn = generate_fn
        self.reward_fn = reward_fn
        self.n_samples = n_samples
        self.min_margin = min_margin
        self.max_candidates = max_candidates

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _effective_n(self) -> int:
        if self.max_candidates is None:
            return self.n_samples
        return min(self.n_samples, self.max_candidates)

    def _score_candidates(self, prompt: str) -> List[Tuple[int, str, float]]:
        """Generate and score candidates. Returns list of (index, text, reward).

        Candidates whose ``reward_fn`` raises are skipped. Index is the
        generation order (stable, used for deterministic tie-breaks).
        """
        n = self._effective_n()
        scored: List[Tuple[int, str, float]] = []
        for i in range(n):
            response = self.generate_fn(prompt)
            try:
                reward = float(self.reward_fn(prompt, response))
            except Exception:
                continue
            scored.append((i, response, reward))
        return scored

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def sample_top_bottom(self, prompt: str) -> Optional[PreferencePair]:
        """Return the (best, worst) preference pair for a single prompt.

        Returns ``None`` if fewer than 2 valid candidates, or if the margin
        is below ``min_margin``.
        """
        scored = self._score_candidates(prompt)
        if len(scored) < 2:
            return None

        # Deterministic tie-break: among ties, prefer the earliest index for
        # both best and worst. When all candidates tie, best and worst are
        # the same candidate, which we treat as "no preference" and return
        # None.
        best = max(scored, key=lambda t: (t[2], -t[0]))
        worst = min(scored, key=lambda t: (t[2], t[0]))
        if best[0] == worst[0]:
            return None

        margin = best[2] - worst[2]
        if margin < self.min_margin:
            return None

        return PreferencePair(
            prompt=prompt,
            chosen=best[1],
            rejected=worst[1],
            chosen_reward=best[2],
            rejected_reward=worst[2],
            margin=margin,
        )

    def sample_all_pairs(self, prompt: str) -> List[PreferencePair]:
        """Return all C(n, 2) pairs for a single prompt.

        Each pair orients the higher-reward candidate as ``chosen``. Pairs
        below ``min_margin`` are filtered out. Pairs with exactly equal
        rewards are dropped (no meaningful preference).
        """
        scored = self._score_candidates(prompt)
        pairs: List[PreferencePair] = []
        for (i, resp_i, r_i), (j, resp_j, r_j) in combinations(scored, 2):
            if r_i > r_j:
                chosen_resp, chosen_r = resp_i, r_i
                rej_resp, rej_r = resp_j, r_j
            elif r_j > r_i:
                chosen_resp, chosen_r = resp_j, r_j
                rej_resp, rej_r = resp_i, r_i
            else:
                # tie -- deterministic but uninformative; skip.
                continue
            margin = chosen_r - rej_r
            if margin < self.min_margin:
                continue
            pairs.append(
                PreferencePair(
                    prompt=prompt,
                    chosen=chosen_resp,
                    rejected=rej_resp,
                    chosen_reward=chosen_r,
                    rejected_reward=rej_r,
                    margin=margin,
                )
            )
        return pairs

    def sample_pairs(self, prompts: List[str]) -> List[PreferencePair]:
        """Build one top/bottom preference pair per prompt.

        Prompts whose margin is below ``min_margin`` (or which cannot form a
        pair at all) are silently skipped.
        """
        out: List[PreferencePair] = []
        for prompt in prompts:
            pair = self.sample_top_bottom(prompt)
            if pair is not None:
                out.append(pair)
        return out
