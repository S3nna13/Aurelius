"""Reward-Guided Search — value-guided beam search for reasoning (ARGS / PRM-guided, 2025).

At each step candidates are scored as:
    score = (1 - λ) * log_prob + λ * value_score
with a length penalty applied:
    score /= length ** α

Used in MCTS-based reasoning, process-reward-model (PRM) guided beam search,
and value-guided decoding for math/code tasks.

Reference: ARGS (Reward-Guided Search, 2025); Lightman et al. "Let's Verify
Step by Step" (2023); value-guided beam search literature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    """Hyperparameters for RewardGuidedSearch."""

    beam_width: int = 4
    max_steps: int = 64
    reward_weight: float = 0.5       # λ: weight on value_fn vs log_prob
    length_penalty: float = 0.6      # α in score /= len^α
    vocab_size: int = 128000
    eos_token_id: int = 2
    pad_token_id: int = 0


# ---------------------------------------------------------------------------
# Beam dataclass
# ---------------------------------------------------------------------------

@dataclass
class SearchBeam:
    """Represents one beam (partial hypothesis) in reward-guided search."""

    token_ids: list[int]
    log_prob_sum: float
    value_sum: float
    score: float = 0.0
    is_finished: bool = False


# ---------------------------------------------------------------------------
# Core search class
# ---------------------------------------------------------------------------

class RewardGuidedSearch:
    """Value-guided beam search that combines LM log-probs with a reward/value signal.

    Algorithm
    ---------
    At each expansion step each active beam is scored as::

        combined = (1 - λ) * log_prob_sum + λ * value_score
        score    = combined / (length ** α)

    The top ``beam_width`` candidates across all beams are kept.
    """

    def __init__(self, config: SearchConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def score_candidate(
        self,
        log_prob: float,
        value: float,
        length: int,
    ) -> float:
        """Compute length-penalised combined score for a candidate.

        Args:
            log_prob: cumulative log-probability of the sequence.
            value:    cumulative value score of the sequence.
            length:   number of tokens (excluding prompt) for length penalty.

        Returns:
            Scalar score.
        """
        lam = self.config.reward_weight
        combined = (1.0 - lam) * log_prob + lam * value
        penalty = max(length, 1) ** self.config.length_penalty
        return combined / penalty

    # ------------------------------------------------------------------
    # Beam expansion
    # ------------------------------------------------------------------

    def expand_beams(
        self,
        beams: list[SearchBeam],
        next_token_logits: Tensor,
        value_fn: Callable[[list[int]], float],
    ) -> list[SearchBeam]:
        """Expand each active beam by one token and return the top-k survivors.

        Args:
            beams:             current beam list (some may be finished).
            next_token_logits: ``[beam_width, vocab_size]`` logits tensor; row *i*
                               corresponds to ``beams[i]``.  Finished beams are
                               skipped but their row is still expected to be present
                               so indices line up.
            value_fn:          callable ``tokens → float`` value score.

        Returns:
            New list of at most ``beam_width`` beams sorted by score (desc).
        """
        cfg = self.config
        bw = cfg.beam_width

        # Ensure we have a 2-D tensor
        if next_token_logits.dim() == 1:
            next_token_logits = next_token_logits.unsqueeze(0)

        log_probs = F.log_softmax(next_token_logits, dim=-1)  # [n_beams, vocab]

        candidates: list[SearchBeam] = []

        for i, beam in enumerate(beams):
            if beam.is_finished:
                # Carry finished beams forward unchanged.
                candidates.append(beam)
                continue

            beam_log_probs = log_probs[i]  # [vocab]
            # Expand top-k tokens for this beam.
            k = min(bw, cfg.vocab_size)
            top_lp, top_tok = torch.topk(beam_log_probs, k)

            for j in range(k):
                tok = int(top_tok[j].item())
                lp = float(top_lp[j].item())

                new_token_ids = beam.token_ids + [tok]
                new_log_prob_sum = beam.log_prob_sum + lp
                # Query value function with the new (extended) sequence.
                v = value_fn(new_token_ids)
                new_value_sum = beam.value_sum + v

                is_eos = tok == cfg.eos_token_id
                length = len(new_token_ids)
                sc = self.score_candidate(new_log_prob_sum, new_value_sum, length)

                new_beam = SearchBeam(
                    token_ids=new_token_ids,
                    log_prob_sum=new_log_prob_sum,
                    value_sum=new_value_sum,
                    score=sc,
                    is_finished=is_eos,
                )
                candidates.append(new_beam)

        # Rank all candidates by score descending; keep top beam_width.
        candidates.sort(key=lambda b: b.score, reverse=True)
        return candidates[:bw]

    # ------------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------------

    def search(
        self,
        initial_tokens: list[int],
        model_fn: Callable[[list[int]], Tensor],
        value_fn: Callable[[list[int]], float],
        n_steps: Optional[int] = None,
    ) -> list[SearchBeam]:
        """Run reward-guided beam search.

        Args:
            initial_tokens: prompt token ids.
            model_fn:       ``tokens → logits [vocab_size]`` (for a single beam).
            value_fn:       ``tokens → float`` value/reward signal.
            n_steps:        override ``config.max_steps`` if provided.

        Returns:
            Final list of beams sorted by score descending (best first).
        """
        cfg = self.config
        max_steps = n_steps if n_steps is not None else cfg.max_steps

        # Initialise a single beam carrying the prompt.
        beams: list[SearchBeam] = [
            SearchBeam(
                token_ids=list(initial_tokens),
                log_prob_sum=0.0,
                value_sum=0.0,
                score=0.0,
                is_finished=False,
            )
        ]

        for _ in range(max_steps):
            # Stop early if every beam is finished.
            if all(b.is_finished for b in beams):
                break

            # Collect logits for every active beam.
            logit_rows: list[Tensor] = []
            for beam in beams:
                if beam.is_finished:
                    # Placeholder logits (will be ignored in expand_beams).
                    logit_rows.append(torch.zeros(cfg.vocab_size))
                else:
                    logits = model_fn(beam.token_ids)
                    logit_rows.append(logits)

            stacked = torch.stack(logit_rows, dim=0)  # [n_beams, vocab]
            beams = self.expand_beams(beams, stacked, value_fn)

        # Final sort: best score first.
        beams.sort(key=lambda b: b.score, reverse=True)
        return beams

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def best_sequence(self, beams: list[SearchBeam]) -> list[int]:
        """Return the token ids of the highest-scored beam.

        Prefers finished beams; falls back to the longest active beam if none
        are finished.

        Args:
            beams: list of SearchBeam (need not be sorted).

        Returns:
            Token id list.
        """
        finished = [b for b in beams if b.is_finished]
        if finished:
            return max(finished, key=lambda b: b.score).token_ids
        # No finished beams — return the longest one.
        return max(beams, key=lambda b: len(b.token_ids)).token_ids

    def diversity_score(self, beams: list[SearchBeam]) -> float:
        """Mean pairwise normalised edit distance across all beams.

        Uses token-level Levenshtein distance.  The distance between two
        sequences is normalised by their maximum length so it lies in [0, 1].

        Args:
            beams: list of SearchBeam.

        Returns:
            Float in [0, 1]; 0.0 when all sequences are identical or there is
            only one beam.
        """
        n = len(beams)
        if n <= 1:
            return 0.0

        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                a = beams[i].token_ids
                b = beams[j].token_ids
                dist = _edit_distance(a, b)
                max_len = max(len(a), len(b), 1)
                total += dist / max_len
                count += 1

        return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _edit_distance(a: list[int], b: list[int]) -> int:
    """Compute token-level Levenshtein (edit) distance between two sequences."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    # DP table — two-row rolling array.
    prev = list(range(lb + 1))
    curr = [0] * (lb + 1)
    for i in range(1, la + 1):
        curr[0] = i
        for j in range(1, lb + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, [0] * (lb + 1)
    return prev[lb]
