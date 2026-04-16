"""Constitutional AI v2: principle-based iterative critique and revision.

This module implements a lightweight, model-agnostic Constitutional AI loop
that critiques model responses against a set of principles and iteratively
revises them toward better alignment.

Reference: Bai et al., 2022 (arXiv:2212.08073).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CAIConfig:
    """Configuration for a Constitutional AI v2 session."""

    principles: List[str] = field(
        default_factory=lambda: ["Be helpful", "Be harmless", "Be honest"]
    )
    n_revisions: int = 2
    critique_prefix: str = "Critique:"
    revision_prefix: str = "Revision:"
    max_critique_len: int = 200
    max_revision_len: int = 500


@dataclass
class Principle:
    """A single alignment principle with optional weight and category."""

    text: str
    weight: float = 1.0
    category: str = "general"


# ---------------------------------------------------------------------------
# Prompt formatters
# ---------------------------------------------------------------------------


def format_critique_prompt(response: str, principle: str, config: CAIConfig) -> str:
    """Return a prompt asking a model to critique *response* against *principle*.

    The returned string contains both the original response and the principle
    so a language model can produce a targeted critique.
    """
    return (
        f"Given the following response, critique it with respect to the principle "
        f'"{principle}".\n\n'
        f"Response:\n{response}\n\n"
        f"Principle: {principle}\n\n"
        f"{config.critique_prefix}"
    )


def format_revision_prompt(
    response: str, critique: str, principle: str, config: CAIConfig
) -> str:
    """Return a prompt asking a model to revise *response* based on *critique*.

    The returned string contains the original response, the critique, and the
    governing principle.
    """
    return (
        f"Given the following response and critique, revise the response to better "
        f'adhere to the principle "{principle}".\n\n'
        f"Original response:\n{response}\n\n"
        f"Critique:\n{critique}\n\n"
        f"Principle: {principle}\n\n"
        f"{config.revision_prefix}"
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def score_principle_adherence(response: str, principle: str) -> float:
    """Heuristic score in [0, 1] for how well *response* adheres to *principle*.

    The score is the fraction of content words (len > 3) from the principle
    that appear in the response (case-insensitive).  Returns 0.0 when there
    are no qualifying words.
    """
    words = [w.lower() for w in principle.split() if len(w) > 3]
    if not words:
        return 0.0
    response_lower = response.lower()
    matches = sum(1 for w in words if w in response_lower)
    return matches / len(words)


def select_worst_principle(
    response: str, principles: List[str]
) -> Tuple[int, float]:
    """Return (index_of_worst_principle, lowest_score).

    "Worst" means the principle with the lowest adherence score — the one
    most in need of a revision pass.
    """
    scores = [score_principle_adherence(response, p) for p in principles]
    idx = int(min(range(len(scores)), key=lambda i: scores[i]))
    return idx, scores[idx]


# ---------------------------------------------------------------------------
# CAI Session
# ---------------------------------------------------------------------------


class CAISession:
    """Stateful Constitutional AI session that manages critique-revision loops."""

    def __init__(self, config: CAIConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def critique(self, response: str, principle: str) -> str:
        """Produce a critique of *response* w.r.t. *principle*.

        In production this would call a language model; here we return a
        deterministic stub suitable for unit testing and integration.
        """
        prompt = format_critique_prompt(response, principle, self.config)
        # Stub output — a real implementation would pass *prompt* to a model.
        return f"{prompt} [Generated critique based on principle]"

    def revise(self, response: str, critique: str, principle: str) -> str:
        """Produce a revision of *response* guided by *critique* and *principle*.

        In production this would call a language model; here we return a
        deterministic stub.
        """
        prompt = format_revision_prompt(response, critique, principle, self.config)
        return f"{prompt} [Revised response]"

    def run_revision_loop(self, initial_response: str) -> List[dict]:
        """Run ``config.n_revisions`` critique-revise iterations.

        Each iteration selects the principle least adhered to by the current
        response, critiques it, then revises.

        Returns a list of dicts with keys:
            ``iteration``, ``principle``, ``critique``, ``revision``,
            ``score_before``, ``score_after``.
        """
        history: List[dict] = []
        current_response = initial_response

        for i in range(self.config.n_revisions):
            idx, score_before = select_worst_principle(
                current_response, self.config.principles
            )
            principle = self.config.principles[idx]

            critique = self.critique(current_response, principle)
            revision = self.revise(current_response, critique, principle)

            score_after = score_principle_adherence(revision, principle)

            history.append(
                {
                    "iteration": i,
                    "principle": principle,
                    "critique": critique,
                    "revision": revision,
                    "score_before": score_before,
                    "score_after": score_after,
                }
            )
            current_response = revision

        return history

    def get_final_response(self, revision_history: List[dict]) -> str:
        """Return the last revised response from *revision_history*."""
        if not revision_history:
            return ""
        return revision_history[-1]["revision"]


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------


def compute_principle_coverage(
    revision_history: List[dict], principles: List[str]
) -> dict:
    """Summarise how well the revision loop covered all principles.

    Returns a dict with:
        ``coverage`` — fraction of principles addressed at least once.
        ``mean_improvement`` — mean of (score_after - score_before) across iterations.
    """
    if not revision_history:
        return {"coverage": 0.0, "mean_improvement": 0.0}

    addressed = {entry["principle"] for entry in revision_history}
    coverage = len(addressed & set(principles)) / max(len(principles), 1)

    improvements = [
        entry["score_after"] - entry["score_before"] for entry in revision_history
    ]
    mean_improvement = sum(improvements) / len(improvements)

    return {"coverage": coverage, "mean_improvement": mean_improvement}
