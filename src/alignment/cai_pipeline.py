"""Constitutional AI Pipeline: iterative self-critique and revision using principles.

Implements CAIConfig, CAIReviser, CAIDataCollector and supporting utilities
as described in Anthropic's Constitutional AI approach (Bai et al., 2022).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Default principles and templates
# ---------------------------------------------------------------------------

_DEFAULT_PRINCIPLES: List[str] = [
    "Be helpful, harmless, and honest.",
    "Avoid harmful or dangerous content.",
    "Be respectful and unbiased.",
]

_DEFAULT_CRITIQUE_TEMPLATE: str = (
    "Identify any issues with the following response based on this principle: {principle}\n\n"
    "Response: {response}\n\nCritique:"
)

_DEFAULT_REVISION_TEMPLATE: str = (
    "Revise the following response to address this critique.\n\n"
    "Original: {response}\nCritique: {critique}\n\nRevised:"
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CAIConfig:
    """Configuration for Constitutional AI revision pipeline."""

    n_revision_steps: int = 2
    principles: Optional[List[str]] = None
    critique_template: str = _DEFAULT_CRITIQUE_TEMPLATE
    revision_template: str = _DEFAULT_REVISION_TEMPLATE

    def __post_init__(self) -> None:
        if self.principles is None:
            self.principles = list(_DEFAULT_PRINCIPLES)


@dataclass
class ConstitutionalPrinciple:
    """A single constitutional principle with an optional weight."""

    text: str
    weight: float = 1.0


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_critique_prompt(response: str, principle: str, template: str) -> str:
    """Format the critique template with the given response and principle."""
    return template.format(principle=principle, response=response)


def build_revision_prompt(response: str, critique: str, template: str) -> str:
    """Format the revision template with the given response and critique."""
    return template.format(response=response, critique=critique)


# ---------------------------------------------------------------------------
# Score extraction
# ---------------------------------------------------------------------------

_NEGATIVE_WORDS = frozenset(
    {"harmful", "dangerous", "biased", "inappropriate", "problematic", "wrong", "bad", "incorrect"}
)
_POSITIVE_WORDS = frozenset({"good", "helpful", "appropriate", "safe", "correct"})


def extract_critique_score(critique_text: str) -> float:
    """Heuristic sentiment score for a critique text.

    Counts occurrences of negative and positive sentiment words.
    Returns a value in [0, 1] where 1 means purely positive.
    """
    words = critique_text.lower().split()
    n_negative = sum(1 for w in words if w.strip(".,!?;:'\"") in _NEGATIVE_WORDS)
    n_positive = sum(1 for w in words if w.strip(".,!?;:'\"") in _POSITIVE_WORDS)

    # raw score in [-1, 1]
    raw = (n_positive - n_negative) / max(n_positive + n_negative, 1)
    # map to [0, 1]
    return (raw + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Safety scoring
# ---------------------------------------------------------------------------

_HARMFUL_WORDS = frozenset(
    {
        "kill", "murder", "bomb", "explode", "attack", "weapon", "harm", "hurt",
        "dangerous", "illegal", "toxic", "poison", "violence", "threat", "abuse",
        "exploit", "hack", "steal", "fraud", "scam",
    }
)


def score_response_safety(response: str) -> float:
    """Lexicon-based safety score.

    Returns a value in [0, 1] where 1 means fully safe and 0 means very unsafe.
    Score = 1 - (n_harmful_words / max(total_words, 1)) clipped to [0, 1].
    """
    words = [w.strip(".,!?;:'\"").lower() for w in response.split()]
    if not words:
        return 1.0
    n_harmful = sum(1 for w in words if w in _HARMFUL_WORDS)
    raw = n_harmful / len(words)
    # Amplify: even a few harmful words should lower the score significantly
    score = max(0.0, 1.0 - raw * 5.0)
    return float(min(score, 1.0))


# ---------------------------------------------------------------------------
# CAIReviser
# ---------------------------------------------------------------------------

class CAIReviser:
    """Applies Constitutional AI critique-revision cycles to a response."""

    def __init__(self, generate_fn: Callable[[str], str], config: CAIConfig) -> None:
        self.generate_fn = generate_fn
        self.config = config

    def critique(self, response: str, principle: str) -> str:
        """Generate a critique of the response with respect to the given principle."""
        prompt = build_critique_prompt(response, principle, self.config.critique_template)
        return self.generate_fn(prompt)

    def revise(self, response: str, critique: str) -> str:
        """Generate a revised response given the original response and a critique."""
        prompt = build_revision_prompt(response, critique, self.config.revision_template)
        return self.generate_fn(prompt)

    def run_revision_cycle(self, response: str) -> List[str]:
        """Iterate through all principles and perform critique-then-revise.

        Returns a flat list of revised responses, one per (step, principle) pair.
        """
        revisions: List[str] = []
        current = response
        principles = self.config.principles or []

        for _step in range(self.config.n_revision_steps):
            for principle in principles:
                critique_text = self.critique(current, principle)
                revised = self.revise(current, critique_text)
                revisions.append(revised)
                current = revised

        return revisions

    def run_full_pipeline(self, initial_response: str) -> Tuple[str, List[str]]:
        """Run the full revision pipeline.

        Returns:
            (final_response, revision_history) where revision_history is the list
            of all intermediate revisions.
        """
        history = self.run_revision_cycle(initial_response)
        final = history[-1] if history else initial_response
        return final, history


# ---------------------------------------------------------------------------
# CAIDataCollector
# ---------------------------------------------------------------------------

class CAIDataCollector:
    """Collects CAI pipeline outputs for training data or analysis."""

    def __init__(self, reviser: CAIReviser) -> None:
        self.reviser = reviser

    def collect(self, responses: List[str]) -> List[Dict[str, str]]:
        """Run the full pipeline on each response.

        Returns a list of dicts with keys: "original", "final", "n_revisions".
        """
        results: List[Dict[str, str]] = []
        for resp in responses:
            final, history = self.reviser.run_full_pipeline(resp)
            results.append(
                {
                    "original": resp,
                    "final": final,
                    "n_revisions": str(len(history)),
                }
            )
        return results

    def get_stats(self, collected: List[Dict]) -> Dict[str, float]:
        """Compute summary statistics over collected data.

        Returns dict with keys: "n_samples", "mean_revisions", "mean_safety_improvement".
        """
        n = len(collected)
        if n == 0:
            return {
                "n_samples": 0.0,
                "mean_revisions": 0.0,
                "mean_safety_improvement": 0.0,
            }

        total_revisions = sum(float(d["n_revisions"]) for d in collected)
        safety_improvements = [
            score_response_safety(d["final"]) - score_response_safety(d["original"])
            for d in collected
        ]
        mean_improvement = sum(safety_improvements) / n

        return {
            "n_samples": float(n),
            "mean_revisions": total_revisions / n,
            "mean_safety_improvement": mean_improvement,
        }
