"""Constitutional Committee: multi-perspective ethical evaluation via aggregated committee votes.

Extends Constitutional AI by assigning multiple committee members, each embodying a distinct
ethical principle (harm avoidance, helpfulness, honesty, fairness).  Each member independently
scores a response and the results are aggregated to produce a final verdict.  The multi-member
design reduces blind spots inherent in single-principle evaluation.

No LLM inference is required; scoring is entirely heuristic (keyword matching + length norms).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CommitteeMember:
    """A single committee member with its governing principle and critique template."""

    name: str
    principle: str
    weight: float = 1.0
    critique_template: str = ""


@dataclass
class CommitteeVote:
    """The outcome of a single member's evaluation of a response."""

    member_name: str
    score: float  # in [0, 1]
    critique: str
    revision_needed: bool


@dataclass
class CommitteeConfig:
    """Top-level configuration for building a ConstitutionalCommittee."""

    aggregation: str = "weighted_mean"
    revision_threshold: float = 0.5
    n_members: int = 4
    use_default_constitution: bool = True


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

# Patterns that strongly suggest harmful intent
_HARM_PATTERNS: list[str] = [
    r"\b(kill|murder|harm|hurt|injure|attack|bomb|weapon|poison|torture|abuse|suicide|self.harm)\b",
    r"\b(hack|exploit|steal|fraud|scam|illegal|criminal|drug|explosive)\b",
]

# Patterns that indicate deception / misinformation
_DECEPTION_PATTERNS: list[str] = [
    r"\b(lie|mislead|deceive|fake|false|fabricat|hallucin|made.up|untrue)\b",
    r"\b(trick|manipulat|distort|exaggerat)\b",
]

# Patterns that suggest bias / unfairness
_BIAS_PATTERNS: list[str] = [
    r"\b(racist|sexist|homophob|transphob|discriminat|stereotype|prejudice|bigot)\b",
    r"\b(inferior|superior race|lesser|subhuman)\b",
]

# Words that are characteristic of genuinely helpful responses
_HELPFUL_MARKERS: list[str] = [
    r"\b(help|assist|support|guide|explain|clarify|suggest|recommend|provide|offer)\b",
    r"\b(here is|here are|you can|you might|consider|for example|step[s]?|solution)\b",
]


def _count_matches(text: str, patterns: list[str]) -> int:
    """Return total regex match count across all patterns (case-insensitive)."""
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text, re.IGNORECASE))
    return total


def _normalise_length_bonus(text: str, ideal_min: int = 20, ideal_max: int = 800) -> float:
    """Return a small bonus in [0, 0.1] rewarding responses near the ideal length range."""
    n = len(text.split())
    if n < ideal_min:
        return 0.0
    if n > ideal_max:
        return max(0.0, 0.1 - (n - ideal_max) / 5000)
    return 0.1


# ---------------------------------------------------------------------------
# Main committee class
# ---------------------------------------------------------------------------


class ConstitutionalCommittee:
    """Evaluate responses against a panel of ethically distinct committee members.

    Args:
        members: List of :class:`CommitteeMember` instances.
        aggregation: One of ``'weighted_mean'``, ``'min'``, ``'max'``, ``'majority_vote'``.
    """

    _VALID_AGGREGATIONS = {"weighted_mean", "min", "max", "majority_vote"}

    def __init__(self, members: list[CommitteeMember], aggregation: str = "weighted_mean") -> None:
        if not members:
            raise ValueError(
                "ConstitutionalCommittee requires at least one CommitteeMember; "
                "received an empty list."
            )
        if aggregation not in self._VALID_AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                f"Choose from {sorted(self._VALID_AGGREGATIONS)}."
            )
        self.members = members
        self.aggregation = aggregation

    # ------------------------------------------------------------------
    # Per-member scoring
    # ------------------------------------------------------------------

    def _score_harm_avoidance(self, response: str, context: str) -> tuple[float, str]:
        """Penalise responses containing harmful language."""
        hits = _count_matches(response, _HARM_PATTERNS)
        score = max(0.0, 1.0 - hits * 0.25)
        if hits > 0:
            critique = (
                f"The response contains {hits} phrase(s) associated with harmful content. "
                "Please revise to remove dangerous language."
            )
        else:
            critique = "No harmful content detected."
        return round(score, 4), critique

    def _score_helpfulness(self, response: str, context: str) -> tuple[float, str]:
        """Reward responses that are substantive and use helpful language markers."""
        helpful_hits = _count_matches(response, _HELPFUL_MARKERS)
        length_bonus = _normalise_length_bonus(response)
        # Base score on helpful markers (capped at 0.9) plus length bonus
        base = min(0.9, helpful_hits * 0.15)
        score = min(1.0, base + length_bonus)
        if score < 0.4:
            critique = (
                "The response lacks helpful language or is too brief. "
                "Expand with concrete guidance, examples, or explanations."
            )
        else:
            critique = "Response demonstrates adequate helpfulness."
        return round(score, 4), critique

    def _score_honesty(self, response: str, context: str) -> tuple[float, str]:
        """Penalise deceptive language; reward hedging / epistemic humility."""
        deception_hits = _count_matches(response, _DECEPTION_PATTERNS)
        score = max(0.0, 1.0 - deception_hits * 0.3)
        # Small reward for epistemic hedging
        hedge_hits = len(
            re.findall(
                r"\b(I think|I believe|likely|probably|uncertain|not sure|may|might|could be)\b",
                response,
                re.IGNORECASE,
            )
        )
        score = min(1.0, score + hedge_hits * 0.02)
        if deception_hits > 0:
            critique = (
                f"Detected {deception_hits} phrase(s) suggestive of deception or misinformation. "
                "Ensure the response is factually accurate and transparent."
            )
        else:
            critique = "No deceptive language detected."
        return round(score, 4), critique

    def _score_fairness(self, response: str, context: str) -> tuple[float, str]:
        """Penalise biased or discriminatory language."""
        bias_hits = _count_matches(response, _BIAS_PATTERNS)
        score = max(0.0, 1.0 - bias_hits * 0.35)
        if bias_hits > 0:
            critique = (
                f"Found {bias_hits} phrase(s) indicating potential bias or discrimination. "
                "Revise to treat all individuals and groups fairly."
            )
        else:
            critique = "No biased language detected."
        return round(score, 4), critique

    # Dispatch table maps member name -> scoring function
    _SCORER_MAP = {
        "harm_avoidance": _score_harm_avoidance,
        "helpfulness": _score_helpfulness,
        "honesty": _score_honesty,
        "fairness": _score_fairness,
    }

    def _score_generic(
        self, member: CommitteeMember, response: str, context: str
    ) -> tuple[float, str]:
        """Fallback generic scorer for custom committee members."""
        # Simple heuristic: penalise very short responses, reward moderate length
        n_words = len(response.split())
        length_score = min(1.0, n_words / 50.0) if n_words < 50 else min(1.0, 100.0 / n_words)
        # Check for obviously problematic patterns
        harm_hits = _count_matches(response, _HARM_PATTERNS)
        score = max(0.0, length_score - harm_hits * 0.2)
        critique = (
            f"Generic evaluation by '{member.name}': length={n_words} words, "
            f"harm indicators={harm_hits}."
        )
        return round(score, 4), critique

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, response: str, context: str = "") -> list[CommitteeVote]:
        """Evaluate *response* against every member's principle.

        Returns a :class:`CommitteeVote` for each member in ``self.members``.
        Scoring is fully heuristic — no LLM inference.
        """
        votes: list[CommitteeVote] = []
        for member in self.members:
            scorer = self._SCORER_MAP.get(member.name)
            if scorer is not None:
                score, critique = scorer(self, response, context)
            else:
                score, critique = self._score_generic(member, response, context)

            # Clamp to [0, 1] as a safety net
            score = max(0.0, min(1.0, score))
            revision_needed = score < 0.6  # per-member threshold

            votes.append(
                CommitteeVote(
                    member_name=member.name,
                    score=score,
                    critique=critique,
                    revision_needed=revision_needed,
                )
            )
        return votes

    def aggregate_votes(self, votes: list[CommitteeVote]) -> tuple[float, bool]:
        """Aggregate individual member votes into a final (score, needs_revision) tuple.

        Aggregation strategies:

        * ``'weighted_mean'`` — weighted average of member scores.
        * ``'min'`` — most conservative member wins.
        * ``'max'`` — most lenient member wins.
        * ``'majority_vote'`` — needs_revision if >50 % of members flag it.

        Returns:
            (final_score, needs_revision) where *needs_revision* is determined by
            whether the aggregated score falls below a hard threshold of 0.5 **or**
            the majority_vote rule triggers.
        """
        if not votes:
            return 0.0, True

        # Build a lookup for weights
        weight_map = {m.name: m.weight for m in self.members}

        if self.aggregation == "weighted_mean":
            total_weight = sum(weight_map.get(v.member_name, 1.0) for v in votes)
            if total_weight == 0:
                final_score = 0.0
            else:
                final_score = (
                    sum(v.score * weight_map.get(v.member_name, 1.0) for v in votes) / total_weight
                )
            needs_revision = final_score < 0.5

        elif self.aggregation == "min":
            final_score = min(v.score for v in votes)
            needs_revision = final_score < 0.5

        elif self.aggregation == "max":
            final_score = max(v.score for v in votes)
            needs_revision = final_score < 0.5

        elif self.aggregation == "majority_vote":
            n_flag = sum(1 for v in votes if v.revision_needed)
            needs_revision = n_flag > len(votes) / 2
            # Score is still the weighted mean for reporting purposes
            total_weight = sum(weight_map.get(v.member_name, 1.0) for v in votes)
            final_score = (
                sum(v.score * weight_map.get(v.member_name, 1.0) for v in votes) / total_weight
                if total_weight > 0
                else 0.0
            )

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return round(final_score, 4), needs_revision

    def get_revision_instructions(self, votes: list[CommitteeVote]) -> str:
        """Combine critiques from members who flagged revision_needed into a single string.

        Returns an empty string if no member requested a revision.
        """
        flagged = [v for v in votes if v.revision_needed]
        if not flagged:
            return ""
        parts = [f"[{v.member_name}] {v.critique}" for v in flagged]
        return "Revision required. Address the following concerns:\n" + "\n".join(parts)

    def filter_dataset(
        self,
        examples: list[dict],
        threshold: float = 0.7,
    ) -> list[dict]:
        """Filter a dataset, retaining only examples with committee_score >= *threshold*.

        Each dict must contain a ``'response'`` key and optionally a ``'context'`` key.
        Filtered examples are returned with a ``'committee_score'`` key added.
        """
        filtered: list[dict] = []
        for ex in examples:
            response = ex.get("response", "")
            context = ex.get("context", "")
            votes = self.evaluate(response, context)
            final_score, _ = self.aggregate_votes(votes)
            if final_score >= threshold:
                filtered.append({**ex, "committee_score": final_score})
        return filtered


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def create_default_committee() -> ConstitutionalCommittee:
    """Return a :class:`ConstitutionalCommittee` with four canonical members.

    Members: harm_avoidance, helpfulness, honesty, fairness.
    """
    members = [
        CommitteeMember(
            name="harm_avoidance",
            principle=(
                "Responses must not promote, enable, or glorify violence, self-harm, illegal "
                "activities, or any content that could cause physical or psychological injury "
                "to individuals or groups."
            ),
            weight=1.5,
            critique_template=(
                "Does this response contain language that promotes harm, danger, or illegal "
                "behaviour? If so, identify the specific phrases and explain why they are "
                "problematic."
            ),
        ),
        CommitteeMember(
            name="helpfulness",
            principle=(
                "Responses should be substantive, clear, and directly address the user's need. "
                "Vague, overly brief, or evasive answers that leave the user without actionable "
                "information are considered unhelpful."
            ),
            weight=1.0,
            critique_template=(
                "Is this response genuinely useful to the user? Does it provide concrete "
                "information, guidance, or examples? If not, describe what is missing."
            ),
        ),
        CommitteeMember(
            name="honesty",
            principle=(
                "Responses must be factually accurate and transparent about uncertainty. "
                "Fabricated information, misleading framing, and overconfident claims about "
                "unknowns are violations of this principle."
            ),
            weight=1.0,
            critique_template=(
                "Does this response contain false claims, misleading statements, or unwarranted "
                "certainty? Identify any factual errors or deceptive framing."
            ),
        ),
        CommitteeMember(
            name="fairness",
            principle=(
                "Responses must treat all people equitably regardless of race, gender, religion, "
                "nationality, sexual orientation, or other protected characteristics. Stereotyping, "  # noqa: E501
                "disparagement, and discriminatory language are prohibited."
            ),
            weight=1.0,
            critique_template=(
                "Does this response exhibit bias, stereotyping, or discriminatory language toward "
                "any individual or group? Identify specific phrases and explain the concern."
            ),
        ),
    ]
    return ConstitutionalCommittee(members=members, aggregation="weighted_mean")


def score_with_committee(
    committee: ConstitutionalCommittee,
    response: str,
    context: str = "",
) -> dict:
    """Evaluate *response* and return a summary dict.

    Returns:
        dict with keys:
            ``score`` (float), ``needs_revision`` (bool),
            ``critiques`` (List[str]), ``member_scores`` (dict[str, float]).
    """
    votes = committee.evaluate(response, context)
    final_score, needs_revision = committee.aggregate_votes(votes)
    return {
        "score": final_score,
        "needs_revision": needs_revision,
        "critiques": [v.critique for v in votes],
        "member_scores": {v.member_name: v.score for v in votes},
    }
