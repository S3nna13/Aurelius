"""Preference data collection: generate, score, and format chosen/rejected pairs for RLHF."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PreferenceConfig:
    """Configuration for preference data collection."""

    n_responses: int = 4
    scoring_method: str = "length"          # "length" | "reward_model" | "rule_based"
    min_response_len: int = 10
    max_response_len: int = 500
    tie_threshold: float = 0.1              # discard pairs with score diff < threshold
    format: str = "dpo"                     # "dpo" | "rlhf" | "sft_only"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PreferencePair:
    """A single chosen/rejected preference pair."""

    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    score_diff: float
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_by_length(response: str, target_len: int = 100) -> float:
    """Score a response based on proximity to *target_len* characters.

    Returns a float in [0, 1] where 1.0 means exact match.
    """
    resp_len = len(response)
    denom = max(resp_len, target_len)
    if denom == 0:
        return 1.0
    score = 1.0 - abs(resp_len - target_len) / denom
    return float(score)


def score_by_rules(response: str) -> float:
    """Rule-based quality scoring.

    Adjustments applied:
    - Start at 0.5 baseline
    - Penalize very short responses (< 20 chars): -0.3
    - Reward ending with sentence-closing punctuation: +0.2
    - Penalize excessive 5-gram repetition (any 5-gram appears > 2 times): -0.3
    - Penalize all-caps text (> 50 % uppercase letters): -0.2

    Clipped to [0, 1].
    """
    score = 0.5

    # Penalize very short responses
    if len(response) < 20:
        score -= 0.3

    # Reward complete sentences (ends with punctuation)
    stripped = response.rstrip()
    if stripped and stripped[-1] in ".!?":
        score += 0.2

    # Penalize excessive 5-gram repetition
    words = response.split()
    if len(words) >= 5:
        ngram_counts: dict[tuple, int] = {}
        for i in range(len(words) - 4):
            gram = tuple(words[i : i + 5])
            ngram_counts[gram] = ngram_counts.get(gram, 0) + 1
        if any(count > 2 for count in ngram_counts.values()):
            score -= 0.3

    # Penalize all-caps
    letters = [ch for ch in response if ch.isalpha()]
    if letters and sum(1 for ch in letters if ch.isupper()) / len(letters) > 0.5:
        score -= 0.2

    return float(max(0.0, min(1.0, score)))


# ---------------------------------------------------------------------------
# Pair creation and formatting
# ---------------------------------------------------------------------------

def create_preference_pair(
    prompt: str,
    responses: list[str],
    scores: list[float],
    config: PreferenceConfig,
) -> PreferencePair | None:
    """Select the best (chosen) and worst (rejected) response by score.

    Returns None if:
    - The score difference is below *tie_threshold* (tie, discard).
    - Either response falls outside the configured length bounds.
    """
    if not responses or not scores or len(responses) != len(scores):
        return None

    ranked = sorted(zip(scores, responses), key=lambda x: x[0])
    worst_score, rejected = ranked[0]
    best_score, chosen = ranked[-1]

    score_diff = best_score - worst_score

    if score_diff < config.tie_threshold:
        return None

    if not (config.min_response_len <= len(chosen) <= config.max_response_len):
        return None
    if not (config.min_response_len <= len(rejected) <= config.max_response_len):
        return None

    return PreferencePair(
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
        chosen_score=best_score,
        rejected_score=worst_score,
        score_diff=score_diff,
    )


def format_for_dpo(pair: PreferencePair) -> dict:
    """Return a DPO-format dict with prompt, chosen, and rejected keys."""
    return {
        "prompt": pair.prompt,
        "chosen": pair.chosen,
        "rejected": pair.rejected,
    }


def format_for_rlhf(pair: PreferencePair) -> list[dict]:
    """Return two RLHF-format dicts: chosen with positive reward, rejected with negative.

    Each entry has keys: prompt, response, reward.
    """
    return [
        {
            "prompt": pair.prompt,
            "response": pair.chosen,
            "reward": pair.chosen_score,
        },
        {
            "prompt": pair.prompt,
            "response": pair.rejected,
            "reward": -pair.rejected_score,
        },
    ]


# ---------------------------------------------------------------------------
# Collector class
# ---------------------------------------------------------------------------

class PreferenceCollector:
    """High-level interface for collecting and formatting preference pairs."""

    def __init__(
        self,
        config: PreferenceConfig,
        score_fn: Callable[[str], float] | None = None,
    ) -> None:
        self.config = config

        if score_fn is not None:
            self._score_fn = score_fn
        elif config.scoring_method == "rule_based":
            self._score_fn = score_by_rules
        else:
            # Default: length-based scoring
            self._score_fn = score_by_length

    # ------------------------------------------------------------------
    # Core collection
    # ------------------------------------------------------------------

    def collect(self, prompt: str, responses: list[str]) -> PreferencePair | None:
        """Score *responses* and return a PreferencePair, or None if ineligible."""
        scores = [self._score_fn(r) for r in responses]
        return create_preference_pair(prompt, responses, scores, self.config)

    def collect_batch(
        self,
        prompts: list[str],
        responses_list: list[list[str]],
    ) -> list[PreferencePair]:
        """Batch collection; None pairs are filtered out."""
        pairs: list[PreferencePair] = []
        for prompt, responses in zip(prompts, responses_list):
            pair = self.collect(prompt, responses)
            if pair is not None:
                pairs.append(pair)
        return pairs

    # ------------------------------------------------------------------
    # Dataset formatting
    # ------------------------------------------------------------------

    def to_dataset(self, pairs: list[PreferencePair]) -> list[dict]:
        """Convert pairs to training-format dicts based on *config.format*."""
        records: list[dict] = []
        for pair in pairs:
            if self.config.format == "dpo":
                records.append(format_for_dpo(pair))
            elif self.config.format == "rlhf":
                records.extend(format_for_rlhf(pair))
            elif self.config.format == "sft_only":
                records.append({"prompt": pair.prompt, "response": pair.chosen})
            else:
                records.append(format_for_dpo(pair))
        return records

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self, pairs: list[PreferencePair]) -> dict[str, float]:
        """Return summary statistics over a list of preference pairs."""
        n = len(pairs)
        if n == 0:
            return {"mean_score_diff": 0.0, "mean_chosen_score": 0.0, "n_pairs": 0}

        mean_score_diff = sum(p.score_diff for p in pairs) / n
        mean_chosen_score = sum(p.chosen_score for p in pairs) / n

        return {
            "mean_score_diff": mean_score_diff,
            "mean_chosen_score": mean_chosen_score,
            "n_pairs": n,
        }
