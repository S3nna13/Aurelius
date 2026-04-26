"""Preference data collection: generate, score, and format chosen/rejected pairs for RLHF/DPO."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PreferenceConfig:
    """Configuration for preference data collection."""

    n_responses: int = 4
    temperature: float = 0.8
    diversity_penalty: float = 0.1
    min_length: int = 10
    max_length: int = 256


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PreferencePair:
    """A single chosen/rejected preference pair."""

    prompt: str
    chosen: str
    rejected: str
    score_chosen: float
    score_rejected: float
    margin: float  # score_chosen - score_rejected
    source: str = "auto"


# ---------------------------------------------------------------------------
# Response pool
# ---------------------------------------------------------------------------


class ResponsePool:
    """Stores (text, score) response pairs and provides selection utilities."""

    def __init__(self) -> None:
        self._responses: list[tuple[str, float]] = []

    def add(self, text: str, score: float) -> None:
        """Add a (text, score) pair to the pool."""
        self._responses.append((text, score))

    def get_best_worst(self) -> tuple[str, str]:
        """Return (highest_score_text, lowest_score_text).

        If only one item, returns the same text for both.
        """
        if not self._responses:
            raise ValueError("ResponsePool is empty.")
        sorted_responses = sorted(self._responses, key=lambda x: x[1])
        worst_text = sorted_responses[0][0]
        best_text = sorted_responses[-1][0]
        return (best_text, worst_text)

    def get_diverse_pair(self, diversity_fn: Callable[[str, str], float]) -> tuple[str, str]:
        """Return (chosen, rejected) pair.

        chosen  = the response with the highest score.
        rejected = the response with the lowest score that is most different
                   (highest diversity_fn value) from chosen.

        Greedy: fix the best as chosen, then scan all others and pick the one
        with the greatest diversity score from chosen as rejected.
        """
        if len(self._responses) < 2:
            raise ValueError("Need at least 2 responses for a diverse pair.")

        sorted_responses = sorted(self._responses, key=lambda x: x[1])
        best_text, best_score = sorted_responses[-1]
        # Candidates for rejected: all others
        candidates = sorted_responses[:-1]
        rejected_text = max(candidates, key=lambda x: diversity_fn(best_text, x[0]))[0]
        return (best_text, rejected_text)

    def __len__(self) -> int:
        return len(self._responses)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_response_heuristic(prompt: str, response: str) -> float:
    """Multi-factor heuristic score in [0, 1].

    Components:
    - Length score    (weight 0.3): sigmoid((len(response) - 50) / 100)
    - Relevance score (weight 0.4): fraction of prompt words appearing in response
    - Coherence score (weight 0.3): no leading/trailing whitespace + ends with .!?
    """
    # Length component
    length_score = 1.0 / (1.0 + math.exp(-(len(response) - 50) / 100))

    # Relevance component
    prompt_words = set(prompt.lower().split())
    response_lower = response.lower()
    if prompt_words:
        matches = sum(1 for w in prompt_words if w in response_lower)
        relevance_score = min(1.0, matches / len(prompt_words))
    else:
        relevance_score = 0.0

    # Coherence component
    no_whitespace_edges = response == response.strip()
    ends_with_punct = bool(response) and response.rstrip()[-1:] in ".!?"
    coherence_score = float(no_whitespace_edges and ends_with_punct)

    return 0.3 * length_score + 0.4 * relevance_score + 0.3 * coherence_score


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_response(
    model,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    tokenizer_decode: Callable[[list[int]], str],
) -> str:
    """Temperature sampling from model, returns decoded string.

    The model is called as: loss, logits, pkv = model(input_ids)
    where input_ids has shape (1, seq_len).
    """
    generated: list[int] = list(prompt_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            input_tensor = torch.tensor([generated], dtype=torch.long)
            _loss, logits, _pkv = model(input_tensor)
            # logits shape: (batch, seq_len, vocab_size) — take last token
            next_token_logits = logits[0, -1, :]  # (vocab_size,)

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            generated.append(int(next_token))

    # Return only the newly generated part
    new_ids = generated[len(prompt_ids) :]
    return tokenizer_decode(new_ids)


# ---------------------------------------------------------------------------
# Collector class
# ---------------------------------------------------------------------------


class PreferenceCollector:
    """High-level interface for collecting and formatting preference pairs using a model."""

    def __init__(
        self,
        model,
        config: PreferenceConfig,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        score_fn: Callable[[str, str], float] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self._score_fn = score_fn if score_fn is not None else score_response_heuristic

    def collect_for_prompt(self, prompt: str) -> PreferencePair | None:
        """Generate n_responses, score each, create PreferencePair from best/worst.

        Returns None if all scores are identical (margin == 0).
        """
        prompt_ids = self.tokenizer_encode(prompt)
        pool = ResponsePool()

        for _ in range(self.config.n_responses):
            text = generate_response(
                self.model,
                prompt_ids,
                max_new_tokens=self.config.max_length,
                temperature=self.config.temperature,
                tokenizer_decode=self.tokenizer_decode,
            )
            score = self._score_fn(prompt, text)
            pool.add(text, score)

        if len(pool) < 2:
            return None

        chosen, rejected = pool.get_best_worst()
        max(s for _, s in pool._responses if _ == chosen or True)
        # Retrieve actual scores
        scores_dict: dict[str, float] = {}
        for text, score in pool._responses:
            # keep best score per text
            if text not in scores_dict or score > scores_dict[text]:
                scores_dict[text] = score

        score_chosen_val = max(pool._responses, key=lambda x: x[1])[1]
        score_rejected_val = min(pool._responses, key=lambda x: x[1])[1]
        margin = score_chosen_val - score_rejected_val

        if margin == 0:
            return None

        return PreferencePair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            score_chosen=score_chosen_val,
            score_rejected=score_rejected_val,
            margin=margin,
        )

    def collect_dataset(self, prompts: list[str]) -> list[PreferencePair]:
        """Collect preference pairs for each prompt, filtering out None results."""
        pairs: list[PreferencePair] = []
        for prompt in prompts:
            pair = self.collect_for_prompt(prompt)
            if pair is not None:
                pairs.append(pair)
        return pairs

    def filter_pairs(
        self,
        pairs: list[PreferencePair],
        min_margin: float = 0.1,
    ) -> list[PreferencePair]:
        """Keep only pairs with margin >= min_margin."""
        return [p for p in pairs if p.margin >= min_margin]

    def export_dpo_format(self, pairs: list[PreferencePair]) -> list[dict]:
        """Convert pairs to DPO-format dicts with prompt, chosen, rejected keys."""
        return [{"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected} for p in pairs]
