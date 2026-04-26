"""Preference data synthesizer for DPO/RLHF training pairs.

Parses real dataset schemas (Nectar, OpenHermes 2.5, Argilla Distilabel) and
provides heuristic-based pair generation from raw instruction data.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NectarSample:
    """Sample from berkeley-nest/Nectar dataset."""

    id: str
    prompt: str
    answers: list[dict]  # {answer, model, rank, score}

    def best_answer(self) -> str:
        """Return answer with rank=1, or highest score if no rank present."""
        if not self.answers:
            return ""
        # prefer rank=1
        rank_one = [a for a in self.answers if a.get("rank") == 1]
        if rank_one:
            return rank_one[0]["answer"]
        # fallback: highest score
        return max(self.answers, key=lambda a: a.get("score", 0.0))["answer"]

    def worst_answer(self) -> str:
        """Return answer with the highest rank number (worst ranking)."""
        if not self.answers:
            return ""
        # highest rank number = worst
        ranked = [a for a in self.answers if "rank" in a]
        if ranked:
            return max(ranked, key=lambda a: a["rank"])["answer"]
        # fallback: lowest score
        return min(self.answers, key=lambda a: a.get("score", 0.0))["answer"]

    def to_preference_pair(self) -> dict:
        """Return {'prompt': ..., 'chosen': best_answer, 'rejected': worst_answer}."""
        return {
            "prompt": self.prompt,
            "chosen": self.best_answer(),
            "rejected": self.worst_answer(),
        }


@dataclass
class OpenHermesSample:
    """Sample from teknium/OpenHermes-2.5 dataset."""

    system_prompt: str
    conversations: list[dict]  # [{'from': 'human'/'gpt', 'value': str}]
    category: str
    model: str

    def to_instruction(self) -> dict:
        """Extract last human turn as instruction and last gpt turn as output."""
        instruction = ""
        output = ""
        for turn in self.conversations:
            src = turn.get("from", "")
            val = turn.get("value", "")
            if src == "human":
                instruction = val
            elif src == "gpt":
                output = val
        return {
            "system": self.system_prompt,
            "instruction": instruction,
            "output": output,
        }


@dataclass
class ArgillaSample:
    """Sample from argilla/distilabel-intel-orca-dpo-pairs dataset."""

    system: str
    question: str
    chosen: str
    rejected: str
    chosen_rating: float
    rejected_rating: float


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_nectar_sample(raw: dict) -> NectarSample:
    """Parse a raw dict into a NectarSample."""
    return NectarSample(
        id=raw.get("id", ""),
        prompt=raw.get("prompt", ""),
        answers=raw.get("answers", []),
    )


def parse_openhermes_sample(raw: dict) -> OpenHermesSample:
    """Parse a raw dict into an OpenHermesSample."""
    return OpenHermesSample(
        system_prompt=raw.get("system_prompt", ""),
        conversations=raw.get("conversations", []),
        category=raw.get("category", ""),
        model=raw.get("model", ""),
    )


def parse_argilla_sample(raw: dict) -> ArgillaSample:
    """Parse a raw dict into an ArgillaSample."""
    chosen_rating = float(raw.get("raw_chosen", {}).get("rating", 0.0))
    rejected_rating = float(raw.get("raw_rejected", {}).get("rating", 0.0))
    return ArgillaSample(
        system=raw.get("system", ""),
        question=raw.get("question", ""),
        chosen=raw.get("chosen", ""),
        rejected=raw.get("rejected", ""),
        chosen_rating=chosen_rating,
        rejected_rating=rejected_rating,
    )


# ---------------------------------------------------------------------------
# Synthesizer
# ---------------------------------------------------------------------------


class PreferenceSynthesizer:
    """Synthesize preference pairs from instruction data using scoring heuristics."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def score_response(self, response: str) -> float:
        """Heuristic quality score for a response.

        Factors:
          - Length: ideal 100–500 words (rough proxy for token count)
          - Contains markdown structure (headers, bullets, code blocks)
          - No excessive repetition
          - Starts with a capital letter

        Returns float in [0, 1].
        """
        if not response:
            return 0.0

        score = 0.0

        # ---- length factor (0–0.4) ----
        words = response.split()
        n_words = len(words)
        if 100 <= n_words <= 500:
            length_score = 0.4
        elif n_words < 100:
            length_score = 0.4 * (n_words / 100)
        else:
            # penalise beyond 500 gradually
            length_score = max(0.0, 0.4 * (1.0 - (n_words - 500) / 1000))
        score += length_score

        # ---- markdown structure (0–0.3) ----
        md_patterns = [
            r"^#{1,6}\s",  # headers
            r"^\s*[-*+]\s",  # bullet lists
            r"```",  # code blocks
            r"^\s*\d+\.\s",  # numbered lists
        ]
        md_hits = sum(1 for pat in md_patterns if re.search(pat, response, re.MULTILINE))
        score += min(0.3, md_hits * 0.1)

        # ---- no repetition (0–0.2) ----
        if n_words > 0:
            unique_ratio = len(set(w.lower() for w in words)) / n_words
            score += 0.2 * unique_ratio

        # ---- starts with capital letter (0–0.1) ----
        stripped = response.lstrip()
        if stripped and stripped[0].isupper():
            score += 0.1

        return min(1.0, score)

    def create_pair_from_instructions(
        self,
        samples: list[dict],  # list of {'instruction', 'input', 'output'}
        n_pairs: int = 4,
    ) -> list[dict]:
        """Create preference pairs by scoring outputs and pairing high vs low.

        Steps:
          1. Score all outputs.
          2. Sort by score descending.
          3. Pair top-half items with bottom-half items.

        Returns list of {'prompt': str, 'chosen': str, 'rejected': str}.
        """
        if not samples:
            return []

        scored = []
        for s in samples:
            instruction = s.get("instruction", "")
            inp = s.get("input", "")
            output = s.get("output", "")
            prompt = f"{instruction}\n{inp}".strip() if inp else instruction
            sc = self.score_response(output)
            scored.append({"prompt": prompt, "output": output, "score": sc})

        scored.sort(key=lambda x: x["score"], reverse=True)

        n = len(scored)
        pairs: list[dict] = []
        mid = max(1, n // 2)

        high = scored[:mid]
        low = scored[mid:] if n > 1 else scored  # fallback: pair with self

        # shuffle low pool so pairs are varied
        self.rng.shuffle(low)

        target = min(n_pairs, max(len(high), 1))
        for i in range(target):
            h = high[i % len(high)]
            lo = low[i % len(low)]
            pairs.append(
                {
                    "prompt": h["prompt"],
                    "chosen": h["output"],
                    "rejected": lo["output"],
                }
            )

        return pairs

    def augment_with_negatives(
        self,
        pairs: list[dict],
        strategies: list[str] = ("truncate", "repeat", "shuffle_words"),
    ) -> list[dict]:
        """Create additional rejected responses by corrupting chosen responses.

        Strategies:
          - truncate: keep first 30% of characters
          - repeat: repeat first sentence 3 times
          - shuffle_words: shuffle words in first sentence

        Returns new list of pairs with corrupted rejected versions (same length).
        """
        augmented: list[dict] = []
        strategies = list(strategies)

        for pair in pairs:
            chosen = pair["chosen"]
            strategy = self.rng.choice(strategies)
            rejected = self._corrupt(chosen, strategy)
            augmented.append(
                {
                    "prompt": pair["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

        return augmented

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _corrupt(self, text: str, strategy: str) -> str:
        if strategy == "truncate":
            cutoff = max(1, int(len(text) * 0.3))
            return text[:cutoff]

        elif strategy == "repeat":
            # find first sentence boundary
            match = re.search(r"[.!?]", text)
            if match:
                first = text[: match.end()].strip()
            else:
                first = text.split()[0] if text.split() else text
            return (first + " ") * 3

        elif strategy == "shuffle_words":
            match = re.search(r"[.!?]", text)
            if match:
                first = text[: match.end()]
                rest = text[match.end() :]
            else:
                first = text
                rest = ""
            words = first.split()
            self.rng.shuffle(words)
            shuffled = " ".join(words)
            return (shuffled + rest).strip()

        # unknown strategy: return as-is
        return text


# ---------------------------------------------------------------------------
# Mock data generators
# ---------------------------------------------------------------------------


def mock_nectar_data(n: int = 4) -> list[dict]:
    """Return n mock Nectar raw dicts."""
    data = []
    for i in range(n):
        data.append(
            {
                "id": f"alpaca_{1000 + i}",
                "prompt": f"Explain concept number {i} in detail.",
                "answers": [
                    {
                        "answer": f"This is a thorough explanation of concept {i}. "
                        "It covers the main ideas, provides examples, and "
                        "concludes with a summary.",
                        "model": "gpt-4",
                        "rank": 1,
                        "score": 4.5,
                    },
                    {
                        "answer": f"Concept {i} is something.",
                        "model": "claude",
                        "rank": 2,
                        "score": 3.2,
                    },
                ],
            }
        )
    return data


def mock_openhermes_data(n: int = 4) -> list[dict]:
    """Return n mock OpenHermes 2.5 raw dicts."""
    data = []
    for i in range(n):
        data.append(
            {
                "system_prompt": "You are a helpful assistant.",
                "conversations": [
                    {"from": "human", "value": f"What is topic {i}?"},
                    {
                        "from": "gpt",
                        "value": f"Topic {i} refers to an important subject. "
                        "Here is a detailed explanation with examples.",
                    },
                ],
                "category": "general",
                "model": "gpt-4",
                "source": "sharegpt",
            }
        )
    return data


def mock_argilla_data(n: int = 4) -> list[dict]:
    """Return n mock Argilla distilabel raw dicts."""
    data = []
    for i in range(n):
        data.append(
            {
                "system": "You are a helpful assistant.",
                "question": f"Please explain item {i}.",
                "chosen": f"Item {i} is well understood. It has many properties "
                "and applications that are worth exploring in depth.",
                "rejected": f"Item {i} exists.",
                "raw_chosen": {"justification": "Comprehensive answer.", "rating": 5},
                "raw_rejected": {"justification": "Too brief.", "rating": 2},
            }
        )
    return data
