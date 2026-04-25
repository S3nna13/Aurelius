"""
vocab_builder.py
Builds a vocabulary from a raw text corpus using simple word-level tokenization.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration / entry dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VocabConfig:
    """Hyperparameters that govern how the vocabulary is built."""

    min_freq: int = 2
    max_vocab: int = 50_000
    special_tokens: list = field(
        default_factory=lambda: ["<pad>", "<unk>", "<bos>", "<eos>"]
    )


@dataclass(frozen=True)
class VocabEntry:
    """A single vocabulary entry (immutable)."""

    token: str
    token_id: int
    freq: int


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class Vocabulary:
    """Incrementally build a vocabulary from text and query it after building."""

    def __init__(self, config: Optional[VocabConfig] = None) -> None:
        self._config: VocabConfig = config if config is not None else VocabConfig()
        self._freq: Counter = Counter()
        # These are populated after build()
        self._token_to_id: dict = {}
        self._id_to_token: dict = {}
        self._built: bool = False

    # ------------------------------------------------------------------
    # Corpus ingestion
    # ------------------------------------------------------------------

    def add_text(self, text: str) -> None:
        """Tokenize *text* and increment token frequency counts."""
        tokens = re.findall(r"\w+|[^\w\s]", text.lower())
        self._freq.update(tokens)
        # Invalidate any previous build
        self._built = False

    # ------------------------------------------------------------------
    # Vocabulary construction
    # ------------------------------------------------------------------

    def build(self) -> dict:
        """Build and return the token→id mapping.

        Special tokens are assigned the lowest IDs (0, 1, 2, …).
        Regular tokens are selected from the corpus by descending frequency
        (ties broken alphabetically) and must appear >= min_freq times.
        At most max_vocab tokens total (including special tokens) are kept.
        """
        config = self._config
        token_to_id: dict = {}

        # 1) Assign special tokens
        for idx, special in enumerate(config.special_tokens):
            token_to_id[special] = idx

        # 2) Select regular tokens
        slots_available = config.max_vocab - len(config.special_tokens)
        eligible = [
            (token, freq)
            for token, freq in self._freq.items()
            if freq >= config.min_freq and token not in token_to_id
        ]
        # Sort: descending frequency, then ascending alphabetical for ties
        eligible.sort(key=lambda x: (-x[1], x[0]))
        selected = eligible[:slots_available]

        next_id = len(config.special_tokens)
        for token, _freq in selected:
            token_to_id[token] = next_id
            next_id += 1

        # 3) Build reverse mapping
        id_to_token = {v: k for k, v in token_to_id.items()}

        self._token_to_id = token_to_id
        self._id_to_token = id_to_token
        self._built = True

        return dict(token_to_id)

    # ------------------------------------------------------------------
    # Query API (requires build() to have been called)
    # ------------------------------------------------------------------

    def _ensure_built(self) -> None:
        if not self._built:
            self.build()

    def token_to_id(self, token: str) -> int:
        """Return the integer ID for *token*, or 1 (<unk>) if not in vocab."""
        self._ensure_built()
        return self._token_to_id.get(token, 1)

    def id_to_token(self, token_id: int) -> str:
        """Return the token string for *token_id*, or '<unk>' if not found."""
        self._ensure_built()
        return self._id_to_token.get(token_id, "<unk>")

    def vocab_size(self) -> int:
        """Return the number of tokens currently in the vocabulary."""
        self._ensure_built()
        return len(self._token_to_id)

    def most_common(self, n: int = 10) -> list:
        """Return the *n* most frequent non-special tokens as VocabEntry objects."""
        self._ensure_built()
        special_set = set(self._config.special_tokens)
        entries = []
        for token, token_id in self._token_to_id.items():
            if token in special_set:
                continue
            freq = self._freq.get(token, 0)
            entries.append(VocabEntry(token=token, token_id=token_id, freq=freq))
        # Sort by descending frequency, then alphabetically
        entries.sort(key=lambda e: (-e.freq, e.token))
        return entries[:n]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

VOCAB_BUILDER_REGISTRY: dict = {"default": Vocabulary}
