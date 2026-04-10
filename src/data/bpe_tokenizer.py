"""Pure-Python byte-pair encoding (BPE) tokenizer — from scratch.

Implements merge learning, encoding, decoding, and JSON serialization
using only the Python standard library.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class BPEConfig:
    """Configuration for the BPE tokenizer."""

    vocab_size: int = 1000
    min_frequency: int = 2
    special_tokens: list[str] = field(
        default_factory=lambda: ["<pad>", "<unk>", "<s>", "</s>"]
    )


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def get_byte_pairs(
    word_tokens: list[tuple[str, ...]],
) -> dict[tuple[str, str], int]:
    """Count all adjacent byte-pair frequencies across a list of tokenized words.

    Each element of *word_tokens* is a tuple of string symbols representing
    one word.  The function returns a mapping from (left, right) symbol pairs
    to their total occurrence count.

    Parameters
    ----------
    word_tokens:
        A list of tuples, where each tuple contains the current sub-word
        symbols for one word.

    Returns
    -------
    dict[tuple[str, str], int]
        Mapping of adjacent pair to count.
    """
    pairs: dict[tuple[str, str], int] = defaultdict(int)
    for symbols in word_tokens:
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += 1
    return dict(pairs)


def merge_vocab(
    vocab: dict[tuple[str, ...], int],
    pair: tuple[str, str],
) -> dict[tuple[str, ...], int]:
    """Return a new vocab where every occurrence of *pair* has been merged.

    Parameters
    ----------
    vocab:
        Mapping from symbol-tuple (word representation) to word frequency.
    pair:
        The adjacent symbol pair ``(left, right)`` to merge into a single
        symbol ``left + right``.

    Returns
    -------
    dict[tuple[str, ...], int]
        Updated vocabulary with the pair merged everywhere.
    """
    merged: dict[tuple[str, ...], int] = {}
    left, right = pair
    merged_sym = left + right

    for symbols, freq in vocab.items():
        new_symbols: list[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == left and symbols[i + 1] == right:
                new_symbols.append(merged_sym)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        merged[tuple(new_symbols)] = freq

    return merged


# ---------------------------------------------------------------------------
# BPETokenizer
# ---------------------------------------------------------------------------


class BPETokenizer:
    """Learn and apply byte-pair encoding (BPE) merges.

    The tokenizer is byte-level: every input string is first converted to its
    raw UTF-8 bytes (represented as single-character strings via
    ``chr(b)``), then BPE merges are applied on top of that byte vocabulary.

    Parameters
    ----------
    config:
        A :class:`BPEConfig` instance controlling vocab size, minimum merge
        frequency, and the list of special tokens.
    """

    def __init__(self, config: BPEConfig | None = None) -> None:
        self.config: BPEConfig = config if config is not None else BPEConfig()

        # token string  ->  int id
        self._token_to_id: dict[str, int] = {}
        # int id  ->  token string
        self._id_to_token: dict[int, str] = {}
        # ordered list of merges learned during training: (left, right)
        self._merges: list[tuple[str, str]] = []
        # fast lookup: pair -> merged symbol
        self._merge_map: dict[tuple[str, str], str] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, texts: list[str]) -> None:
        """Learn BPE merges from *texts*.

        Steps:
        1. Build the initial byte-level vocabulary (256 bytes as single-char
           strings, i.e. ``chr(0)`` … ``chr(255)``).
        2. Add special tokens on top of that.
        3. Tokenize each text into word-frequency counts, where each word is
           represented as a tuple of byte symbols.
        4. Greedily merge the most-frequent adjacent pair until
           ``vocab_size`` is reached or no pair meets ``min_frequency``.

        Parameters
        ----------
        texts:
            List of raw training strings.
        """
        # ---- Step 1: byte-level base vocab --------------------------------
        self._token_to_id = {}
        self._id_to_token = {}
        self._merges = []
        self._merge_map = {}

        for byte_val in range(256):
            sym = chr(byte_val)
            tid = len(self._token_to_id)
            self._token_to_id[sym] = tid
            self._id_to_token[tid] = sym

        # ---- Step 2: special tokens ----------------------------------------
        for sp in self.config.special_tokens:
            if sp not in self._token_to_id:
                tid = len(self._token_to_id)
                self._token_to_id[sp] = tid
                self._id_to_token[tid] = sp

        # ---- Step 3: build word-frequency vocab ----------------------------
        # Split texts on whitespace; each "word" becomes a tuple of byte syms.
        word_freq: dict[tuple[str, ...], int] = defaultdict(int)
        for text in texts:
            for word in re.split(r"(\s+)", text):
                if not word:
                    continue
                symbols = tuple(chr(b) for b in word.encode("utf-8"))
                word_freq[symbols] += 1

        word_freq = dict(word_freq)

        # ---- Step 4: iterative merge --------------------------------------
        target = self.config.vocab_size
        while len(self._token_to_id) < target:
            # Compute pair frequencies weighted by word frequency
            pair_counts: dict[tuple[str, str], int] = defaultdict(int)
            for symbols, freq in word_freq.items():
                for i in range(len(symbols) - 1):
                    pair_counts[(symbols[i], symbols[i + 1])] += freq

            if not pair_counts:
                break

            best_pair, best_count = max(pair_counts.items(), key=lambda x: x[1])

            if best_count < self.config.min_frequency:
                break

            # Create new merged symbol
            new_sym = best_pair[0] + best_pair[1]

            # Add to vocab
            if new_sym not in self._token_to_id:
                tid = len(self._token_to_id)
                self._token_to_id[new_sym] = tid
                self._id_to_token[tid] = new_sym

            # Record merge
            self._merges.append(best_pair)
            self._merge_map[best_pair] = new_sym

            # Apply merge to word_freq
            word_freq = merge_vocab(word_freq, best_pair)

    # ------------------------------------------------------------------
    # Encoding / Decoding
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode *text* to a list of token IDs.

        The text is first split into byte symbols, then all learned BPE merges
        are applied in training order.  Any byte that cannot be represented is
        replaced with the ``<unk>`` token ID.

        Parameters
        ----------
        text:
            Raw string to encode.

        Returns
        -------
        list[int]
            Sequence of integer token IDs.
        """
        if not text:
            return []

        unk_id = self._token_to_id.get("<unk>", 0)

        # Split on whitespace, preserving whitespace tokens, and encode each
        # fragment separately so we don't merge across word boundaries.
        fragments = re.split(r"(\s+)", text)
        all_ids: list[int] = []

        for fragment in fragments:
            if not fragment:
                continue
            # Convert to byte symbols
            try:
                byte_symbols: list[str] = [chr(b) for b in fragment.encode("utf-8")]
            except Exception:
                all_ids.append(unk_id)
                continue

            # Apply merges greedily in order
            symbols = byte_symbols
            changed = True
            while changed:
                changed = False
                new_syms: list[str] = []
                i = 0
                while i < len(symbols):
                    if i < len(symbols) - 1:
                        pair = (symbols[i], symbols[i + 1])
                        merged = self._merge_map.get(pair)
                        if merged is not None:
                            new_syms.append(merged)
                            i += 2
                            changed = True
                            continue
                    new_syms.append(symbols[i])
                    i += 1
                symbols = new_syms

            # Map symbols to IDs
            for sym in symbols:
                tid = self._token_to_id.get(sym)
                if tid is None:
                    all_ids.append(unk_id)
                else:
                    all_ids.append(tid)

        return all_ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string.

        Tokens are joined and then interpreted as UTF-8 byte values (using the
        ``chr`` representation stored in the vocabulary).  Special tokens are
        emitted as their string representation.

        Parameters
        ----------
        ids:
            Sequence of integer token IDs to decode.

        Returns
        -------
        str
            Reconstructed text.
        """
        if not ids:
            return ""

        special_set = set(self.config.special_tokens)
        byte_values: list[int] = []
        result_parts: list[str] = []

        def flush_bytes() -> None:
            if byte_values:
                result_parts.append(
                    bytes(byte_values).decode("utf-8", errors="replace")
                )
                byte_values.clear()

        for tid in ids:
            token = self._id_to_token.get(tid)
            if token is None:
                continue
            if token in special_set:
                flush_bytes()
                result_parts.append(token)
            else:
                # token is a multi-char merged symbol; break into byte ints
                for ch in token:
                    byte_values.append(ord(ch))

        flush_bytes()
        return "".join(result_parts)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Current number of tokens in the vocabulary."""
        return len(self._token_to_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the tokenizer to a JSON file at *path*.

        The JSON contains:
        - ``merges``: list of ``["left", "right"]`` pairs in training order.
        - ``vocab``: mapping from token string to int ID.
        - ``config``: the :class:`BPEConfig` fields.

        Parameters
        ----------
        path:
            Destination file path (will be created or overwritten).
        """
        data = {
            "config": {
                "vocab_size": self.config.vocab_size,
                "min_frequency": self.config.min_frequency,
                "special_tokens": self.config.special_tokens,
            },
            "merges": [[left, right] for left, right in self._merges],
            "vocab": {token: tid for token, tid in self._token_to_id.items()},
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a tokenizer from a JSON file previously saved with :meth:`save`.

        Parameters
        ----------
        path:
            Path to the JSON file.

        Returns
        -------
        BPETokenizer
            Fully restored tokenizer ready for encoding/decoding.
        """
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        cfg_data = raw["config"]
        config = BPEConfig(
            vocab_size=cfg_data["vocab_size"],
            min_frequency=cfg_data["min_frequency"],
            special_tokens=cfg_data["special_tokens"],
        )

        tok = cls(config)

        # Restore vocab
        tok._token_to_id = {token: int(tid) for token, tid in raw["vocab"].items()}
        tok._id_to_token = {int(tid): token for token, tid in raw["vocab"].items()}

        # Restore merges
        tok._merges = [(left, right) for left, right in raw["merges"]]
        tok._merge_map = {
            (left, right): left + right for left, right in tok._merges
        }

        return tok
