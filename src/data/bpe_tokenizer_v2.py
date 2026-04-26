"""Byte Pair Encoding (BPE) tokenizer — pure Python stdlib, no external deps.

Implements the core BPE algorithm used by GPT-style tokenizers:
  - ByteVocabulary: 256 base byte tokens
  - BPEMergeRule: a single (pair -> new_token) merge rule
  - BPETrainer: learns merge rules from a corpus
  - BPEVocabulary: full token <-> bytes mapping
  - BPETokenizer: encode/decode API with save/load
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# ByteVocabulary
# ---------------------------------------------------------------------------


class ByteVocabulary:
    """Initial byte-level vocabulary with 256 single-byte tokens (0-255)."""

    def __init__(self) -> None:
        # token_id -> bytes representation (length 1 for base vocab)
        self._id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # ------------------------------------------------------------------
    def to_bytes(self, text: str) -> list[int]:
        """UTF-8 encode *text* and return list of byte values (0-255 each)."""
        return list(text.encode("utf-8"))

    def from_bytes(self, byte_ids: list[int]) -> str:
        """Decode a list of byte values back to a string."""
        return bytes(byte_ids).decode("utf-8")

    def vocab_size(self) -> int:
        return 256


# ---------------------------------------------------------------------------
# BPEMergeRule
# ---------------------------------------------------------------------------


@dataclass
class BPEMergeRule:
    """A single BPE merge rule: (left, right) -> merged_token."""

    pair: tuple[int, int]
    merged_token: int
    frequency: int

    def __repr__(self) -> str:
        return f"BPEMergeRule({self.pair} -> {self.merged_token}, freq={self.frequency})"


# ---------------------------------------------------------------------------
# BPETrainer
# ---------------------------------------------------------------------------


class BPETrainer:
    """Learns BPE merge rules from a text corpus."""

    def __init__(self, vocab_size: int = 512) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (base byte vocabulary)")
        self.vocab_size = vocab_size

    # ------------------------------------------------------------------
    def _count_pairs(self, token_sequences: list[list[int]]) -> dict[tuple[int, int], int]:
        """Count all adjacent pairs across all sequences."""
        counts: dict[tuple[int, int], int] = {}
        for seq in token_sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    # ------------------------------------------------------------------
    def _merge(
        self,
        token_sequences: list[list[int]],
        pair: tuple[int, int],
        new_token: int,
    ) -> list[list[int]]:
        """Replace every occurrence of *pair* with *new_token* in all sequences."""
        left, right = pair
        result: list[list[int]] = []
        for seq in token_sequences:
            new_seq: list[int] = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == left and seq[i + 1] == right:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            result.append(new_seq)
        return result

    # ------------------------------------------------------------------
    def train(self, texts: list[str]) -> list[BPEMergeRule]:
        """Learn BPE merges from *texts*; stop when vocab_size is reached."""
        bv = ByteVocabulary()
        token_sequences: list[list[int]] = [bv.to_bytes(t) for t in texts]

        merge_rules: list[BPEMergeRule] = []
        next_token_id = 256  # first merged token id

        while next_token_id < self.vocab_size:
            pair_counts = self._count_pairs(token_sequences)
            if not pair_counts:
                break
            # Pick the most frequent pair; break ties by pair value for determinism
            best_pair = max(pair_counts, key=lambda p: (pair_counts[p], -p[0], -p[1]))
            best_freq = pair_counts[best_pair]
            if best_freq < 1:
                break

            rule = BPEMergeRule(
                pair=best_pair,
                merged_token=next_token_id,
                frequency=best_freq,
            )
            merge_rules.append(rule)
            token_sequences = self._merge(token_sequences, best_pair, next_token_id)
            next_token_id += 1

        return merge_rules


# ---------------------------------------------------------------------------
# BPEVocabulary
# ---------------------------------------------------------------------------


class BPEVocabulary:
    """Full token-to-bytes mapping: 256 base bytes + learned merges."""

    def __init__(
        self,
        merge_rules: list[BPEMergeRule],
        vocab: dict[int, bytes] | None = None,
    ) -> None:
        self._merge_rules = merge_rules

        if vocab is not None:
            self._id_to_bytes: dict[int, bytes] = dict(vocab)
        else:
            # Build from base + merges
            self._id_to_bytes = {i: bytes([i]) for i in range(256)}
            for rule in merge_rules:
                left_bytes = self._id_to_bytes[rule.pair[0]]
                right_bytes = self._id_to_bytes[rule.pair[1]]
                self._id_to_bytes[rule.merged_token] = left_bytes + right_bytes

        # Special tokens appended after merges
        self._special: dict[str, int] = {}

    # ------------------------------------------------------------------
    def decode_token(self, token_id: int) -> bytes:
        """Return the byte string for *token_id*."""
        if token_id not in self._id_to_bytes:
            raise KeyError(f"Unknown token id: {token_id}")
        return self._id_to_bytes[token_id]

    def encode_special(self, special_tokens: list[str]) -> dict[str, int]:
        """Add special tokens to the vocab, assigning new ids."""
        next_id = max(self._id_to_bytes.keys()) + 1
        mapping: dict[str, int] = {}
        for tok in special_tokens:
            encoded = tok.encode("utf-8")
            self._id_to_bytes[next_id] = encoded
            self._special[tok] = next_id
            mapping[tok] = next_id
            next_id += 1
        return mapping

    def vocab_size(self) -> int:
        return len(self._id_to_bytes)


# ---------------------------------------------------------------------------
# BPETokenizer
# ---------------------------------------------------------------------------


class BPETokenizer:
    """Full BPE tokenizer: train → encode → decode, with JSON persistence."""

    def __init__(self, vocab_size: int = 512) -> None:
        self.vocab_size = vocab_size
        self._trainer = BPETrainer(vocab_size=vocab_size)
        self._merge_rules: list[BPEMergeRule] = []
        self._bv = ByteVocabulary()
        self._vocabulary: BPEVocabulary | None = None
        # Ordered list of (pair, new_token) for encoding
        self._pair_to_token: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------
    def train(self, texts: list[str]) -> None:
        """Train BPE on *texts*."""
        self._merge_rules = self._trainer.train(texts)
        self._vocabulary = BPEVocabulary(self._merge_rules)
        self._pair_to_token = {rule.pair: rule.merged_token for rule in self._merge_rules}

    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        """Tokenize *text* into token ids using learned BPE merges."""
        if not self._merge_rules:
            # No training done yet — fall back to raw bytes
            return self._bv.to_bytes(text)

        tokens: list[int] = self._bv.to_bytes(text)

        # Apply merge rules in training order (greedy left-to-right per rule)
        for rule in self._merge_rules:
            pair = rule.pair
            new_token = rule.merged_token
            tokens = self._apply_single_merge(tokens, pair, new_token)

        return tokens

    @staticmethod
    def _apply_single_merge(tokens: list[int], pair: tuple[int, int], new_token: int) -> list[int]:
        left, right = pair
        result: list[int] = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == left and tokens[i + 1] == right:
                result.append(new_token)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    # ------------------------------------------------------------------
    def decode(self, token_ids: list[int]) -> str:
        """Convert token ids back to a UTF-8 string."""
        if not token_ids:
            return ""
        if self._vocabulary is None:
            # No vocabulary yet — treat ids as raw bytes
            return bytes(token_ids).decode("utf-8")
        raw = b"".join(self._vocabulary.decode_token(tid) for tid in token_ids)
        return raw.decode("utf-8")

    # ------------------------------------------------------------------
    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    # ------------------------------------------------------------------
    def save_vocab(self, path: str) -> None:
        """Serialise merge rules to JSON."""
        data = {
            "vocab_size": self.vocab_size,
            "merge_rules": [
                {
                    "pair": list(rule.pair),
                    "merged_token": rule.merged_token,
                    "frequency": rule.frequency,
                }
                for rule in self._merge_rules
            ],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    def load_vocab(self, path: str) -> None:
        """Load merge rules from JSON produced by save_vocab."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self.vocab_size = data["vocab_size"]
        self._merge_rules = [
            BPEMergeRule(
                pair=(entry["pair"][0], entry["pair"][1]),
                merged_token=entry["merged_token"],
                frequency=entry["frequency"],
            )
            for entry in data["merge_rules"]
        ]
        self._vocabulary = BPEVocabulary(self._merge_rules)
        self._pair_to_token = {rule.pair: rule.merged_token for rule in self._merge_rules}

    # ------------------------------------------------------------------
    def compression_ratio(self, text: str) -> float:
        """Ratio of UTF-8 bytes to BPE tokens (>= 1.0 means BPE compresses)."""
        byte_len = len(text.encode("utf-8"))
        token_len = len(self.encode(text))
        if token_len == 0:
            return 1.0
        return byte_len / token_len
