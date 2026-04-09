"""Byte-level BPE tokenizer: learn merge rules from text, encode/decode strings to/from token ids."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenizerConfig:
    """Configuration for the BPE tokenizer."""

    vocab_size: int = 256
    min_frequency: int = 2
    max_merges: int = 1000
    special_tokens: list[str] = field(
        default_factory=lambda: ["<pad>", "<bos>", "<eos>", "<unk>"]
    )


def get_byte_vocab() -> dict[int, bytes]:
    """Return base vocabulary: 256 byte values {0: b'\\x00', 1: b'\\x01', ..., 255: b'\\xff'}."""
    return {i: bytes([i]) for i in range(256)}


def count_pairs(token_sequences: list[list[int]]) -> dict[tuple[int, int], int]:
    """Count all adjacent token pairs across all sequences."""
    counts: dict[tuple[int, int], int] = {}
    for seq in token_sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(
    token_sequences: list[list[int]], pair: tuple[int, int], new_id: int
) -> list[list[int]]:
    """Replace all occurrences of *pair* with *new_id* in all sequences."""
    result: list[list[int]] = []
    for seq in token_sequences:
        new_seq: list[int] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                new_seq.append(new_id)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        result.append(new_seq)
    return result


class BPETokenizer:
    """Byte-level BPE tokenizer."""

    def __init__(self, config: TokenizerConfig) -> None:
        self.config = config
        self.merges: list[tuple[int, int]] = []
        self.vocab: dict[int, bytes] = get_byte_vocab()
        self.special_tokens: dict[str, int] = {}

    def train(self, texts: list[str]) -> None:
        """Learn merge rules from *texts*."""
        # Convert texts to byte sequences
        sequences: list[list[int]] = [list(t.encode("utf-8")) for t in texts]

        num_merges = 0
        next_id = 256

        while num_merges < self.config.max_merges:
            # Check if we've reached the target vocab size (excluding special tokens)
            if next_id >= self.config.vocab_size:
                break

            pairs = count_pairs(sequences)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.__getitem__)
            if pairs[best_pair] < self.config.min_frequency:
                break

            sequences = merge_pair(sequences, best_pair, next_id)
            self.merges.append(best_pair)
            self.vocab[next_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            next_id += 1
            num_merges += 1

        # Assign special tokens after last merge id
        for token_str in self.config.special_tokens:
            self.special_tokens[token_str] = next_id
            self.vocab[next_id] = token_str.encode("utf-8")
            next_id += 1

    def encode(self, text: str) -> list[int]:
        """Encode *text* to a list of token ids."""
        ids = list(text.encode("utf-8"))
        for pair in self.merges:
            new_id = 256 + self.merges.index(pair)
            # Apply this merge rule
            new_ids: list[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token *ids* back to a string."""
        special_ids = set(self.special_tokens.values())
        parts: list[bytes] = []
        for token_id in ids:
            if token_id in special_ids:
                continue
            parts.append(self.vocab[token_id])
        return b"".join(parts).decode("utf-8", errors="replace")

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode multiple texts."""
        return [self.encode(t) for t in texts]

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size (base 256 + merges + special tokens)."""
        return 256 + len(self.merges) + len(self.special_tokens)

    def get_special_token_id(self, token: str) -> int:
        """Return id of special token; raises KeyError if not found."""
        return self.special_tokens[token]


def train_tokenizer(texts: list[str], config: TokenizerConfig) -> BPETokenizer:
    """Convenience function: create tokenizer, train, return."""
    tokenizer = BPETokenizer(config)
    tokenizer.train(texts)
    return tokenizer
