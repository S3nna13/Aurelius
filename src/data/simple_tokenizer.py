"""Simple tokenizer utilities — no SentencePiece, no tiktoken dependency.

Provides character-level and word-level tokenizers for testing and
lightweight usage, plus a VocabBuilder for constructing word-level
vocabularies from a corpus.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re


@dataclass
class TokenizerConfig:
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    pad_token: str = "<pad>"


class CharTokenizer:
    """Simple character-level tokenizer for testing.

    Vocabulary: all printable ASCII + special tokens.
    """

    def __init__(self, cfg: TokenizerConfig | None = None) -> None:
        self.cfg = cfg or TokenizerConfig()
        # Build vocab: special tokens + printable ASCII (32-126) + newline/tab
        specials = [
            self.cfg.pad_token,
            self.cfg.unk_token,
            self.cfg.bos_token,
            self.cfg.eos_token,
        ]
        chars = [chr(i) for i in range(32, 127)] + ["\n", "\t"]
        vocab_tokens = specials + chars
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(vocab_tokens)}
        self.id_to_token: dict[int, str] = {i: t for i, t in enumerate(vocab_tokens)}
        self.unk_id: int = self.token_to_id.get("<unk>", 1)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Encode text to list of token IDs."""
        ids: list[int] = []
        if add_bos and self.cfg.bos_token in self.token_to_id:
            ids.append(self.token_to_id[self.cfg.bos_token])
        for ch in text:
            ids.append(self.token_to_id.get(ch, self.unk_id))
        if add_eos and self.cfg.eos_token in self.token_to_id:
            ids.append(self.token_to_id[self.cfg.eos_token])
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token IDs to string."""
        special_ids: set[int] = set()
        if skip_special:
            for st in [self.cfg.pad_token, self.cfg.unk_token, self.cfg.bos_token, self.cfg.eos_token]:
                if st in self.token_to_id:
                    special_ids.add(self.token_to_id[st])
        chars: list[str] = []
        for i in ids:
            if i in special_ids:
                continue
            chars.append(self.id_to_token.get(i, ""))
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def __len__(self) -> int:
        return self.vocab_size


class SimpleTokenizer:
    """Tokenizer that uses a pre-built word-level vocabulary.

    Splits text on whitespace+punctuation, looks up each token.
    Unknown tokens map to UNK.
    """

    def __init__(self, vocab: dict[str, int], cfg: TokenizerConfig | None = None) -> None:
        self.cfg = cfg or TokenizerConfig()
        self.token_to_id: dict[str, int] = vocab
        self.id_to_token: dict[int, str] = {v: k for k, v in vocab.items()}
        self.unk_id: int = vocab.get(self.cfg.unk_token, 0)
        self.bos_id: int | None = vocab.get(self.cfg.bos_token, None)
        self.eos_id: int | None = vocab.get(self.cfg.eos_token, None)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """Tokenize by splitting on whitespace/punctuation."""
        tokens = _tokenize_words(text)
        ids: list[int] = []
        if add_bos and self.bos_id is not None:
            ids.append(self.bos_id)
        for t in tokens:
            ids.append(self.token_to_id.get(t, self.unk_id))
        if add_eos and self.eos_id is not None:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token IDs back to a space-joined string."""
        special_ids: set[int] = set()
        if skip_special:
            for st in [self.cfg.pad_token, self.cfg.unk_token, self.cfg.bos_token, self.cfg.eos_token]:
                tid = self.token_to_id.get(st)
                if tid is not None:
                    special_ids.add(tid)
        tokens: list[str] = []
        for i in ids:
            if i in special_ids:
                continue
            tok = self.id_to_token.get(i, "")
            if tok:
                tokens.append(tok)
        return " ".join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def __len__(self) -> int:
        return self.vocab_size


class VocabBuilder:
    """Build a word-level vocabulary from a corpus.

    Counts token frequencies and keeps top-N most frequent.
    """

    def __init__(self, max_vocab_size: int = 10000, min_freq: int = 2) -> None:
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self._counter: Counter[str] = Counter()

    def update(self, text: str) -> None:
        """Add text to corpus for vocabulary building."""
        tokens = _tokenize_words(text)
        self._counter.update(tokens)

    def build(self, cfg: TokenizerConfig | None = None) -> dict[str, int]:
        """Build and return token_to_id dict."""
        cfg = cfg or TokenizerConfig()
        # Special tokens first
        specials = [cfg.pad_token, cfg.unk_token, cfg.bos_token, cfg.eos_token]
        vocab: dict[str, int] = {t: i for i, t in enumerate(specials)}

        # Add frequent words
        frequent = [(w, c) for w, c in self._counter.most_common() if c >= self.min_freq]
        for word, _ in frequent[: self.max_vocab_size - len(specials)]:
            if word not in vocab:
                vocab[word] = len(vocab)
        return vocab

    def build_tokenizer(self, cfg: TokenizerConfig | None = None) -> SimpleTokenizer:
        """Build a SimpleTokenizer from the accumulated corpus."""
        vocab = self.build(cfg)
        return SimpleTokenizer(vocab, cfg)


def _tokenize_words(text: str) -> list[str]:
    """Split text into word tokens (lowercase, split on non-alphanumeric)."""
    return re.findall(r"\b\w+\b", text.lower())
