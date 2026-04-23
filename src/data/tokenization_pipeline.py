"""Tokenization pipeline: normalization, BPE-style chunking, padding, special tokens."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TokenizationConfig:
    max_length: int = 512
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    truncation: bool = True
    padding: bool = True


@dataclass
class TokenizedOutput:
    input_ids: List[int]
    attention_mask: List[int]
    length: int
    truncated: bool


class TokenizationPipeline:
    def __init__(self, vocab_size: int = 32000, config: Optional[TokenizationConfig] = None) -> None:
        self.vocab_size = vocab_size
        self.config = config if config is not None else TokenizationConfig()

    def normalize(self, text: str) -> str:
        """Lowercase, collapse whitespace, strip."""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def char_to_id(self, char: str) -> int:
        """Simple stub: ord(char) % vocab_size."""
        return ord(char) % self.vocab_size

    def tokenize(self, text: str) -> List[int]:
        """Normalize, prepend BOS, convert each char to char_to_id, append EOS."""
        normalized = self.normalize(text)
        ids = [self.config.bos_token_id]
        for ch in normalized:
            ids.append(self.char_to_id(ch))
        ids.append(self.config.eos_token_id)
        return ids

    def encode(self, text: str) -> TokenizedOutput:
        """Full encode with optional truncation and padding."""
        cfg = self.config
        input_ids = self.tokenize(text)
        truncated = False

        # Truncation
        if cfg.truncation and len(input_ids) > cfg.max_length:
            input_ids = input_ids[:cfg.max_length]
            input_ids[-1] = cfg.eos_token_id
            truncated = True

        # Padding
        if cfg.padding and len(input_ids) < cfg.max_length:
            pad_count = cfg.max_length - len(input_ids)
            input_ids = input_ids + [cfg.pad_token_id] * pad_count

        attention_mask = [1 if id_ != cfg.pad_token_id else 0 for id_ in input_ids]

        return TokenizedOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            length=len(input_ids),
            truncated=truncated,
        )

    def decode(self, token_ids: List[int]) -> str:
        """Filter special tokens (BOS/EOS/PAD), join chr(id % 128) for printable."""
        cfg = self.config
        special = {cfg.bos_token_id, cfg.eos_token_id, cfg.pad_token_id}
        chars = []
        for id_ in token_ids:
            if id_ in special:
                continue
            chars.append(chr(id_ % 128))
        return "".join(chars)

    def batch_encode(self, texts: List[str]) -> List[TokenizedOutput]:
        """Encode a list of texts."""
        return [self.encode(t) for t in texts]
