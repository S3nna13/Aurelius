"""Format Aurelius model outputs for display."""

import re
import textwrap
from typing import List, Tuple

_SPECIAL_TOKENS = re.compile(r"<\|(?:system|user|assistant|end)\|>")
_CODE_FENCE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)


def strip_special_tokens(text: str) -> str:
    """Remove model special tokens from text."""
    return _SPECIAL_TOKENS.sub("", text)


def extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """Return list of (language, code) tuples for every fenced code block."""
    return [(m.group(1), m.group(2)) for m in _CODE_FENCE.finditer(text)]


def format_for_terminal(text: str, width: int = 80) -> str:
    """Strip special tokens and wrap long lines to the given width."""
    cleaned = strip_special_tokens(text)
    lines = []
    for line in cleaned.splitlines():
        if len(line) > width:
            lines.extend(textwrap.wrap(line, width=width) or [""])
        else:
            lines.append(line)
    return "\n".join(lines)


class ResponseFormatter:
    """Format model response text for downstream display or logging."""

    def __init__(self, max_length: int = 2048, strip_tokens: bool = True) -> None:
        self.max_length = max_length
        self.strip_tokens = strip_tokens

    def format(self, text: str) -> str:
        if self.strip_tokens:
            text = strip_special_tokens(text)
        return text[: self.max_length]

    def has_code(self, text: str) -> bool:
        return "```" in text

    def word_count(self, text: str) -> int:
        return len(text.split())

    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)
