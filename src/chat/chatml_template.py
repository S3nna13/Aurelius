"""ChatML template encoder/decoder for Aurelius.

Implements the ChatML wire format used by OpenAI/Qwen/other frontier
models:

    <|im_start|>{role}\n{content}<|im_end|>\n

Roles supported: {system, user, assistant, tool}. The encoder is strict:
user-supplied content containing the ChatML control tokens is rejected
with ``ChatMLFormatError`` to prevent role-confusion / prompt-injection
attacks. There are no silent fallbacks.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})

# Pre-compiled decoder pattern. DOTALL so content may contain newlines.
_MESSAGE_RE = re.compile(
    r"<\|im_start\|>(?P<role>[^\n]+)\n(?P<content>.*?)<\|im_end\|>",
    re.DOTALL,
)


class ChatMLFormatError(ValueError):
    """Raised for any malformed ChatML payload or rejected input."""


@dataclass
class Message:
    """A single chat turn.

    Attributes:
        role: One of {system, user, assistant, tool}.
        content: Text payload. Must not contain ChatML control tokens.
        name: Optional name (e.g. tool name). Currently informational;
            not emitted on the wire by this template.
    """

    role: str
    content: str
    name: str | None = None


def _validate_content(role: str, content: str) -> None:
    """Reject content that would break out of its ChatML envelope."""
    if not isinstance(content, str):
        raise ChatMLFormatError(
            f"content for role={role!r} must be str, got {type(content).__name__}"
        )
    if IM_START in content or IM_END in content:
        raise ChatMLFormatError(
            f"content for role={role!r} contains a ChatML control token "
            "(<|im_start|> or <|im_end|>); rejecting to prevent role-confusion "
            "attack. Sanitize upstream if this is legitimate."
        )


class ChatMLTemplate:
    """Bidirectional ChatML codec.

    Instances are stateless and safe to share. All validation is
    explicit: bad roles or injection attempts raise ChatMLFormatError.
    """

    name = "chatml"

    # ------------------------------------------------------------------ encode

    def encode(
        self,
        messages: list[Message],
        add_generation_prompt: bool = False,
    ) -> str:
        """Serialize messages to a ChatML string.

        Args:
            messages: Ordered list of Message. An empty list is allowed;
                returns "" unless ``add_generation_prompt`` is True.
            add_generation_prompt: If True, append an opening assistant
                tag (`<|im_start|>assistant\n`) with NO closing tag, so
                the model will generate into it.
        """
        if not isinstance(messages, list):
            raise ChatMLFormatError("messages must be a list[Message]")

        parts: list[str] = []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, Message):
                raise ChatMLFormatError(f"messages[{idx}] is not a Message dataclass")
            if msg.role not in VALID_ROLES:
                raise ChatMLFormatError(
                    f"messages[{idx}] has invalid role={msg.role!r}; "
                    f"valid roles are {sorted(VALID_ROLES)}"
                )
            _validate_content(msg.role, msg.content)
            parts.append(f"{IM_START}{msg.role}\n{msg.content}{IM_END}\n")

        if add_generation_prompt:
            parts.append(f"{IM_START}assistant\n")

        return "".join(parts)

    # ------------------------------------------------------------------ decode

    def decode(self, text: str) -> list[Message]:
        """Parse a ChatML string back into Message objects.

        Any trailing whitespace and an optional open generation prompt
        (`<|im_start|>assistant\n...` with no closing `<|im_end|>`) are
        tolerated — the open prompt is discarded, not returned. Any
        other structural anomaly raises ChatMLFormatError.
        """
        if not isinstance(text, str):
            raise ChatMLFormatError("decode input must be str")

        text = text.rstrip()
        if text == "":
            return []

        messages: list[Message] = []
        pos = 0
        n = len(text)

        while pos < n:
            # Skip benign inter-message whitespace.
            while pos < n and text[pos] in ("\n", "\r", "\t", " "):
                pos += 1
            if pos >= n:
                break

            if not text.startswith(IM_START, pos):
                raise ChatMLFormatError(
                    f"expected {IM_START!r} at offset {pos}, got {text[pos : pos + 32]!r}"
                )

            m = _MESSAGE_RE.match(text, pos)
            if m is None:
                # Could be an open generation prompt: <|im_start|>role\n...
                # with no closing <|im_end|>. Accept only if it is the
                # final fragment; otherwise malformed.
                head = text[pos + len(IM_START) :]
                nl = head.find("\n")
                if nl == -1:
                    # rstrip may have eaten the trailing newline of a
                    # dangling generation prompt; treat the remainder
                    # as a bare role header.
                    role = head
                    remainder = ""
                else:
                    role = head[:nl]
                    remainder = head[nl + 1 :]
                if IM_END in remainder or IM_START in remainder:
                    raise ChatMLFormatError(f"malformed message at offset {pos}: missing {IM_END}")
                if role not in VALID_ROLES:
                    raise ChatMLFormatError(f"invalid role {role!r} in open generation prompt")
                # Treat as dangling generation prompt; stop parsing.
                return messages

            role = m.group("role")
            content = m.group("content")
            if role not in VALID_ROLES:
                raise ChatMLFormatError(
                    f"invalid role {role!r} at offset {pos}; valid roles are {sorted(VALID_ROLES)}"
                )
            # Content must not itself contain control tokens (the regex
            # is non-greedy so a nested <|im_start|> would be captured
            # here); reject it.
            if IM_START in content or IM_END in content:
                raise ChatMLFormatError(
                    f"nested ChatML control token inside message at offset {pos}"
                )
            # The content body may have a trailing newline before the
            # closing tag (the encoder does not emit one, but other
            # encoders do). Preserve content exactly as captured except
            # for a single optional trailing "\n" that is structural.
            if content.endswith("\n"):
                content = content[:-1]

            messages.append(Message(role=role, content=content))
            pos = m.end()

        return messages

    # ---------------------------------------------------------------- tokens

    def encode_token_ids(
        self,
        messages: list[Message],
        tokenizer: Callable[[str], Iterable[int]],
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """Encode messages to a flat list of token ids.

        ``tokenizer`` is any callable mapping ``str -> Iterable[int]``.
        We deliberately do not special-case the control tokens here —
        the caller's tokenizer is responsible for knowing whether
        <|im_start|> / <|im_end|> are atomic or byte-level. This keeps
        the template decoupled from any specific BPE/SentencePiece impl.
        """
        if not callable(tokenizer):
            raise ChatMLFormatError("tokenizer must be callable: str -> list[int]")
        rendered = self.encode(messages, add_generation_prompt=add_generation_prompt)
        ids = tokenizer(rendered)
        out = list(ids)
        for i, tok in enumerate(out):
            if not isinstance(tok, int):
                raise ChatMLFormatError(
                    f"tokenizer returned non-int at position {i}: {type(tok).__name__}"
                )
        return out
