"""Llama-3 chat template encoder/decoder for Aurelius.

Implements Meta's Llama-3 header-style wire format:

    <|begin_of_text|>
    <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>

Roles supported: {system, user, assistant, tool, ipython}. The encoder
is strict: any user-supplied content containing the Llama-3 control
tokens is rejected with ``Llama3FormatError`` to prevent role-confusion
/ prompt-injection attacks. There are no silent fallbacks.

Reference: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from .chatml_template import Message

BEGIN_OF_TEXT = "<|begin_of_text|>"
START_HEADER = "<|start_header_id|>"
END_HEADER = "<|end_header_id|>"
EOT = "<|eot_id|>"

VALID_ROLES = frozenset({"system", "user", "assistant", "tool", "ipython"})

# All control tokens that must never appear inside user-supplied content.
_CONTROL_TOKENS = (BEGIN_OF_TEXT, START_HEADER, END_HEADER, EOT)

# Decoder pattern: one message turn. DOTALL so content may contain newlines.
_MESSAGE_RE = re.compile(
    r"<\|start_header_id\|>(?P<role>[^<\n]+?)<\|end_header_id\|>\n\n"
    r"(?P<content>.*?)<\|eot_id\|>",
    re.DOTALL,
)


class Llama3FormatError(ValueError):
    """Raised for any malformed Llama-3 payload or rejected input."""


def _validate_content(role: str, content: str) -> None:
    """Reject content that would break out of its Llama-3 envelope."""
    if not isinstance(content, str):
        raise Llama3FormatError(
            f"content for role={role!r} must be str, got {type(content).__name__}"
        )
    for tok in _CONTROL_TOKENS:
        if tok in content:
            raise Llama3FormatError(
                f"content for role={role!r} contains Llama-3 control token "
                f"{tok!r}; rejecting to prevent role-confusion / injection "
                "attack. Sanitize upstream if this is legitimate."
            )


class Llama3Template:
    """Bidirectional Llama-3 chat codec.

    Instances are stateless and safe to share. All validation is
    explicit: bad roles or injection attempts raise Llama3FormatError.
    """

    name = "llama3"

    # ------------------------------------------------------------------ encode

    def encode(
        self,
        messages: list[Message],
        add_generation_prompt: bool = False,
    ) -> str:
        """Serialize messages to a Llama-3 chat string.

        Args:
            messages: Ordered list of Message. An empty list is allowed;
                returns "" if ``add_generation_prompt`` is False, else
                ``<|begin_of_text|>`` plus an open assistant header.
            add_generation_prompt: If True, append a single open
                assistant header with no trailing ``<|eot_id|>`` so the
                model generates the assistant turn.
        """
        if not isinstance(messages, list):
            raise Llama3FormatError("messages must be a list[Message]")

        if not messages and not add_generation_prompt:
            return ""

        parts: list[str] = [BEGIN_OF_TEXT]
        for idx, msg in enumerate(messages):
            if not isinstance(msg, Message):
                raise Llama3FormatError(f"messages[{idx}] is not a Message dataclass")
            if msg.role not in VALID_ROLES:
                raise Llama3FormatError(
                    f"messages[{idx}] has invalid role={msg.role!r}; "
                    f"valid roles are {sorted(VALID_ROLES)}"
                )
            _validate_content(msg.role, msg.content)
            parts.append(f"{START_HEADER}{msg.role}{END_HEADER}\n\n{msg.content}{EOT}")

        if add_generation_prompt:
            parts.append(f"{START_HEADER}assistant{END_HEADER}\n\n")

        return "".join(parts)

    # ------------------------------------------------------------------ decode

    def decode(self, text: str) -> list[Message]:
        """Parse a Llama-3 chat string back into Message objects.

        Tolerates a leading ``<|begin_of_text|>``, inter-message
        whitespace (Meta's reference serializer emits no trailing
        newline after ``<|eot_id|>``, but other serializers do), and a
        trailing open assistant generation prompt (which is discarded).
        Any other structural anomaly raises Llama3FormatError.
        """
        if not isinstance(text, str):
            raise Llama3FormatError("decode input must be str")

        text = text.rstrip()
        if text == "":
            return []

        # Optional BOS.
        if text.startswith(BEGIN_OF_TEXT):
            pos = len(BEGIN_OF_TEXT)
        else:
            pos = 0

        messages: list[Message] = []
        n = len(text)

        while pos < n:
            # Skip benign inter-message whitespace.
            while pos < n and text[pos] in ("\n", "\r", "\t", " "):
                pos += 1
            if pos >= n:
                break

            if not text.startswith(START_HEADER, pos):
                raise Llama3FormatError(
                    f"expected {START_HEADER!r} at offset {pos}, got {text[pos : pos + 32]!r}"
                )

            m = _MESSAGE_RE.match(text, pos)
            if m is None:
                # Possibly an open generation prompt:
                # <|start_header_id|>role<|end_header_id|>\n\n... with
                # no closing <|eot_id|>. Accept only as the final fragment.
                after_start = text[pos + len(START_HEADER) :]
                end_idx = after_start.find(END_HEADER)
                if end_idx == -1:
                    raise Llama3FormatError(
                        f"malformed header at offset {pos}: missing {END_HEADER!r}"
                    )
                role = after_start[:end_idx]
                remainder = after_start[end_idx + len(END_HEADER) :]
                # Any further control token in the tail means malformed,
                # not just a dangling open prompt.
                for tok in _CONTROL_TOKENS:
                    if tok in remainder:
                        raise Llama3FormatError(
                            f"malformed message at offset {pos}: no "
                            f"{EOT!r} terminator before next control token"
                        )
                if role not in VALID_ROLES:
                    raise Llama3FormatError(f"invalid role {role!r} in open generation prompt")
                # Dangling open prompt. Stop parsing.
                return messages

            role = m.group("role")
            content = m.group("content")
            if role not in VALID_ROLES:
                raise Llama3FormatError(
                    f"invalid role {role!r} at offset {pos}; valid roles are {sorted(VALID_ROLES)}"
                )
            # Non-greedy regex means a nested control token would be
            # captured in ``content``; reject to avoid smuggling.
            for tok in _CONTROL_TOKENS:
                if tok in content:
                    raise Llama3FormatError(
                        f"nested Llama-3 control token {tok!r} inside message at offset {pos}"
                    )
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
        ``<|begin_of_text|>`` / ``<|start_header_id|>`` / ``<|eot_id|>``
        are atomic or byte-level. This keeps the template decoupled from
        any specific BPE/SentencePiece impl.
        """
        if not callable(tokenizer):
            raise Llama3FormatError("tokenizer must be callable: str -> list[int]")
        rendered = self.encode(messages, add_generation_prompt=add_generation_prompt)
        ids = tokenizer(rendered)
        out = list(ids)
        for i, tok in enumerate(out):
            if not isinstance(tok, int):
                raise Llama3FormatError(
                    f"tokenizer returned non-int at position {i}: {type(tok).__name__}"
                )
        return out
