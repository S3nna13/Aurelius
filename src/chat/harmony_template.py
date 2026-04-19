"""OpenAI Harmony chat template encoder/decoder for Aurelius.

Implements OpenAI's Harmony wire format used by ``gpt-oss-20b`` /
``gpt-oss-120b`` and related models. Harmony distinguishes role turns
(system / developer / user / assistant / tool) and, for assistant
turns, a channel dimension (final / analysis / commentary) that lets
the model route its internal reasoning separately from user-visible
output.

Envelope (no channel, non-assistant roles)::

    <|start|>{role}<|message|>{content}<|end|>

Envelope (assistant, with channel)::

    <|start|>assistant<|channel|>{channel}<|message|>{content}<|end|>

Design constraints:

* Strict validation: unknown roles, unknown channels, or user content
  that contains any Harmony control token is rejected. There are no
  silent fallbacks — a corrupted payload must surface as an error.
* Stateless: ``HarmonyTemplate`` instances are safe to share across
  threads / processes.
* Parallel (not subclass) dataclass ``HarmonyMessage`` so the type
  check in ``encode`` is explicit and cannot accept a plain ChatML
  ``Message`` that would be missing channel awareness.
* Pure stdlib: no ``transformers`` / third-party dependency.

Reference: OpenAI ``openai-harmony`` format spec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

START = "<|start|>"
END = "<|end|>"
MESSAGE = "<|message|>"
CHANNEL = "<|channel|>"

# All Harmony control tokens. Any one of these appearing inside
# user-supplied content is a role-confusion / prompt-injection attempt.
_CONTROL_TOKENS = (START, END, MESSAGE, CHANNEL)

VALID_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
VALID_CHANNELS = frozenset({"final", "analysis", "commentary"})

# Role that permits (and requires) a channel. The spec models a channel
# solely on assistant turns; tool / user / system / developer never
# carry one.
_CHANNELED_ROLES = frozenset({"assistant"})

# Default channel applied when an assistant message is constructed with
# ``channel=None``. ``final`` is the user-visible channel and is the
# safe default for a caller that hasn't thought about channels yet.
_DEFAULT_ASSISTANT_CHANNEL = "final"

# Decoder pattern. Two alternatives: with channel (assistant) and
# without (other roles). DOTALL so content may contain newlines; the
# ``?`` qualifiers are non-greedy so we don't swallow the next turn.
_MESSAGE_RE = re.compile(
    r"<\|start\|>(?P<role>[^<|]+?)"
    r"(?:<\|channel\|>(?P<channel>[^<|]+?))?"
    r"<\|message\|>(?P<content>.*?)<\|end\|>",
    re.DOTALL,
)


class HarmonyFormatError(ValueError):
    """Raised for any malformed Harmony payload or rejected input."""


@dataclass
class HarmonyMessage:
    """A single Harmony chat turn.

    Parallel to ``src.chat.chatml_template.Message`` (intentionally not
    a subclass, so the encoder can enforce that callers opt into the
    channel-aware surface).

    Attributes:
        role: One of {system, developer, user, assistant, tool}.
        content: Text payload. Must not contain Harmony control tokens.
        channel: For ``assistant`` only, one of {final, analysis,
            commentary}. ``None`` is accepted for assistant turns and
            resolves to ``final`` at encode time. Must be ``None`` for
            every other role.
        name: Optional name (e.g. tool name). Informational; not
            emitted on the wire by this template.
    """

    role: str
    content: str
    channel: Optional[str] = None
    name: Optional[str] = None


def _validate_content(role: str, content: str) -> None:
    """Reject content that would break out of its Harmony envelope."""
    if not isinstance(content, str):
        raise HarmonyFormatError(
            f"content for role={role!r} must be str, got "
            f"{type(content).__name__}"
        )
    for tok in _CONTROL_TOKENS:
        if tok in content:
            raise HarmonyFormatError(
                f"content for role={role!r} contains Harmony control token "
                f"{tok!r}; rejecting to prevent role-confusion / injection "
                "attack. Sanitize upstream if this is legitimate."
            )


def _resolve_channel(role: str, channel: Optional[str], idx: int) -> Optional[str]:
    """Validate and normalize the channel for one message.

    Assistant turns always emit a channel (defaulting to ``final``).
    Non-assistant turns must have ``channel is None``.
    """
    if role in _CHANNELED_ROLES:
        if channel is None:
            return _DEFAULT_ASSISTANT_CHANNEL
        if channel not in VALID_CHANNELS:
            raise HarmonyFormatError(
                f"messages[{idx}] has invalid channel={channel!r} for role "
                f"{role!r}; valid channels are {sorted(VALID_CHANNELS)}"
            )
        return channel
    # Non-channelled role.
    if channel is not None:
        raise HarmonyFormatError(
            f"messages[{idx}] role={role!r} must not carry a channel "
            f"(got channel={channel!r}); channels are assistant-only"
        )
    return None


class HarmonyTemplate:
    """Bidirectional Harmony chat codec.

    Mirrors the ``ChatMLTemplate`` / ``Llama3Template`` API:
    ``encode``, ``decode``, ``encode_token_ids``. Instances are
    stateless and safe to share.
    """

    name = "harmony"

    # ------------------------------------------------------------------ encode

    def encode(
        self,
        messages: List[HarmonyMessage],
        add_generation_prompt: bool = False,
    ) -> str:
        """Serialize ``messages`` to a Harmony string.

        Args:
            messages: Ordered list of ``HarmonyMessage``. An empty list
                is allowed; returns ``""`` unless
                ``add_generation_prompt`` is True.
            add_generation_prompt: If True, append an opening assistant
                header (``<|start|>assistant``) with no closing tag so
                the model generates the assistant turn (including its
                channel selection).
        """
        if not isinstance(messages, list):
            raise HarmonyFormatError("messages must be a list[HarmonyMessage]")

        parts: List[str] = []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, HarmonyMessage):
                raise HarmonyFormatError(
                    f"messages[{idx}] is not a HarmonyMessage dataclass "
                    f"(got {type(msg).__name__})"
                )
            if msg.role not in VALID_ROLES:
                raise HarmonyFormatError(
                    f"messages[{idx}] has invalid role={msg.role!r}; "
                    f"valid roles are {sorted(VALID_ROLES)}"
                )
            _validate_content(msg.role, msg.content)
            resolved = _resolve_channel(msg.role, msg.channel, idx)

            if resolved is None:
                parts.append(
                    f"{START}{msg.role}{MESSAGE}{msg.content}{END}"
                )
            else:
                parts.append(
                    f"{START}{msg.role}{CHANNEL}{resolved}"
                    f"{MESSAGE}{msg.content}{END}"
                )

        if add_generation_prompt:
            # No channel is pre-committed: the model selects its
            # channel as part of generation.
            parts.append(f"{START}assistant")

        return "".join(parts)

    # ------------------------------------------------------------------ decode

    def decode(self, text: str) -> List[HarmonyMessage]:
        """Parse a Harmony string back into ``HarmonyMessage`` objects.

        Tolerates inter-message whitespace and a trailing open
        assistant generation prompt (``<|start|>assistant`` with no
        terminating ``<|end|>``); the open prompt is discarded. Any
        other structural anomaly raises ``HarmonyFormatError``.
        """
        if not isinstance(text, str):
            raise HarmonyFormatError("decode input must be str")

        text = text.rstrip()
        if text == "":
            return []

        messages: List[HarmonyMessage] = []
        pos = 0
        n = len(text)

        while pos < n:
            # Skip benign inter-message whitespace.
            while pos < n and text[pos] in ("\n", "\r", "\t", " "):
                pos += 1
            if pos >= n:
                break

            if not text.startswith(START, pos):
                raise HarmonyFormatError(
                    f"expected {START!r} at offset {pos}, got "
                    f"{text[pos:pos + 32]!r}"
                )

            m = _MESSAGE_RE.match(text, pos)
            if m is None:
                # Possibly an open generation prompt. Shape:
                #   <|start|>role                     (bare header)
                #   <|start|>role<|channel|>chan      (header + channel)
                # with no <|message|> / <|end|>. Accept only as the
                # trailing fragment; anything else is malformed.
                tail = text[pos + len(START):]
                # Must not contain <|message|> or <|end|> here — that
                # would mean our regex failed for some other reason.
                if MESSAGE in tail or END in tail:
                    raise HarmonyFormatError(
                        f"malformed message at offset {pos}: could not "
                        "parse turn envelope"
                    )
                if CHANNEL in tail:
                    role, _, chan = tail.partition(CHANNEL)
                    if CHANNEL in chan:
                        raise HarmonyFormatError(
                            f"duplicate {CHANNEL!r} in open prompt at "
                            f"offset {pos}"
                        )
                    if role not in VALID_ROLES:
                        raise HarmonyFormatError(
                            f"invalid role {role!r} in open generation "
                            "prompt"
                        )
                    if role not in _CHANNELED_ROLES:
                        raise HarmonyFormatError(
                            f"role {role!r} cannot carry a channel"
                        )
                    if chan and chan not in VALID_CHANNELS:
                        raise HarmonyFormatError(
                            f"invalid channel {chan!r} in open "
                            "generation prompt"
                        )
                else:
                    role = tail
                    if role not in VALID_ROLES:
                        raise HarmonyFormatError(
                            f"invalid role {role!r} in open generation "
                            "prompt"
                        )
                # Dangling open prompt. Stop parsing.
                return messages

            role = m.group("role")
            channel = m.group("channel")
            content = m.group("content")

            if role not in VALID_ROLES:
                raise HarmonyFormatError(
                    f"invalid role {role!r} at offset {pos}; "
                    f"valid roles are {sorted(VALID_ROLES)}"
                )
            if channel is not None:
                if role not in _CHANNELED_ROLES:
                    raise HarmonyFormatError(
                        f"role {role!r} at offset {pos} carries a "
                        f"channel; channels are assistant-only"
                    )
                if channel not in VALID_CHANNELS:
                    raise HarmonyFormatError(
                        f"invalid channel {channel!r} at offset {pos}; "
                        f"valid channels are {sorted(VALID_CHANNELS)}"
                    )
            # Content must not contain nested control tokens. The
            # regex is non-greedy so a nested <|start|> would be
            # captured here; reject smuggling.
            for tok in _CONTROL_TOKENS:
                if tok in content:
                    raise HarmonyFormatError(
                        f"nested Harmony control token {tok!r} inside "
                        f"message at offset {pos}"
                    )

            messages.append(
                HarmonyMessage(role=role, content=content, channel=channel)
            )
            pos = m.end()

        return messages

    # ---------------------------------------------------------------- tokens

    def encode_token_ids(
        self,
        messages: List[HarmonyMessage],
        tokenizer: Callable[[str], Iterable[int]],
        add_generation_prompt: bool = False,
    ) -> List[int]:
        """Encode messages to a flat list of token ids.

        ``tokenizer`` is any callable mapping ``str -> Iterable[int]``.
        We deliberately do not special-case Harmony control tokens
        here — the caller's tokenizer is responsible for knowing
        whether ``<|start|>``, ``<|end|>``, ``<|message|>``, and
        ``<|channel|>`` are atomic ids or byte-level sequences. This
        keeps the template decoupled from any specific BPE /
        SentencePiece implementation.
        """
        if not callable(tokenizer):
            raise HarmonyFormatError(
                "tokenizer must be callable: str -> list[int]"
            )
        rendered = self.encode(
            messages, add_generation_prompt=add_generation_prompt
        )
        ids = tokenizer(rendered)
        out = list(ids)
        for i, tok in enumerate(out):
            if not isinstance(tok, int):
                raise HarmonyFormatError(
                    f"tokenizer returned non-int at position {i}: "
                    f"{type(tok).__name__}"
                )
        return out
