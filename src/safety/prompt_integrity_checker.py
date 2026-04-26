"""System-prompt integrity checker for the Aurelius safety surface.

This module provides a deterministic, stdlib-only guard that agent loops
can run at every step to verify that the *system prompt* they are about
to send to the model has not been tampered with.

It looks for four classes of compromise:

1. **Checksum drift** — SHA-256 of the current system prompt differs
   from the baseline fingerprint registered at agent boot time. This
   catches in-memory patching as well as accidental string-interpolation
   bugs that silently mutate the prompt.
2. **Length delta** — even if a full hash check is disabled, a prompt
   whose length has drifted more than ``tolerated_length_delta``
   characters from the baseline is flagged. Useful when the agent
   template splices in e.g. a timestamp that changes the hash but whose
   size is known and bounded.
3. **Role-break tokens** — the current prompt must not embed any of
   the ChatML / Llama-3 / Harmony control tokens. Their presence inside
   what is supposed to be plain system-prompt text is a strong signal
   that a user turn or tool output has been merged into the system
   slot by an upstream bug or a prompt-injection attack.
4. **Round-trip mismatch** — when given a full ``messages`` list, we
   encode it through the configured chat template, decode back, and
   assert that the *decoded* system message equals ``current_prompt``.
   Any divergence here means the wire format rewrote the prompt (e.g.
   an injected user message claiming the ``system`` role).

Everything is stdlib. No numpy, no torch, no transformers.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from src.chat.chatml_template import ChatMLTemplate, Message

# Control tokens that must never appear in a plain system prompt.
# Sourced from the chat-template siblings so we catch the formats
# Aurelius actually ships (ChatML, Llama-3, Harmony).
_ROLE_BREAK_TOKENS: tuple[str, ...] = (
    # ChatML
    "<|im_start|>",
    "<|im_end|>",
    # Llama-3
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    # Harmony / OpenAI-style
    "<|start|>",
    "<|end|>",
    "<|message|>",
    "<|channel|>",
    "<|return|>",
)


_SUPPORTED_TEMPLATES: frozenset[str] = frozenset({"chatml"})


@dataclass
class IntegrityReport:
    """Result of a single integrity check pass.

    Attributes:
        valid: True iff the current prompt is considered safe to use.
        signals: Stable-ordered list of short machine-readable labels
            naming each anomaly found. Empty when ``valid`` is True
            (except for the benign ``"no_baseline"`` warning).
        expected_checksum: SHA-256 hex of the registered baseline, or
            ``None`` if no baseline has been registered.
        actual_checksum: SHA-256 hex of ``current_prompt``.
        length_delta: ``len(current_prompt) - len(baseline_prompt)``.
            Zero when no baseline has been registered.
    """

    valid: bool
    signals: list[str] = field(default_factory=list)
    expected_checksum: str | None = None
    actual_checksum: str = ""
    length_delta: int = 0


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class PromptIntegrityChecker:
    """Guard that verifies a system prompt against a registered baseline.

    Args:
        expected_prompt: Optional baseline prompt to register at
            construction time. Equivalent to calling
            :meth:`register_baseline` immediately after ``__init__``.
        tolerated_length_delta: Maximum absolute character-count delta
            that does not trigger a ``length_delta_exceeded`` signal.
        strict_checksum: If True, any checksum mismatch marks the
            report invalid. If False, a checksum mismatch is still
            reported but only downgrades validity if it is accompanied
            by another signal (length / role-break).
        template: Name of the chat template to use for
            :meth:`check_round_trip`. Only ``"chatml"`` is currently
            supported; unknown values raise ``ValueError`` at
            construction time.
    """

    def __init__(
        self,
        expected_prompt: str | None = None,
        tolerated_length_delta: int = 0,
        strict_checksum: bool = False,
        template: str = "chatml",
    ) -> None:
        if tolerated_length_delta < 0:
            raise ValueError(f"tolerated_length_delta must be >= 0, got {tolerated_length_delta}")
        if template not in _SUPPORTED_TEMPLATES:
            raise ValueError(
                f"unknown template {template!r}; supported: {sorted(_SUPPORTED_TEMPLATES)}"
            )

        self.tolerated_length_delta = int(tolerated_length_delta)
        self.strict_checksum = bool(strict_checksum)
        self.template_name = template
        self._template = ChatMLTemplate()

        self._baseline_prompt: str | None = None
        self._baseline_checksum: str | None = None

        if expected_prompt is not None:
            self.register_baseline(expected_prompt)

    # ------------------------------------------------------------------ baseline

    def register_baseline(self, prompt: str) -> None:
        """Install ``prompt`` as the approved system prompt fingerprint."""
        if not isinstance(prompt, str):
            raise TypeError(f"baseline prompt must be str, got {type(prompt).__name__}")
        self._baseline_prompt = prompt
        self._baseline_checksum = _sha256_hex(prompt)

    # --------------------------------------------------------------------- check

    def check(self, current_prompt: str) -> IntegrityReport:
        """Check ``current_prompt`` against the registered baseline."""
        if not isinstance(current_prompt, str):
            raise TypeError(f"current_prompt must be str, got {type(current_prompt).__name__}")

        actual = _sha256_hex(current_prompt)
        signals: list[str] = []

        # Role-break scan is independent of whether a baseline exists.
        role_break_hits = [tok for tok in _ROLE_BREAK_TOKENS if tok in current_prompt]
        if role_break_hits:
            signals.append("role_break_token")

        if self._baseline_prompt is None:
            signals.append("no_baseline")
            # No baseline means we cannot confirm tampering; we only
            # fail closed if a role-break token was spotted.
            return IntegrityReport(
                valid=not role_break_hits,
                signals=signals,
                expected_checksum=None,
                actual_checksum=actual,
                length_delta=0,
            )

        expected = self._baseline_checksum
        length_delta = len(current_prompt) - len(self._baseline_prompt)

        checksum_ok = actual == expected
        if not checksum_ok:
            signals.append("checksum_mismatch")

        if abs(length_delta) > self.tolerated_length_delta:
            signals.append("length_delta_exceeded")

        # Validity rules:
        #   - role-break tokens are always fatal.
        #   - length_delta_exceeded is always fatal.
        #   - checksum_mismatch alone is fatal only in strict mode.
        valid = True
        if role_break_hits:
            valid = False
        if "length_delta_exceeded" in signals:
            valid = False
        if not checksum_ok and self.strict_checksum:
            valid = False

        return IntegrityReport(
            valid=valid,
            signals=signals,
            expected_checksum=expected,
            actual_checksum=actual,
            length_delta=length_delta,
        )

    # -------------------------------------------------------------- round-trip

    def check_round_trip(
        self,
        current_prompt: str,
        messages: list[Message],
    ) -> IntegrityReport:
        """Verify that encoding and decoding ``messages`` preserves the system prompt.

        The first ``system`` message in ``messages`` (if any) must, after
        a full encode -> decode pass through the configured chat
        template, still equal ``current_prompt``. Any deviation adds the
        ``round_trip_mismatch`` signal. If no system message is present
        at all, that is itself an injection red flag and yields the
        ``system_message_missing`` signal.
        """
        report = self.check(current_prompt)

        if not isinstance(messages, list):
            raise TypeError("messages must be a list[Message]")

        # Locate the first system message.
        system_idx = next(
            (i for i, m in enumerate(messages) if isinstance(m, Message) and m.role == "system"),
            None,
        )

        round_trip_signals: list[str] = []
        if system_idx is None:
            round_trip_signals.append("system_message_missing")
        else:
            # A user-role message appearing *before* the first system
            # message is a strong signal that someone tried to inject
            # content ahead of the real system slot.
            if any(isinstance(m, Message) and m.role == "user" for m in messages[:system_idx]):
                round_trip_signals.append("user_before_system")

            encoded = self._template.encode(messages)
            decoded = self._template.decode(encoded)
            decoded_system = next((m for m in decoded if m.role == "system"), None)
            if decoded_system is None or decoded_system.content != current_prompt:
                round_trip_signals.append("round_trip_mismatch")

        if round_trip_signals:
            # Extend while preserving insertion order.
            for sig in round_trip_signals:
                if sig not in report.signals:
                    report.signals.append(sig)
            report.valid = False

        return report


__all__ = [
    "IntegrityReport",
    "PromptIntegrityChecker",
]
