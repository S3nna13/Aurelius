"""Terminal UI accessibility support: screen-reader announcements and ARIA regions.

Only stdlib and project-local imports are used.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Braille mapping for common characters (optional support)
# ---------------------------------------------------------------------------

_BRAILLE_MAP: dict[str, str] = {
    "a": "⠁", "b": "⠃", "c": "⠉", "d": "⠙", "e": "⠑",
    "f": "⠋", "g": "⠛", "h": "⠓", "i": "⠊", "j": "⠚",
    "k": "⠅", "l": "⠇", "m": "⠍", "n": "⠝", "o": "⠕",
    "p": "⠏", "q": "⠟", "r": "⠗", "s": "⠎", "t": "⠞",
    "u": "⠥", "v": "⠧", "w": "⠺", "x": "⠭", "y": "⠽",
    "z": "⠵",
    "A": "⠠⠁", "B": "⠠⠃", "C": "⠠⠉", "D": "⠠⠙", "E": "⠠⠑",
    "F": "⠠⠋", "G": "⠠⠛", "H": "⠠⠓", "I": "⠠⠊", "J": "⠠⠚",
    "K": "⠠⠅", "L": "⠠⠇", "M": "⠠⠍", "N": "⠠⠝", "O": "⠠⠕",
    "P": "⠠⠏", "Q": "⠠⠟", "R": "⠠⠗", "S": "⠠⠎", "T": "⠠⠞",
    "U": "⠠⠥", "V": "⠠⠧", "W": "⠠⠺", "X": "⠠⠭", "Y": "⠠⠽",
    "Z": "⠠⠵",
    "0": "⠴", "1": "⠂", "2": "⠆", "3": "⠒", "4": "⠲",
    "5": "⠢", "6": "⠖", "7": "⠶", "8": "⠦", "9": "⠔",
    " ": " ",
}

_PRIORITY_INDICATOR: dict[str, str] = {
    "critical": "[CRITICAL] ",
    "important": "[IMPORTANT] ",
    "normal": "[NORMAL] ",
    "low": "[LOW] ",
}

_VALID_PRIORITIES = frozenset(_PRIORITY_INDICATOR.keys())

_MAX_MESSAGE_LENGTH = 256


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AccessibilityAnnouncerError(ValueError):
    """Raised for invalid input to the accessibility announcer."""


# ---------------------------------------------------------------------------
# AriaLiveRegion
# ---------------------------------------------------------------------------


@dataclass
class AriaLiveRegion:
    """Tracks an ARIA live region configuration.

    Attributes:
        region_id: Unique identifier for this live region.
        mode: Either ``"polite"`` or ``"assertive"``.
        label: Human-readable label describing the region's purpose.
    """

    region_id: str
    mode: str
    label: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("polite", "assertive"):
            raise AccessibilityAnnouncerError(
                f"AriaLiveRegion mode must be 'polite' or 'assertive', got {self.mode!r}"
            )
        if not isinstance(self.region_id, str) or not self.region_id.strip():
            raise AccessibilityAnnouncerError("region_id must be a non-empty str")
        if not isinstance(self.label, str):
            raise AccessibilityAnnouncerError("label must be a str")


# ---------------------------------------------------------------------------
# AccessibilityAnnouncer
# ---------------------------------------------------------------------------


class AccessibilityAnnouncer:
    """Manages a queue of screen-reader announcements with priority support.

    All messages are sanitised (control characters stripped except ``\\n``
    and ``\\t``), truncated to 256 characters, and prefixed with a priority
    indicator when formatted for a screen reader.
    """

    def __init__(self) -> None:
        self._queue: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def announce(self, message: str, priority: str = "normal") -> str:
        """Queue an announcement after sanitising and truncating it.

        Args:
            message: The raw announcement text.
            priority: One of ``"critical"``, ``"important"``, ``"normal"``,
                or ``"low"``.

        Returns:
            The formatted string that was queued.

        Raises:
            AccessibilityAnnouncerError: If *priority* is not a recognised
                level or *message* is not a string.
        """
        if not isinstance(message, str):
            raise AccessibilityAnnouncerError("message must be a str")
        if priority not in _VALID_PRIORITIES:
            raise AccessibilityAnnouncerError(
                f"priority must be one of {_VALID_PRIORITIES}, got {priority!r}"
            )

        clean = self._sanitize(message)
        truncated = self._truncate(clean)
        formatted = self.format_for_screen_reader(truncated, priority)
        self._queue.append(formatted)
        return formatted

    def clear_queue(self) -> None:
        """Remove all pending announcements."""
        self._queue.clear()

    def pending(self) -> list[str]:
        """Return a shallow copy of pending announcements in queue order."""
        return list(self._queue)

    @staticmethod
    def format_for_screen_reader(message: str, priority: str = "normal") -> str:
        """Return *message* prefixed with a priority indicator.

        Args:
            message: Sanitised and truncated message text.
            priority: One of the supported priority levels.

        Returns:
            Formatted string such as ``"[CRITICAL] something happened"``.

        Raises:
            AccessibilityAnnouncerError: If *priority* is not recognised.
        """
        if priority not in _VALID_PRIORITIES:
            raise AccessibilityAnnouncerError(
                f"priority must be one of {_VALID_PRIORITIES}, got {priority!r}"
            )
        return f"{_PRIORITY_INDICATOR[priority]}{message}"

    @staticmethod
    def to_braille(text: str) -> str:
        """Map common ASCII characters to Unicode Braille patterns.

        Unmapped characters are returned unchanged.

        Args:
            text: Input text to transliterate.

        Returns:
            Braille-pattern string.
        """
        return "".join(_BRAILLE_MAP.get(ch, ch) for ch in text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize(text: str) -> str:
        """Strip control characters except ``\\n`` and ``\\t``.

        Also strips zero-width and formatting characters.
        """
        # Remove control chars except newline and tab
        cleaned = "".join(
            ch for ch in text
            if ch in ("\n", "\t") or not unicodedata.category(ch).startswith("C")
        )
        # Also strip common zero-width/formatting characters explicitly
        cleaned = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", cleaned)
        return cleaned

    @staticmethod
    def _truncate(text: str) -> str:
        """Truncate *text* to ``_MAX_MESSAGE_LENGTH`` characters, appending ``...``."""
        if len(text) <= _MAX_MESSAGE_LENGTH:
            return text
        return text[: _MAX_MESSAGE_LENGTH - 3] + "..."


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

DEFAULT_ACCESSIBILITY_ANNOUNCER = AccessibilityAnnouncer()

__all__ = [
    "AccessibilityAnnouncer",
    "AccessibilityAnnouncerError",
    "AriaLiveRegion",
    "DEFAULT_ACCESSIBILITY_ANNOUNCER",
]
