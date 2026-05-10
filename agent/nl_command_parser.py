"""Natural-language command parser for the Aurelius agent system."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_NL_PARSER",
    "NL_PARSER_REGISTRY",
    "NLCommandParseError",
    "NLCommandParser",
    "ParsedCommand",
]


class NLCommandParseError(Exception):
    """Raised when input text cannot be parsed as a valid command."""


@dataclass
class ParsedCommand:
    """Structured representation of a parsed natural-language command."""

    action: str
    target: str | None = None
    args: dict[str, str] = field(default_factory=dict)
    raw_text: str = ""


@dataclass
class NLCommandParser:
    """Parse natural-language strings into structured :class:`ParsedCommand` objects."""

    _PATTERNS: list[tuple[re.Pattern, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._PATTERNS:
            self._PATTERNS = self._build_patterns()

    def _build_patterns(self) -> list[tuple[re.Pattern, str]]:
        """Return the default list of (compiled regex, action_name) tuples."""
        return [
            (
                re.compile(
                    r"^(?:run|execute|use)\s+skill\s+(?P<target>\S+)"
                    r"(?:\s+on\s+(?P<on>\S+))?$"
                ),
                "run_skill",
            ),
            (re.compile(r"^(?:list|show)\s+skills$"), "list_skills"),
            (
                re.compile(r"^what\s+skills\s+are\s+available$"),
                "list_skills",
            ),
            (
                re.compile(r"^(?:load|install)\s+plugin\s+(?P<target>\S+)$"),
                "load_plugin",
            ),
            (re.compile(r"^(?:list|show)\s+plugins$"), "list_plugins"),
            (
                re.compile(r"^(?:activate|enable)\s+skill\s+(?P<target>\S+)$"),
                "activate_skill",
            ),
            (
                re.compile(r"^(?:deactivate|disable)\s+skill\s+(?P<target>\S+)$"),
                "deactivate_skill",
            ),
            (re.compile(r"^show\s+agent\s+status$"), "agent_status"),
            (
                re.compile(r"^what\s+is\s+the\s+agent\s+doing$"),
                "agent_status",
            ),
            (re.compile(r"^(?:list|show)\s+agents$"), "list_agents"),
            (
                re.compile(r"^(?:run|execute)\s+task\s+(?P<target>\S+)$"),
                "run_task",
            ),
            (
                re.compile(r"^show\s+(?:board|tasks|work)$"),
                "show_board",
            ),
        ]

    def parse(self, text: str) -> ParsedCommand:
        """Parse *text* into a :class:`ParsedCommand`.

        Args:
            text: Raw natural-language command.

        Returns:
            A :class:`ParsedCommand` with the extracted action, target, and args.

        Raises:
            NLCommandParseError: If *text* is empty or whitespace-only.
        """
        if not isinstance(text, str) or not text.strip():
            raise NLCommandParseError("Input text must be a non-empty string")

        raw = text
        normalized = " ".join(text.lower().split())

        for pattern, action in self._PATTERNS:
            match = pattern.match(normalized)
            if match:
                groups = match.groupdict()
                target = groups.get("target")
                args = {k: v for k, v in groups.items() if k != "target" and v is not None}
                return ParsedCommand(
                    action=action,
                    target=target,
                    args=args,
                    raw_text=raw,
                )

        return ParsedCommand(action="chat", raw_text=raw)


DEFAULT_NL_PARSER = NLCommandParser()
NL_PARSER_REGISTRY: dict[str, NLCommandParser] = {"default": DEFAULT_NL_PARSER}
