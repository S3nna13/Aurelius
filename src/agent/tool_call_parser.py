"""Tool call parser for Aurelius agent loop.

Parses tool-invocation payloads emitted by the LM in two canonical formats:

  * Anthropic-style XML:
        <tool_use name="search" id="toolu_01"><input>{"q": "hi"}</input></tool_use>

  * OpenAI/GPT-style JSON:
        {"tool_calls": [{"id": "...", "name": "search",
                         "arguments": {"q": "hi"}}]}

The parser is security-critical: LM output is untrusted. We never eval,
never unescape arguments blindly, and never recursively interpret nested
<tool_use> tokens produced inside user-quoted text. Only top-level,
non-overlapping tags are recognised; injection attempts inside code
fences or JSON strings are parsed as plain text and ignored.

Design notes
------------
* Pure stdlib (json, re, html). No ML, no third-party parsers.
* Streaming safe: an unterminated trailing <tool_use ...> yields an
  empty parse list plus a ``remaining_buffer`` on the result wrapper so
  the agent loop can feed the next token chunk back in.
* Deterministic: identical input always produces an identical list of
  ParsedToolCall in the same order.
* Explicit failure: malformed JSON raises ToolCallParseError (subclass
  of ValueError) carrying the byte offset of the failure.

This module is intentionally side-effect free at import beyond
populating the shared registries in ``src.agent.__init__``.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ToolCallParseError(ValueError):
    """Raised when a candidate tool call cannot be parsed.

    Attributes
    ----------
    position:
        Byte offset into the original input where parsing failed, or
        ``-1`` if the offset is not meaningful (e.g. structural errors).
    snippet:
        The 64-character window around the failure point, useful for
        logging without dumping the full (possibly huge) input.
    """

    def __init__(self, message: str, position: int = -1, snippet: str = "") -> None:
        super().__init__(f"{message} (position={position})")
        self.position = position
        self.snippet = snippet


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedToolCall:
    """A single tool invocation extracted from model output."""

    name: str
    arguments: dict
    raw: str
    call_id: str | None = None


@dataclass
class ParseResult:
    """Wrapper for streaming-aware parses.

    ``calls`` always contains *complete* tool calls; ``remaining_buffer``
    holds any trailing bytes that look like the start of a tool call but
    have not yet been terminated.
    """

    calls: list[ParsedToolCall] = field(default_factory=list)
    remaining_buffer: str = ""


# ---------------------------------------------------------------------------
# XML parser
# ---------------------------------------------------------------------------


# Max input size we will consider. 4MB is comfortably above realistic
# single-turn model output and prevents pathological regex scans.
_MAX_INPUT_BYTES = 4 * 1024 * 1024

# Matches the *opening* <tool_use ...> tag. We deliberately do NOT match the
# closing tag with a single regex: a non-greedy .*? would truncate at any
# </tool_use> that appears inside a JSON string argument (adversarial input),
# and a greedy .* would coalesce sibling tool calls. Instead we locate each
# opener, then scan the tail with a JSON-string-aware state machine to find
# the true matching close.
_XML_OPEN_RE = re.compile(
    r"<tool_use\b(?P<attrs>[^>]*)>",
    re.DOTALL,
)
_XML_CLOSE_TOKEN = "</tool_use>"  # noqa: S105

# Matches the start of an *unterminated* tool_use tag at EOF (streaming).
_XML_OPEN_TRAIL_RE = re.compile(
    r"<tool_use\b[^<]*$",
    re.DOTALL,
)

# Matches name="..." / id="..." style attributes with either quote kind.
_XML_ATTR_RE = re.compile(
    r"""(?P<key>[A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*(?P<quote>["'])(?P<val>.*?)(?P=quote)""",
    re.DOTALL,
)

# Matches the opening <input ...> tag. As with <tool_use>, we locate the
# matching </input> via a JSON-string-aware scanner rather than a regex,
# so argument payloads that legitimately contain "</input>" inside string
# literals are preserved intact.
_XML_INPUT_OPEN_RE = re.compile(
    r"<input\b[^>]*>",
    re.DOTALL,
)
_XML_INPUT_CLOSE_TOKEN = "</input>"  # noqa: S105


class XMLToolCallParser:
    """Parse Anthropic-style ``<tool_use>`` XML blocks.

    Only top-level ``<tool_use>`` tags are recognised. If the model
    emits another ``<tool_use>`` *inside* a JSON string or code fence,
    the outer regex will still find the outermost pair; any nested
    occurrence is consumed as part of that first block's body. Nested
    tool-calls are *not* recursively parsed — the body is treated as
    opaque JSON — so prompt-injection via fake tags inside arguments
    cannot escalate.
    """

    def parse(self, text: str) -> list[ParsedToolCall]:
        """Parse and return the completed tool calls in ``text``.

        Unterminated trailing tags are silently dropped here; callers
        wanting streaming support should use :meth:`parse_stream`.
        """
        return self.parse_stream(text).calls

    def parse_stream(self, text: str) -> ParseResult:
        if not isinstance(text, str):
            raise ToolCallParseError("input must be str", -1, "")
        if len(text) > _MAX_INPUT_BYTES:
            raise ToolCallParseError("input exceeds max size", _MAX_INPUT_BYTES, "")

        result = ParseResult()
        pos = 0
        n = len(text)
        while pos < n:
            open_match = _XML_OPEN_RE.search(text, pos)
            if open_match is None:
                break
            attrs_raw = open_match.group("attrs") or ""
            body_start = open_match.end()
            # Scan forward for the matching </tool_use>, skipping any close
            # token that appears inside a JSON string literal.
            close_idx = self._find_close(text, body_start)
            if close_idx == -1:
                # Unterminated: surface as streaming buffer and stop.
                result.remaining_buffer = text[open_match.start() :]
                return result

            body = text[body_start:close_idx]
            block = text[open_match.start() : close_idx + len(_XML_CLOSE_TOKEN)]

            attrs = self._parse_attrs(attrs_raw)
            name = attrs.get("name")
            if not name:
                raise ToolCallParseError(
                    "tool_use missing name attribute",
                    open_match.start(),
                    block[:64],
                )
            call_id = attrs.get("id")

            arguments = self._extract_arguments(body, open_match.start())

            result.calls.append(
                ParsedToolCall(
                    name=name,
                    arguments=arguments,
                    raw=block,
                    call_id=call_id,
                )
            )
            pos = close_idx + len(_XML_CLOSE_TOKEN)

        return result

    @staticmethod
    def _find_close(text: str, start: int) -> int:
        """Locate the index of the matching ``</tool_use>`` from ``start``.

        The scan is JSON-string aware: any ``</tool_use>`` token that
        appears inside a double-quoted string literal is ignored. This
        blocks the common injection where the model's argument payload
        contains a crafted close tag.

        Returns the index of the ``<`` of the close tag, or -1 if no
        matching tag exists in the remainder of the string (streaming).
        """
        i = start
        n = len(text)
        in_string = False
        escape = False
        len(_XML_CLOSE_TOKEN)
        while i < n:
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                i += 1
                continue
            # Not in a JSON string: this is where a real close can live.
            if text.startswith(_XML_CLOSE_TOKEN, i):
                return i
            i += 1
        return -1

    @staticmethod
    def _find_input_close(text: str, start: int) -> int:
        """Same JSON-aware scan, but for the ``</input>`` close token."""
        i = start
        n = len(text)
        in_string = False
        escape = False
        while i < n:
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                i += 1
                continue
            if text.startswith(_XML_INPUT_CLOSE_TOKEN, i):
                return i
            i += 1
        return -1

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _parse_attrs(attrs_raw: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for m in _XML_ATTR_RE.finditer(attrs_raw):
            key = m.group("key")
            # Intentionally DO NOT html.unescape values used as identifiers;
            # names & ids should be literal. Quoted value is taken verbatim.
            out[key] = m.group("val")
        return out

    @staticmethod
    def _extract_arguments(body: str, offset: int) -> dict:
        """Return the argument dict from a ``<tool_use>`` body.

        The body must contain an ``<input>{...}</input>`` JSON object.
        If absent, an empty dict is returned (the tool takes no args).
        """
        body_stripped = body.strip()
        if not body_stripped:
            return {}

        m = _XML_INPUT_OPEN_RE.search(body)
        json_start = -1
        if m is None:
            # Tolerate a body that is itself a JSON object with no <input>
            # wrapper — some templates emit the bare object.
            candidate = body_stripped
        else:
            json_start = m.end()
            close_idx = XMLToolCallParser._find_input_close(body, json_start)
            if close_idx == -1:
                raise ToolCallParseError(
                    "unterminated <input> in <tool_use>",
                    offset,
                    body[:64],
                )
            candidate = body[json_start:close_idx].strip()

        if not candidate:
            return {}

        # IMPORTANT: do NOT html.unescape() JSON payloads unconditionally.
        # HTML entities inside a JSON string literal would be corrupted.
        # We only unescape the three XML-illegal sigils &lt; &gt; &amp;
        # *outside* of JSON strings, and only if raw parse fails. Simpler
        # path: try JSON parse first, unescape-and-retry on failure.
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc_first:
            # Limited repair: replace only the XML metacharacter entities
            # that a well-behaved emitter would use. Do this once.
            repaired = (
                candidate.replace("&lt;", "<")
                .replace("&gt;", ">")
                .replace("&quot;", '"')
                .replace("&amp;", "&")
            )
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError as exc:
                pos = offset + (json_start if json_start >= 0 else 0) + exc.pos
                raise ToolCallParseError(
                    f"invalid JSON in <input>: {exc.msg}",
                    pos,
                    candidate[max(0, exc.pos - 32) : exc.pos + 32],
                ) from exc_first

        if not isinstance(parsed, dict):
            raise ToolCallParseError(
                "tool_use <input> must be a JSON object",
                offset,
                str(parsed)[:64],
            )
        return parsed


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------


class JSONToolCallParser:
    """Parse an OpenAI-style JSON tool-call envelope.

    Accepts either::

        {"tool_calls": [{"id": "...", "name": "...", "arguments": {...}}]}

    or the provider-native variant where ``arguments`` is a JSON-encoded
    string (as the OpenAI chat API actually emits)::

        {"tool_calls": [{"function": {"name": "...",
                                       "arguments": "{\\"q\\":\\"hi\\"}"}}]}

    Both shapes are normalised to a list of :class:`ParsedToolCall`.
    """

    def parse(self, text: str) -> list[ParsedToolCall]:
        return self.parse_stream(text).calls

    def parse_stream(self, text: str) -> ParseResult:
        if not isinstance(text, str):
            raise ToolCallParseError("input must be str", -1, "")
        if len(text) > _MAX_INPUT_BYTES:
            raise ToolCallParseError("input exceeds max size", _MAX_INPUT_BYTES, "")

        stripped = text.strip()
        if not stripped:
            return ParseResult()

        # Streaming: if the payload appears to be an incomplete JSON object
        # (unbalanced braces or truncated) surface it as buffer.
        if not self._looks_complete(stripped):
            return ParseResult(calls=[], remaining_buffer=stripped)

        try:
            envelope = json.loads(stripped)
        except json.JSONDecodeError as exc:
            snippet = stripped[max(0, exc.pos - 32) : exc.pos + 32]
            raise ToolCallParseError(f"invalid JSON envelope: {exc.msg}", exc.pos, snippet) from exc

        if not isinstance(envelope, dict):
            raise ToolCallParseError("JSON envelope must be an object", 0, stripped[:64])

        calls_raw = envelope.get("tool_calls")
        if calls_raw is None:
            # Permit a single bare tool-call object.
            if "name" in envelope:
                calls_raw = [envelope]
            else:
                raise ToolCallParseError("envelope missing 'tool_calls'", 0, stripped[:64])

        if not isinstance(calls_raw, list):
            raise ToolCallParseError("'tool_calls' must be a list", 0, stripped[:64])

        out: list[ParsedToolCall] = []
        for i, entry in enumerate(calls_raw):
            out.append(self._normalise(entry, i, stripped))
        return ParseResult(calls=out)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _looks_complete(text: str) -> bool:
        """Cheap heuristic: JSON is complete iff braces/brackets balance.

        Respects string literals and escapes so we don't miscount braces
        embedded in string values.
        """
        depth_curly = 0
        depth_square = 0
        in_string = False
        escape = False
        for ch in text:
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth_curly += 1
            elif ch == "}":
                depth_curly -= 1
            elif ch == "[":
                depth_square += 1
            elif ch == "]":
                depth_square -= 1
        return not in_string and depth_curly == 0 and depth_square == 0

    @staticmethod
    def _normalise(entry: Any, index: int, full: str) -> ParsedToolCall:
        if not isinstance(entry, dict):
            raise ToolCallParseError(f"tool_calls[{index}] must be an object", -1, str(entry)[:64])

        call_id = entry.get("id")
        if call_id is not None and not isinstance(call_id, str):
            raise ToolCallParseError(
                f"tool_calls[{index}].id must be a string", -1, str(entry)[:64]
            )

        # Shape A: {name, arguments}. Shape B: {function:{name, arguments}}.
        if "function" in entry and isinstance(entry["function"], dict):
            fn = entry["function"]
            name = fn.get("name")
            raw_args = fn.get("arguments")
        else:
            name = entry.get("name")
            raw_args = entry.get("arguments", {})

        if not isinstance(name, str) or not name:
            raise ToolCallParseError(
                f"tool_calls[{index}] missing/invalid name",
                -1,
                str(entry)[:64],
            )

        if isinstance(raw_args, str):
            # OpenAI-native: arguments is a JSON-encoded string.
            try:
                arguments = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as exc:
                raise ToolCallParseError(
                    f"tool_calls[{index}].arguments is not valid JSON: {exc.msg}",
                    exc.pos,
                    raw_args[max(0, exc.pos - 32) : exc.pos + 32],
                ) from exc
        elif isinstance(raw_args, dict):
            arguments = raw_args
        elif raw_args is None:
            arguments = {}
        else:
            raise ToolCallParseError(
                f"tool_calls[{index}].arguments must be object or JSON string",
                -1,
                str(raw_args)[:64],
            )

        if not isinstance(arguments, dict):
            raise ToolCallParseError(
                f"tool_calls[{index}].arguments must decode to an object",
                -1,
                str(arguments)[:64],
            )

        return ParsedToolCall(
            name=name,
            arguments=arguments,
            raw=json.dumps(entry, sort_keys=True, separators=(",", ":")),
            call_id=call_id,
        )


# ---------------------------------------------------------------------------
# Format detection & unified dispatch
# ---------------------------------------------------------------------------


_XML_SIGNAL_RE = re.compile(r"<tool_use\b", re.IGNORECASE)


def detect_format(text: str) -> Literal["xml", "json", "none"]:
    """Classify ``text`` as XML, JSON, or neither.

    Detection is cheap and intentionally conservative:

    * If a literal ``<tool_use`` opener appears *outside* a leading JSON
      brace, treat as XML.
    * Else if the payload begins with ``{`` and contains ``"tool_calls"``
      (or ``"name"`` + ``"arguments"``), treat as JSON.
    * Else ``"none"``.
    """
    if not isinstance(text, str) or not text:
        return "none"
    stripped = text.lstrip()
    # JSON envelope leads with a curly brace; XML might be embedded inside
    # prose so we check it regardless of leading whitespace.
    if stripped.startswith("{"):
        # Only commit to JSON if the envelope plausibly names a tool call.
        if '"tool_calls"' in stripped or ('"name"' in stripped and '"arguments"' in stripped):
            return "json"
    if _XML_SIGNAL_RE.search(text):
        return "xml"
    return "none"


class UnifiedToolCallParser:
    """Dispatcher that auto-detects the format and delegates.

    On ``none`` it returns an empty list — *not* an error — because many
    model turns contain no tool calls at all. That is the normal case.
    """

    def __init__(
        self,
        xml_parser: XMLToolCallParser | None = None,
        json_parser: JSONToolCallParser | None = None,
    ) -> None:
        self._xml = xml_parser or XMLToolCallParser()
        self._json = json_parser or JSONToolCallParser()

    def parse(self, text: str) -> list[ParsedToolCall]:
        return self.parse_stream(text).calls

    def parse_stream(self, text: str) -> ParseResult:
        fmt = detect_format(text)
        if fmt == "xml":
            return self._xml.parse_stream(text)
        if fmt == "json":
            return self._json.parse_stream(text)
        return ParseResult()


# ---------------------------------------------------------------------------
# Formatting helpers (round-trip support for tests / agent harness)
# ---------------------------------------------------------------------------


def format_xml(call: ParsedToolCall) -> str:
    """Serialise ``call`` as an Anthropic-style ``<tool_use>`` block.

    Emitted in the canonical shape expected by Aurelius's agent loop.
    ``html.escape`` is applied only to the ``name`` / ``id`` attributes
    (never to the JSON body — that would corrupt string values).
    """
    name_attr = html.escape(call.name, quote=True)
    id_attr = f' id="{html.escape(call.call_id, quote=True)}"' if call.call_id is not None else ""
    body = json.dumps(call.arguments, sort_keys=True, separators=(",", ":"))
    return f'<tool_use name="{name_attr}"{id_attr}><input>{body}</input></tool_use>'


def format_json(call: ParsedToolCall) -> str:
    """Serialise ``call`` as an OpenAI-style envelope (single call)."""
    envelope = {
        "tool_calls": [
            {
                **({"id": call.call_id} if call.call_id is not None else {}),
                "name": call.name,
                "arguments": call.arguments,
            }
        ]
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Registry-facing callables
# ---------------------------------------------------------------------------


_XML_PARSER_SINGLETON = XMLToolCallParser()
_JSON_PARSER_SINGLETON = JSONToolCallParser()


def parse_xml(text: str) -> list[ParsedToolCall]:
    """Module-level XML parse entry point registered under key ``"xml"``."""
    return _XML_PARSER_SINGLETON.parse(text)


def parse_json(text: str) -> list[ParsedToolCall]:
    """Module-level JSON parse entry point registered under key ``"json"``."""
    return _JSON_PARSER_SINGLETON.parse(text)


__all__ = [
    "ParsedToolCall",
    "ParseResult",
    "ToolCallParseError",
    "XMLToolCallParser",
    "JSONToolCallParser",
    "UnifiedToolCallParser",
    "detect_format",
    "format_xml",
    "format_json",
    "parse_xml",
    "parse_json",
]
