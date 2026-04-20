"""Minimal YARA-like rule engine for the Aurelius cybersecurity surface.

Implements a simplified, pure-stdlib subset of the YARA pattern-matching
language for malware/signature-style scanning of byte buffers or text.

Supported features:
    - Rule parser (``rule NAME { strings: ... condition: ... }``)
    - Text strings ("literal"), hex strings ({ FF 00 ?? }), regex
      strings (/pattern/)
    - Boolean conditions (``and``, ``or``, ``not``), presence ($s1),
      string count (``#s1 >= 3`` or ``$s1 >= 3``), and ``filesize``
      comparisons (e.g. ``filesize < 1MB``).
    - Metadata block (``meta: author = "x" severity = 5``).

This is an intentionally compact re-implementation -- no ``yara-python``
dependency, no imports/globals/modules from the full YARA grammar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


class YaraParseError(ValueError):
    """Raised when a rule cannot be parsed."""


class YaraEvalError(ValueError):
    """Raised when a condition cannot be evaluated."""


@dataclass
class YaraRule:
    """A parsed YARA rule."""

    name: str
    strings: dict[str, tuple[str, bytes | str]]
    condition: str
    meta: dict = field(default_factory=dict)


@dataclass
class YaraMatch:
    """A rule match on a buffer."""

    rule_name: str
    matched_strings: list[tuple[str, int, bytes | str]]
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


_RULE_HEADER = re.compile(r"\brule\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")


class YaraRuleParser:
    """Parses YARA-like rule text into :class:`YaraRule` objects."""

    def parse(self, rule_text: str) -> list[YaraRule]:
        rules: list[YaraRule] = []
        i = 0
        text = rule_text
        while i < len(text):
            match = _RULE_HEADER.search(text, i)
            if match is None:
                # Trailing whitespace/comments are fine; anything else is not.
                tail = text[i:].strip()
                if tail:
                    raise YaraParseError(
                        f"unexpected trailing content: {tail[:40]!r}"
                    )
                break
            name = match.group(1)
            body_start = match.end()
            body_end = self._find_matching_brace(text, body_start - 1)
            body = text[body_start:body_end]
            rule = self._parse_body(name, body)
            rules.append(rule)
            i = body_end + 1
        if not rules:
            raise YaraParseError("no rules found")
        return rules

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _find_matching_brace(text: str, open_idx: int) -> int:
        depth = 0
        in_string = False
        in_regex = False
        escape = False
        i = open_idx
        while i < len(text):
            ch = text[i]
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif in_string:
                if ch == '"':
                    in_string = False
            elif in_regex:
                if ch == "/":
                    in_regex = False
            elif ch == '"':
                in_string = True
            elif ch == "/" and depth >= 1:
                # Only treat as regex inside a rule body.
                in_regex = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
            i += 1
        raise YaraParseError("unbalanced braces in rule body")

    def _parse_body(self, name: str, body: str) -> YaraRule:
        sections = self._split_sections(body)
        meta = self._parse_meta(sections.get("meta", ""))
        strings = self._parse_strings(sections.get("strings", ""))
        condition = sections.get("condition", "").strip()
        if not condition:
            raise YaraParseError(f"rule {name!r} has no condition")
        return YaraRule(name=name, strings=strings, condition=condition, meta=meta)

    @staticmethod
    def _split_sections(body: str) -> dict[str, str]:
        """Split a body into meta/strings/condition sections."""
        # Find section keyword positions respecting quoted strings/regex.
        keywords = ("meta", "strings", "condition")
        positions: list[tuple[int, str]] = []
        in_string = False
        in_regex = False
        escape = False
        i = 0
        while i < len(body):
            ch = body[i]
            if escape:
                escape = False
                i += 1
                continue
            if ch == "\\":
                escape = True
                i += 1
                continue
            if in_string:
                if ch == '"':
                    in_string = False
                i += 1
                continue
            if in_regex:
                if ch == "/":
                    in_regex = False
                i += 1
                continue
            if ch == '"':
                in_string = True
                i += 1
                continue
            if ch == "/":
                in_regex = True
                i += 1
                continue
            # Keyword match must be at a word boundary and followed by ':'.
            matched = False
            for kw in keywords:
                if body.startswith(kw, i) and (i == 0 or not body[i - 1].isalnum()):
                    after = i + len(kw)
                    # skip whitespace
                    j = after
                    while j < len(body) and body[j] in " \t\r\n":
                        j += 1
                    if j < len(body) and body[j] == ":":
                        positions.append((i, kw))
                        i = j + 1
                        matched = True
                        break
            if not matched:
                i += 1

        if not positions:
            raise YaraParseError("no sections found in rule body")
        sections: dict[str, str] = {}
        for idx, (start, kw) in enumerate(positions):
            content_start = start + len(kw)
            # advance past colon
            colon = body.find(":", content_start)
            content_start = colon + 1
            content_end = (
                positions[idx + 1][0] if idx + 1 < len(positions) else len(body)
            )
            sections[kw] = body[content_start:content_end]
        return sections

    @staticmethod
    def _parse_meta(meta_text: str) -> dict:
        meta: dict = {}
        if not meta_text.strip():
            return meta
        # key = value pairs; value may be quoted string, int, or bool.
        pattern = re.compile(
            r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*("(?:[^"\\]|\\.)*"|true|false|-?\d+)'
        )
        for m in pattern.finditer(meta_text):
            key = m.group(1)
            raw = m.group(2)
            if raw.startswith('"'):
                value: object = bytes(raw[1:-1], "utf-8").decode("unicode_escape")
            elif raw == "true":
                value = True
            elif raw == "false":
                value = False
            else:
                value = int(raw)
            meta[key] = value
        return meta

    def _parse_strings(self, strings_text: str) -> dict[str, tuple[str, bytes | str]]:
        out: dict[str, tuple[str, bytes | str]] = {}
        text = strings_text
        i = 0
        while i < len(text):
            # Find next '$'
            while i < len(text) and text[i] != "$":
                i += 1
            if i >= len(text):
                break
            # Parse identifier
            j = i + 1
            while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                j += 1
            ident = text[i + 1 : j]
            if not ident:
                raise YaraParseError("empty string identifier")
            # Skip whitespace and '='
            while j < len(text) and text[j] in " \t\r\n":
                j += 1
            if j >= len(text) or text[j] != "=":
                raise YaraParseError(
                    f"expected '=' after ${ident}"
                )
            j += 1
            while j < len(text) and text[j] in " \t\r\n":
                j += 1
            if j >= len(text):
                raise YaraParseError(f"missing value for ${ident}")
            ch = text[j]
            if ch == '"':
                end = self._find_string_end(text, j)
                raw = text[j + 1 : end]
                value = bytes(raw, "utf-8").decode("unicode_escape")
                out[ident] = ("text", value)
                i = end + 1
            elif ch == "{":
                end = text.find("}", j)
                if end == -1:
                    raise YaraParseError(f"unclosed hex string for ${ident}")
                hex_body = text[j + 1 : end]
                out[ident] = ("hex", hex_body.strip())
                i = end + 1
            elif ch == "/":
                end = self._find_regex_end(text, j)
                pattern = text[j + 1 : end]
                out[ident] = ("regex", pattern)
                i = end + 1
            else:
                raise YaraParseError(
                    f"unknown string type for ${ident}: {ch!r}"
                )
        return out

    @staticmethod
    def _find_string_end(text: str, start: int) -> int:
        # start points at opening quote
        i = start + 1
        escape = False
        while i < len(text):
            ch = text[i]
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                return i
            i += 1
        raise YaraParseError("unterminated text string")

    @staticmethod
    def _find_regex_end(text: str, start: int) -> int:
        i = start + 1
        escape = False
        while i < len(text):
            ch = text[i]
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "/":
                return i
            i += 1
        raise YaraParseError("unterminated regex string")


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _hex_to_regex(hex_body: str) -> re.Pattern[bytes]:
    """Translate a YARA-style hex body into a byte-level regex."""
    tokens = hex_body.split()
    parts: list[bytes] = []
    for tok in tokens:
        if len(tok) != 2:
            raise YaraParseError(f"hex token must be 2 chars: {tok!r}")
        if tok == "??":
            parts.append(b".")
        elif tok[0] == "?" and tok[1] != "?":
            # High-nibble wildcard: match any byte whose low nibble matches.
            low = tok[1].upper()
            parts.append(
                b"[" + b"".join(
                    b"\\x%02X" % ((hi << 4) | int(low, 16)) for hi in range(16)
                ) + b"]"
            )
        elif tok[1] == "?" and tok[0] != "?":
            hi = tok[0].upper()
            parts.append(
                b"[" + b"".join(
                    b"\\x%02X" % ((int(hi, 16) << 4) | lo) for lo in range(16)
                ) + b"]"
            )
        else:
            try:
                byte = int(tok, 16)
            except ValueError as exc:
                raise YaraParseError(f"invalid hex token: {tok!r}") from exc
            parts.append(b"\\x%02X" % byte)
    return re.compile(b"".join(parts), re.DOTALL)


def _find_all(
    kind: str, value: bytes | str, data: bytes
) -> list[tuple[int, bytes | str]]:
    """Return all (offset, matched_bytes) hits for a single string."""
    hits: list[tuple[int, bytes | str]] = []
    if kind == "text":
        needle = value.encode("utf-8") if isinstance(value, str) else value
        if not needle:
            return hits
        start = 0
        while True:
            idx = data.find(needle, start)
            if idx == -1:
                break
            hits.append((idx, needle))
            start = idx + 1
    elif kind == "hex":
        pat = _hex_to_regex(value if isinstance(value, str) else value.decode("ascii"))
        for m in pat.finditer(data):
            hits.append((m.start(), m.group(0)))
    elif kind == "regex":
        pat_str = value if isinstance(value, str) else value.decode("utf-8")
        pat = re.compile(pat_str.encode("utf-8"), re.DOTALL)
        for m in pat.finditer(data):
            hits.append((m.start(), m.group(0)))
    else:
        raise YaraEvalError(f"unknown string kind: {kind!r}")
    return hits


# ---------------------------------------------------------------------------
# Condition evaluator
# ---------------------------------------------------------------------------


# Tokens: identifiers, numbers (with optional KB/MB suffix), operators.
_TOKEN_RE = re.compile(
    r"""
    \s+                                  # whitespace
  | (?P<size>\d+(?:KB|MB|GB|B)?)         # size literal
  | (?P<ident>[\#\$][A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op><=|>=|==|!=|<|>|=|\(|\)|,)
    """,
    re.VERBOSE,
)


def _tokenize_condition(condition: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    while i < len(condition):
        m = _TOKEN_RE.match(condition, i)
        if m is None:
            raise YaraEvalError(
                f"unexpected token at position {i}: {condition[i:i+16]!r}"
            )
        if m.group("size") is not None:
            tokens.append(("SIZE", m.group("size")))
        elif m.group("ident") is not None:
            ident = m.group("ident")
            low = ident.lower()
            if low in ("and", "or", "not", "true", "false", "filesize"):
                tokens.append((low.upper(), low))
            else:
                tokens.append(("IDENT", ident))
        elif m.group("op") is not None:
            tokens.append(("OP", m.group("op")))
        i = m.end()
    return tokens


def _parse_size(literal: str) -> int:
    multipliers = {"KB": 1024, "MB": 1024 * 1024, "GB": 1024 ** 3, "B": 1}
    for suffix, mult in multipliers.items():
        if literal.endswith(suffix):
            return int(literal[: -len(suffix)]) * mult
    return int(literal)


class _ConditionEvaluator:
    """Recursive-descent evaluator for the condition mini-language."""

    def __init__(
        self,
        tokens: list[tuple[str, str]],
        hits: dict[str, list[tuple[int, bytes | str]]],
        data_len: int,
    ) -> None:
        self.tokens = tokens
        self.pos = 0
        self.hits = hits
        self.data_len = data_len

    def parse(self) -> bool:
        result = self._parse_or()
        if self.pos != len(self.tokens):
            raise YaraEvalError(
                f"unexpected token at end: {self.tokens[self.pos]!r}"
            )
        return bool(result)

    # or > and > not > primary
    def _parse_or(self) -> bool:
        left = self._parse_and()
        while self._peek() == ("OR", "or"):
            self.pos += 1
            right = self._parse_and()
            left = left or right
        return left

    def _parse_and(self) -> bool:
        left = self._parse_not()
        while self._peek() == ("AND", "and"):
            self.pos += 1
            right = self._parse_not()
            left = left and right
        return left

    def _parse_not(self) -> bool:
        if self._peek() == ("NOT", "not"):
            self.pos += 1
            return not self._parse_not()
        return self._parse_primary()

    def _parse_primary(self) -> bool:
        tok = self._peek()
        if tok is None:
            raise YaraEvalError("unexpected end of condition")
        kind, value = tok
        if kind == "OP" and value == "(":
            self.pos += 1
            result = self._parse_or()
            if self._peek() != ("OP", ")"):
                raise YaraEvalError("missing ')'")
            self.pos += 1
            return result
        if kind == "TRUE":
            self.pos += 1
            return True
        if kind == "FALSE":
            self.pos += 1
            return False
        if kind == "FILESIZE":
            self.pos += 1
            return self._parse_comparison(self.data_len)
        if kind == "IDENT" and value.startswith("$"):
            self.pos += 1
            ident = value[1:]
            count = len(self.hits.get(ident, []))
            # Optionally: "$s1 >= 3" style (count comparison).
            if self._peek_is_comparator():
                return self._parse_comparison(count)
            return count > 0
        if kind == "IDENT" and value.startswith("#"):
            self.pos += 1
            ident = value[1:]
            count = len(self.hits.get(ident, []))
            return self._parse_comparison(count)
        raise YaraEvalError(f"unexpected token: {tok!r}")

    def _parse_comparison(self, left_value: int) -> bool:
        tok = self._peek()
        if tok is None or tok[0] != "OP" or tok[1] not in (
            "<",
            "<=",
            ">",
            ">=",
            "==",
            "!=",
        ):
            # Bare reference (non-numeric). For filesize we *require* comparator.
            raise YaraEvalError("expected comparison operator")
        op = tok[1]
        self.pos += 1
        right_tok = self._peek()
        if right_tok is None:
            raise YaraEvalError("expected value after comparator")
        rkind, rvalue = right_tok
        if rkind == "SIZE":
            right_value = _parse_size(rvalue)
        elif rkind == "IDENT" and rvalue.startswith("#"):
            right_value = len(self.hits.get(rvalue[1:], []))
        else:
            raise YaraEvalError(f"expected size literal, got {right_tok!r}")
        self.pos += 1
        return self._cmp(left_value, op, right_value)

    @staticmethod
    def _cmp(a: int, op: str, b: int) -> bool:
        return {
            "<": a < b,
            "<=": a <= b,
            ">": a > b,
            ">=": a >= b,
            "==": a == b,
            "!=": a != b,
        }[op]

    def _peek(self) -> tuple[str, str] | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _peek_is_comparator(self) -> bool:
        tok = self._peek()
        return tok is not None and tok[0] == "OP" and tok[1] in (
            "<",
            "<=",
            ">",
            ">=",
            "==",
            "!=",
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class YaraRuleEngine:
    """A collection of compiled rules that can be scanned against a buffer."""

    def __init__(self, rules: list[YaraRule] | None = None) -> None:
        self._rules: list[YaraRule] = []
        self._parser = YaraRuleParser()
        if rules:
            for rule in rules:
                self.add_rule(rule)

    def add_rule(self, rule: YaraRule) -> None:
        if not isinstance(rule, YaraRule):
            raise TypeError("rule must be a YaraRule instance")
        self._rules.append(rule)

    def compile(self, rule_text: str) -> None:
        """Parse ``rule_text`` and append all rules found to this engine."""
        for rule in self._parser.parse(rule_text):
            self.add_rule(rule)

    @property
    def rules(self) -> list[YaraRule]:
        return list(self._rules)

    def scan(self, data: bytes | str) -> list[YaraMatch]:
        if isinstance(data, str):
            buf = data.encode("utf-8")
        elif isinstance(data, (bytes, bytearray)):
            buf = bytes(data)
        else:
            raise TypeError("data must be bytes or str")

        matches: list[YaraMatch] = []
        for rule in self._rules:
            hits: dict[str, list[tuple[int, bytes | str]]] = {}
            for ident, (kind, value) in rule.strings.items():
                hits[ident] = _find_all(kind, value, buf)
            tokens = _tokenize_condition(rule.condition)
            try:
                ok = _ConditionEvaluator(tokens, hits, len(buf)).parse()
            except YaraEvalError:
                raise
            if ok:
                matched: list[tuple[str, int, bytes | str]] = []
                for ident, occ in hits.items():
                    for offset, blob in occ:
                        matched.append((ident, offset, blob))
                matched.sort(key=lambda t: (t[1], t[0]))
                matches.append(
                    YaraMatch(
                        rule_name=rule.name,
                        matched_strings=matched,
                        meta=dict(rule.meta),
                    )
                )
        return matches

    def scan_file(self, path: str) -> list[YaraMatch]:
        with open(path, "rb") as fh:
            data = fh.read()
        return self.scan(data)


__all__ = [
    "YaraRule",
    "YaraMatch",
    "YaraRuleParser",
    "YaraRuleEngine",
    "YaraParseError",
    "YaraEvalError",
]


# Silence unused-import style linters when ``Iterable`` is only re-exported.
_ = Iterable
