"""PII (Personally Identifiable Information) detector and redactor.

This module is a stdlib-only, deterministic detector for personally identifiable
information. It sits on the same guard layer as the jailbreak detector and
prompt-injection scanner but has a different job: find spans of text that look
like emails, phones, SSNs, credit cards, IPs, MACs, passports, IBANs, dates of
birth, street addresses, and API tokens, and either surface them as structured
matches or replace them in the source text.

The algorithmic catalogue is inspired by Microsoft Presidio's open-source
recognizer set (pattern-based detection + validation passes such as Luhn for
PAN and mod-97 for IBAN). Only the algorithms are referenced; no third-party
code or packages are imported.

Design principles
-----------------
* Pure ``re`` + :mod:`ipaddress` — no heavy ML, no tokenizer, no network.
* Deterministic: same text in, same matches out, in span order.
* Never silently drops a misconfiguration: an unknown ``redaction_mode`` raises
  ``ValueError`` at construction time, as does an unknown entry in ``types``.
* All input is untrusted. The detector never ``eval``s, never decodes, and
  treats bytes / control characters as ordinary characters (the caller is
  responsible for upstream encoding).
* Overlap resolution prefers higher confidence, then longer span, then the
  earlier start position — this makes nested matches (e.g. a credit-card
  number sitting inside what would otherwise scan as a long digit run) stable
  and explainable.

Public API
----------
:class:`PIIMatch`, :class:`PIIResult`, :class:`PIIDetector`, plus two small
utilities — :func:`luhn_valid` and :func:`iban_valid` — that are used
internally but exposed so tests and downstream filters can reuse them without
rebuilding the logic.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass, field
from typing import Iterable


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PIIMatch:
    """A single detected PII span.

    Attributes
    ----------
    type:
        Canonical type label, one of :data:`PII_TYPES`.
    value:
        The substring from the source text (kept verbatim — no normalisation).
    span:
        ``(start, end)`` inclusive-exclusive offsets into the source text.
    confidence:
        Score in ``[0, 1]``. Pattern hits without a secondary validation
        (e.g. date of birth, street address heuristic) are assigned lower
        confidence than checksummed hits (credit card, IBAN, SSN).
    """

    type: str
    value: str
    span: tuple[int, int]
    confidence: float


@dataclass
class PIIResult:
    """Result of a redaction pass."""

    matches: list[PIIMatch] = field(default_factory=list)
    redacted_text: str = ""


# ---------------------------------------------------------------------------
# Type catalogue
# ---------------------------------------------------------------------------

PII_TYPES: tuple[str, ...] = (
    "email",
    "phone",
    "ssn",
    "credit_card",
    "ipv4",
    "ipv6",
    "mac",
    "passport",
    "iban",
    "dob",
    "address",
    "api_key",
)

_REDACTION_MODES: frozenset[str] = frozenset({"mask", "placeholder", "remove"})


# ---------------------------------------------------------------------------
# Checksum utilities
# ---------------------------------------------------------------------------


def luhn_valid(digits: str) -> bool:
    """Return ``True`` iff ``digits`` (a decimal string) passes the Luhn mod-10
    checksum used by PAN (primary account number) schemes.

    Non-digit characters are rejected — strip separators before calling. Length
    is bounded to the standard PAN range ``[12, 19]``.
    """

    if not digits.isdigit():
        return False
    n = len(digits)
    if n < 12 or n > 19:
        return False
    total = 0
    # Iterate right-to-left; every second digit is doubled.
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i & 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def iban_valid(iban: str) -> bool:
    """Simplified IBAN mod-97 check (ISO-13616).

    Whitespace is stripped; the structure must be 2 letters + 2 digits + up to
    30 alphanumerics. The check digits are validated by rearranging and
    converting to the canonical integer form, then taking ``mod 97 == 1``.
    """

    s = re.sub(r"\s+", "", iban).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    rearranged = s[4:] + s[:4]
    # Convert letters to digits: A=10, B=11, ..., Z=35.
    buf = []
    for ch in rearranged:
        if ch.isdigit():
            buf.append(ch)
        else:
            buf.append(str(ord(ch) - 55))
    try:
        return int("".join(buf)) % 97 == 1
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Regex catalogue
# ---------------------------------------------------------------------------

# Email — RFC-5322 simplified. We intentionally reject leading/trailing dots
# and consecutive dots in the local part and require at least one dot in the
# domain (this rules out ``user@localhost`` but that is rarely PII).
_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+\-])"
    r"[A-Za-z0-9](?:[A-Za-z0-9._%+\-]{0,62}[A-Za-z0-9])?"
    r"@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,24}"
    r"(?![A-Za-z0-9])"
)

# US phone — accepts ``(415) 555-1212``, ``415-555-1212``, ``415.555.1212``,
# ``415 555 1212``, and the ``+1`` international form. The leading boundary
# deliberately excludes ``:`` and digits so version strings and timestamps do
# not match.
_PHONE = re.compile(
    r"(?<![\d.\-])"
    r"(?:\+?1[\s.\-]?)?"
    r"(?:\(\d{3}\)|\d{3})"
    r"[\s.\-]?"
    r"\d{3}"
    r"[\s.\-]"
    r"\d{4}"
    r"(?!\d)"
)

# SSN — AAA-GG-SSSS with the standard invalid-area blocks (000, 666, 900-999)
# excluded and the group/serial zero-blocks excluded. Matches the literal
# hyphenated form only; SSN without hyphens is ambiguous with many other
# 9-digit strings and is handled only as part of the API-key heuristic.
_SSN = re.compile(
    r"(?<!\d)"
    r"(?!000|666|9\d\d)\d{3}-"
    r"(?!00)\d{2}-"
    r"(?!0000)\d{4}"
    r"(?!\d)"
)

# Credit card — 13 to 19 digits with optional single-character separators
# between groups. The Luhn check below is the source of truth for validity.
_CREDIT_CARD = re.compile(
    r"(?<!\d)"
    r"(?:\d[ \-]?){12,18}\d"
    r"(?!\d)"
)

# IPv4 — four dotted octets 0-255. The boundary checks prevent ``1.2.3.4`` in
# a version string from matching (we guard against an immediately-preceding
# ``v``/``V`` or trailing alnum/``.``).
_IPV4 = re.compile(
    r"(?<![A-Za-z0-9.])"
    r"(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?![A-Za-z0-9.])"
)

# IPv6 — a permissive regex; the :mod:`ipaddress` module does the final
# validation. We require at least one ``::`` or two ``:`` groups so random
# colon-laden text doesn't match.
_IPV6 = re.compile(
    r"(?<![A-Za-z0-9:])"
    r"(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}"
    r"|::(?:[0-9A-Fa-f]{1,4}:){0,6}[0-9A-Fa-f]{1,4}"
    r"|(?:[0-9A-Fa-f]{1,4}:){1,7}:"
)

# MAC — six hex octets separated by ``:`` or ``-``.
_MAC = re.compile(
    r"(?<![0-9A-Fa-f:\-])"
    r"(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}"
    r"(?![0-9A-Fa-f:\-])"
)

# US passport — single letter followed by 8 digits. Older books used 9 digits
# without the letter, but the modern format is letter+8.
_PASSPORT = re.compile(r"(?<![A-Za-z0-9])[A-Z]\d{8}(?![A-Za-z0-9])")

# IBAN — country code + 2 digits + up to 30 alnum. Validated by mod-97.
_IBAN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){11,30}"
    r"(?![A-Za-z0-9])"
)

# Date of birth — mm/dd/yyyy, dd-mm-yyyy, yyyy-mm-dd, and ``January 1, 1970``.
# We deliberately keep confidence low; without context we can't distinguish a
# DOB from any other date.
_DOB_NUMERIC = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}"
    r"|"
    r"(?:19|20)\d{2}[/\-](?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])"
    r")"
    r"(?!\d)"
)
_DOB_WORD = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},\s+(?:19|20)\d{2}\b"
)

# US street address — heuristic only: number + street words + a common suffix.
_ADDRESS = re.compile(
    r"\b\d{1,6}\s+"
    r"(?:[A-Z][A-Za-z0-9.\-]*\s+){1,5}"
    r"(?:Street|St\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Road|Rd\.?|Drive|Dr\.?|"
    r"Lane|Ln\.?|Court|Ct\.?|Circle|Cir\.?|Way|Place|Pl\.?|Terrace|Ter\.?|"
    r"Parkway|Pkwy\.?|Highway|Hwy\.?)"
    r"\b"
)

# API keys / tokens. We keep three well-known prefixes with high confidence
# and one generic high-entropy fallback with low confidence.
_AWS_KEY = re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])")
_GITHUB_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{36,255}(?![A-Za-z0-9_])"
)
_SLACK_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])xox[baprs]-[A-Za-z0-9\-]{10,}(?![A-Za-z0-9_])"
)
# Generic high-entropy alnum string, 32+ chars. Confidence is intentionally low
# so it never out-votes a checksummed hit.
_GENERIC_KEY = re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{32,}(?![A-Za-z0-9])")


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class PIIDetector:
    """Pattern + checksum based PII detector and redactor.

    Parameters
    ----------
    redaction_mode:
        ``"mask"`` replaces each match with ``"****"``; ``"placeholder"``
        replaces it with ``"<TYPE>"`` (e.g. ``"<EMAIL>"``); ``"remove"`` drops
        the matched substring entirely.
    types:
        Optional allow-list of PII type labels. ``None`` (the default) enables
        every type in :data:`PII_TYPES`. Unknown labels raise ``ValueError``.
    confidence_threshold:
        Matches whose confidence is strictly below this threshold are filtered
        out. Accepts ``[0, 1]``.
    """

    __slots__ = ("_redaction_mode", "_types", "_threshold")

    def __init__(
        self,
        redaction_mode: str = "placeholder",
        types: list[str] | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        if redaction_mode not in _REDACTION_MODES:
            raise ValueError(
                f"unknown redaction_mode {redaction_mode!r}; expected one of "
                f"{sorted(_REDACTION_MODES)}"
            )
        if not 0.0 <= float(confidence_threshold) <= 1.0:
            raise ValueError(
                "confidence_threshold must lie in [0, 1], got "
                f"{confidence_threshold}"
            )
        if types is None:
            enabled = set(PII_TYPES)
        else:
            enabled = set()
            for t in types:
                if t not in PII_TYPES:
                    raise ValueError(
                        f"unknown PII type {t!r}; known types: {PII_TYPES}"
                    )
                enabled.add(t)
        self._redaction_mode = redaction_mode
        self._types = frozenset(enabled)
        self._threshold = float(confidence_threshold)

    # -- public API --------------------------------------------------------

    def detect(self, text: str) -> list[PIIMatch]:
        """Return the PII matches in ``text``, sorted by span start."""

        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        raw: list[PIIMatch] = []
        self._scan_all(text, raw)
        filtered = [m for m in raw if m.confidence >= self._threshold]
        resolved = _resolve_overlaps(filtered)
        resolved.sort(key=lambda m: m.span[0])
        return resolved

    def redact(self, text: str) -> PIIResult:
        """Detect and redact ``text``. Returns matches + redacted string."""

        matches = self.detect(text)
        if not matches:
            return PIIResult(matches=[], redacted_text=text)
        # Apply redactions right-to-left so earlier spans stay valid.
        out = text
        for m in sorted(matches, key=lambda x: x.span[0], reverse=True):
            start, end = m.span
            out = out[:start] + self._render(m) + out[end:]
        return PIIResult(matches=matches, redacted_text=out)

    def has_pii(self, text: str, types: list[str] | None = None) -> bool:
        """Fast path: ``True`` iff any match exists, optionally restricted."""

        if types is not None:
            wanted = set(types)
            for t in wanted:
                if t not in PII_TYPES:
                    raise ValueError(f"unknown PII type {t!r}")
        else:
            wanted = None
        for m in self.detect(text):
            if wanted is None or m.type in wanted:
                return True
        return False

    # -- rendering ---------------------------------------------------------

    def _render(self, m: PIIMatch) -> str:
        mode = self._redaction_mode
        if mode == "mask":
            return "****"
        if mode == "placeholder":
            return f"<{m.type.upper()}>"
        if mode == "remove":
            return ""
        # Unreachable — __init__ validates the mode.
        raise ValueError(f"unknown redaction_mode {mode!r}")

    # -- scan drivers ------------------------------------------------------

    def _scan_all(self, text: str, out: list[PIIMatch]) -> None:
        if "email" in self._types:
            for mo in _EMAIL.finditer(text):
                out.append(PIIMatch("email", mo.group(0), mo.span(), 0.95))
        if "phone" in self._types:
            for mo in _PHONE.finditer(text):
                out.append(PIIMatch("phone", mo.group(0), mo.span(), 0.80))
        if "ssn" in self._types:
            for mo in _SSN.finditer(text):
                out.append(PIIMatch("ssn", mo.group(0), mo.span(), 0.95))
        if "credit_card" in self._types:
            for mo in _CREDIT_CARD.finditer(text):
                raw = mo.group(0)
                digits = re.sub(r"[ \-]", "", raw)
                if luhn_valid(digits):
                    out.append(
                        PIIMatch("credit_card", raw, mo.span(), 0.99)
                    )
        if "ipv4" in self._types:
            for mo in _IPV4.finditer(text):
                try:
                    ipaddress.IPv4Address(mo.group(0))
                except ValueError:
                    continue
                out.append(PIIMatch("ipv4", mo.group(0), mo.span(), 0.85))
        if "ipv6" in self._types:
            for mo in _IPV6.finditer(text):
                try:
                    ipaddress.IPv6Address(mo.group(0))
                except ValueError:
                    continue
                out.append(PIIMatch("ipv6", mo.group(0), mo.span(), 0.90))
        if "mac" in self._types:
            for mo in _MAC.finditer(text):
                out.append(PIIMatch("mac", mo.group(0), mo.span(), 0.85))
        if "passport" in self._types:
            for mo in _PASSPORT.finditer(text):
                out.append(PIIMatch("passport", mo.group(0), mo.span(), 0.60))
        if "iban" in self._types:
            for mo in _IBAN.finditer(text):
                if iban_valid(mo.group(0)):
                    out.append(PIIMatch("iban", mo.group(0), mo.span(), 0.97))
        if "dob" in self._types:
            for mo in _DOB_NUMERIC.finditer(text):
                out.append(PIIMatch("dob", mo.group(0), mo.span(), 0.55))
            for mo in _DOB_WORD.finditer(text):
                out.append(PIIMatch("dob", mo.group(0), mo.span(), 0.60))
        if "address" in self._types:
            for mo in _ADDRESS.finditer(text):
                out.append(PIIMatch("address", mo.group(0), mo.span(), 0.55))
        if "api_key" in self._types:
            for mo in _AWS_KEY.finditer(text):
                out.append(PIIMatch("api_key", mo.group(0), mo.span(), 0.98))
            for mo in _GITHUB_TOKEN.finditer(text):
                out.append(PIIMatch("api_key", mo.group(0), mo.span(), 0.98))
            for mo in _SLACK_TOKEN.finditer(text):
                out.append(PIIMatch("api_key", mo.group(0), mo.span(), 0.90))
            for mo in _GENERIC_KEY.finditer(text):
                val = mo.group(0)
                if _looks_like_token(val):
                    out.append(
                        PIIMatch("api_key", val, mo.span(), 0.55)
                    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _looks_like_token(s: str) -> bool:
    """Entropy-ish check: token must mix at least two character classes.

    This filters out long runs of a single class (a 40-digit string is almost
    certainly not an API token; a 40-letter string is almost certainly a word)
    while keeping true mixed-class tokens like base64url blobs.
    """

    has_lower = any(c.islower() for c in s)
    has_upper = any(c.isupper() for c in s)
    has_digit = any(c.isdigit() for c in s)
    classes = sum((has_lower, has_upper, has_digit))
    return classes >= 2


def _resolve_overlaps(matches: Iterable[PIIMatch]) -> list[PIIMatch]:
    """Greedy overlap resolution.

    Sorted by (confidence desc, length desc, start asc). A match is accepted
    if it doesn't overlap any already-accepted span.
    """

    def key(m: PIIMatch) -> tuple[float, int, int]:
        return (
            -m.confidence,
            -(m.span[1] - m.span[0]),
            m.span[0],
        )

    ordered = sorted(matches, key=key)
    accepted: list[PIIMatch] = []
    for m in ordered:
        s, e = m.span
        conflict = False
        for a in accepted:
            as_, ae = a.span
            if s < ae and as_ < e:
                conflict = True
                break
        if not conflict:
            accepted.append(m)
    return accepted


__all__ = [
    "PIIMatch",
    "PIIResult",
    "PIIDetector",
    "PII_TYPES",
    "luhn_valid",
    "iban_valid",
]
