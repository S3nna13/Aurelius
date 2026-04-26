"""CWE remediation guidance lookup with prompt-injection-safe formatting.

Adapted from BUGBOUNTY_AGENT/reporting/remediation.py. Stdlib-only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RemediationEntry:
    """A single CWE remediation record."""

    cwe_id: str
    title: str
    guidance: str
    references: list[str]


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"^SYSTEM\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^ASSISTANT\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^USER\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"you\s+are\s+now\s+a\s+different", re.IGNORECASE),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?secrets?", re.IGNORECASE),
]


def _sanitize_for_prompt(text: str) -> str:
    """Strip C0 control chars and redact common prompt injection patterns."""
    if not text:
        return ""
    out = _CONTROL_CHAR_RE.sub("", text)
    for pattern in _INJECTION_PATTERNS:
        out = pattern.sub("[REDACTED]", out)
    return out


_MITRE = "https://cwe.mitre.org/data/definitions"
_OWASP = "https://owasp.org/www-community/attacks"


CWE_REMEDIATION_MAP: dict[str, RemediationEntry] = {
    "CWE-79": RemediationEntry(
        cwe_id="CWE-79",
        title="Cross-site Scripting (XSS)",
        guidance=(
            "Encode all user input before rendering in HTML. Use Content-Security-Policy "
            "headers. Implement input validation with allowlists."
        ),
        references=[f"{_MITRE}/79.html", f"{_OWASP}/xss/"],
    ),
    "CWE-89": RemediationEntry(
        cwe_id="CWE-89",
        title="SQL Injection",
        guidance=(
            "Use parameterized queries or prepared statements. Never concatenate user "
            "input into SQL. Use an ORM where possible."
        ),
        references=[f"{_MITRE}/89.html", f"{_OWASP}/SQL_Injection"],
    ),
    "CWE-22": RemediationEntry(
        cwe_id="CWE-22",
        title="Path Traversal",
        guidance=(
            "Validate and canonicalize file paths. Use allowlists for permitted "
            "directories. Never pass user input directly to file system operations."
        ),
        references=[f"{_MITRE}/22.html"],
    ),
    "CWE-78": RemediationEntry(
        cwe_id="CWE-78",
        title="OS Command Injection",
        guidance=(
            "Avoid shell commands with user input. Use subprocess with list args "
            "(not shell=True). Validate input against strict allowlists."
        ),
        references=[f"{_MITRE}/78.html"],
    ),
    "CWE-352": RemediationEntry(
        cwe_id="CWE-352",
        title="Cross-Site Request Forgery (CSRF)",
        guidance=(
            "Implement anti-CSRF tokens on all state-changing forms. Use SameSite "
            "cookie attributes. Verify Origin/Referer headers."
        ),
        references=[f"{_MITRE}/352.html"],
    ),
    "CWE-918": RemediationEntry(
        cwe_id="CWE-918",
        title="Server-Side Request Forgery (SSRF)",
        guidance=(
            "Validate and allowlist destination URLs. Block requests to internal/private "
            "IP ranges. Use a server-side proxy with URL validation."
        ),
        references=[f"{_MITRE}/918.html"],
    ),
    "CWE-502": RemediationEntry(
        cwe_id="CWE-502",
        title="Deserialization of Untrusted Data",
        guidance=(
            "Never deserialize untrusted data. Use safe serialization formats (JSON). "
            "Implement integrity checks on serialized objects."
        ),
        references=[f"{_MITRE}/502.html"],
    ),
    "CWE-611": RemediationEntry(
        cwe_id="CWE-611",
        title="XML External Entity (XXE)",
        guidance=(
            "Disable external entity processing in XML parsers. Use defusedxml library. "
            "Validate XML input against schema."
        ),
        references=[f"{_MITRE}/611.html"],
    ),
    "CWE-416": RemediationEntry(
        cwe_id="CWE-416",
        title="Use After Free",
        guidance=(
            "Null out pointers after free. Use smart pointers / RAII. Enable "
            "address sanitizers (ASan) during testing."
        ),
        references=[f"{_MITRE}/416.html"],
    ),
    "CWE-798": RemediationEntry(
        cwe_id="CWE-798",
        title="Use of Hard-coded Credentials",
        guidance=(
            "Remove credentials from source code. Use a secrets manager or "
            "environment variables. Rotate any exposed secrets."
        ),
        references=[f"{_MITRE}/798.html"],
    ),
    "CWE-434": RemediationEntry(
        cwe_id="CWE-434",
        title="Unrestricted Upload of File with Dangerous Type",
        guidance=(
            "Validate file type, size, and content on upload. Store uploads outside "
            "the web root. Use randomized filenames."
        ),
        references=[f"{_MITRE}/434.html"],
    ),
    "CWE-601": RemediationEntry(
        cwe_id="CWE-601",
        title="URL Redirection to Untrusted Site (Open Redirect)",
        guidance=(
            "Validate redirect targets against an allowlist. Avoid user-controlled "
            "redirect URLs. Use relative paths when possible."
        ),
        references=[f"{_MITRE}/601.html"],
    ),
    "CWE-94": RemediationEntry(
        cwe_id="CWE-94",
        title="Code Injection",
        guidance=(
            "Never evaluate user-supplied code. Use sandboxing if code execution is "
            "required. Validate input against strict allowlists."
        ),
        references=[f"{_MITRE}/94.html"],
    ),
    "CWE-306": RemediationEntry(
        cwe_id="CWE-306",
        title="Missing Authentication for Critical Function",
        guidance=(
            "Require authentication on all sensitive endpoints. Default-deny all "
            "access and explicitly grant it. Audit routes for missing auth."
        ),
        references=[f"{_MITRE}/306.html"],
    ),
    "CWE-307": RemediationEntry(
        cwe_id="CWE-307",
        title="Improper Restriction of Excessive Authentication Attempts",
        guidance=(
            "Implement rate limiting and account lockout for authentication. "
            "Use CAPTCHAs or exponential backoff on repeated failures."
        ),
        references=[f"{_MITRE}/307.html"],
    ),
    "CWE-287": RemediationEntry(
        cwe_id="CWE-287",
        title="Improper Authentication",
        guidance=(
            "Implement multi-factor authentication. Use proven auth libraries. "
            "Enforce strong password policies with rate limiting."
        ),
        references=[f"{_MITRE}/287.html"],
    ),
    "CWE-862": RemediationEntry(
        cwe_id="CWE-862",
        title="Missing Authorization",
        guidance=(
            "Implement authorization checks on every endpoint. Use role-based access "
            "control. Validate object-level permissions."
        ),
        references=[f"{_MITRE}/862.html"],
    ),
}


class CWERemediator:
    """Looks up and formats CWE remediation guidance."""

    def __init__(self, mapping: dict[str, RemediationEntry] | None = None) -> None:
        self._map: dict[str, RemediationEntry] = dict(mapping or CWE_REMEDIATION_MAP)

    @staticmethod
    def _normalize(cwe_id: str) -> str:
        norm = (cwe_id or "").upper().strip()
        if not norm:
            return ""
        if not norm.startswith("CWE-"):
            norm = f"CWE-{norm}"
        return norm

    def lookup(self, cwe_id: str) -> RemediationEntry | None:
        """Return the remediation entry for the CWE id, if known."""
        return self._map.get(self._normalize(cwe_id))

    def lookup_all(self, cwe_ids: list[str]) -> list[RemediationEntry]:
        """Return remediation entries for all known CWEs in the list."""
        out: list[RemediationEntry] = []
        for cid in cwe_ids or []:
            entry = self.lookup(cid)
            if entry is not None:
                out.append(entry)
        return out

    def format_guidance(self, cwe_id: str, context: str = "") -> str:
        """Render remediation guidance, safely interpolating user context."""
        entry = self.lookup(cwe_id)
        safe_context = _sanitize_for_prompt(context)
        if entry is None:
            base = (
                f"No specific remediation available for {self._normalize(cwe_id) or 'this CWE'}. "
                "Review the finding details and apply appropriate security controls."
            )
            if safe_context:
                return f"{base}\nContext: {safe_context}"
            return base
        body = f"{entry.cwe_id} — {entry.title}\n{entry.guidance}"
        if safe_context:
            body += f"\nContext: {safe_context}"
        if entry.references:
            body += "\nReferences: " + ", ".join(entry.references)
        return body

    def known_cwes(self) -> list[str]:
        """Sorted list of CWE IDs supported by this remediator."""
        return sorted(self._map.keys())

    def search(self, query: str) -> list[RemediationEntry]:
        """Case-insensitive substring search across title + guidance."""
        q = (query or "").lower().strip()
        if not q:
            return []
        hits: list[RemediationEntry] = []
        for cid in sorted(self._map.keys()):
            entry = self._map[cid]
            haystack = f"{entry.title}\n{entry.guidance}".lower()
            if q in haystack:
                hits.append(entry)
        return hits


CWE_REMEDIATION_REGISTRY: dict[str, type[CWERemediator]] = {"default": CWERemediator}
