"""Intent-level code reviewer (s0cli-inspired).

Traditional SAST flags surface-level patterns; this reviewer is wired for
*intent-level* vulnerabilities (SSRF, IDOR, mass-assignment, crypto
mistakes, business-logic bugs) that typically require LLM judgment.

Aurelius stays judge-agnostic: callers supply ``judge_fn(code, file_path)``
which must return a list of Finding-shaped dicts / dataclasses with the
required fields enforced in ``filter_low_quality``. The reviewer never
executes or imports the code under review — it is pattern-level only.

Pure stdlib: ``re``, ``dataclasses``, ``time``. No foreign imports.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

VULNHUNTER_SYSTEM_PROMPT: str = """\
You are VulnHunter, an intent-level code reviewer. Your job is to surface
real, exploitable vulnerabilities — not style issues, not lint, not
theoretical what-ifs.

You look specifically for intent-level issues that surface-level SAST
tools miss:
  - SSRF (server-side request forgery) where attacker-controlled URL
    feeds an outbound HTTP client
  - IDOR (insecure direct object reference) where an authenticated
    request reads or mutates another tenant's row
  - Mass-assignment where a model's field list is populated from a raw
    request body
  - Cryptographic mistakes: ECB mode, static IVs, weak KDFs, weak RNG
    used for secrets, MD5/SHA1 for auth, hard-coded keys
  - Business-logic bugs: race conditions on balances, negative-amount
    transfers, coupon replays, privilege escalation via role strings
  - Command-injection and deserialization sinks
  - Path traversal where user input joins a base directory
  - XXE / SSRF via XML / URL parsers
  - Authentication bypasses and missing authorization checks
  - Time-of-check / time-of-use (TOCTOU) flaws

SKEPTICISM RULE (mandatory, non-negotiable):
Don't flag just because it touches user input; you must articulate attacker identity + controlled input + sink line + one-line fix before emitting a finding.
If you cannot name all four, stay silent.

For every finding you emit, produce the following fields:
  - why_real: a single sentence naming (a) the attacker, (b) the input
    they control, and (c) the sink line that turns it into impact
  - fix_hint: ONE line describing the concrete fix
  - cwe: the most specific CWE identifier (e.g. "CWE-918" for SSRF)
  - severity: one of {critical, high, medium, low, info}
  - line: best-effort 1-indexed line number of the sink

NEGATIVE LIST — Do NOT flag:
  - style issues, naming conventions, import order, line length
  - test fixtures with test creds (e.g. password="test" inside tests/)
  - env-var secrets like os.environ['API_KEY'] (the secret is external)
  - commented-out code
  - TODO / FIXME comments alone
  - dead code that has no reachable sink
  - theoretical issues with no attacker-controlled input

Output must be machine-parseable JSON. Silence is better than a noisy
finding. The harness will drop any finding missing why_real / fix_hint /
cwe — so make each field count.
"""  # noqa: E501


NEGATIVE_EXAMPLES: tuple[str, ...] = (
    "# test fixture: pw='test'",
    "os.environ['API_KEY']",
    "os.getenv('SECRET_TOKEN')",
    "# TODO: refactor later",
    "password = 'test'  # pytest fixture",
    "API_KEY = os.environ.get('API_KEY', '')",
)


REQUIRED_FIELDS: tuple[str, ...] = ("why_real", "fix_hint", "cwe")

_VALID_SEVERITIES: tuple[str, ...] = (
    "critical",
    "high",
    "medium",
    "low",
    "info",
)

_SEVERITY_RANK: dict[str, int] = {
    "info": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


@dataclass
class VibeFinding:
    why_real: str
    fix_hint: str
    cwe: str
    severity: str
    line: int
    snippet: str = ""
    file_path: str = ""


@dataclass
class VibeReviewReport:
    file_path: str
    findings: list[VibeFinding]
    total_lines: int
    elapsed_s: float
    dropped: int = 0
    warning: str | None = None


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_snippet(lines: list[str], line_no: int, radius: int = 2) -> str:
    if not lines or line_no < 1:
        return ""
    idx = min(max(line_no, 1), len(lines)) - 1
    lo = max(0, idx - radius)
    hi = min(len(lines), idx + radius + 1)
    return "\n".join(lines[lo:hi])


class VibeCodeReviewer:
    """Intent-level reviewer that delegates to a caller-supplied judge.

    The reviewer never executes or imports reviewed code — it only passes
    the raw source and path to ``judge_fn`` and post-filters the returned
    findings.
    """

    def __init__(
        self,
        judge_fn: Callable[[str, str], list[Any]],
        min_severity: str = "medium",
    ) -> None:
        if min_severity not in _SEVERITY_RANK:
            raise ValueError(
                f"min_severity must be one of {list(_SEVERITY_RANK)}, got {min_severity!r}"
            )
        self.judge_fn = judge_fn
        self.min_severity = min_severity

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------
    def filter_low_quality(self, findings: list[Any]) -> tuple[list[Any], int]:
        """Drop findings missing why_real / fix_hint / cwe.

        Loud-fail by dropping: silent heuristics would let low-quality
        reports through. Returns ``(kept, dropped_count)``.
        """
        kept: list[Any] = []
        dropped = 0
        for f in findings:
            ok = True
            for field_name in REQUIRED_FIELDS:
                val = _get(f, field_name, "")
                if not isinstance(val, str) or not val.strip():
                    ok = False
                    break
            if ok:
                kept.append(f)
            else:
                dropped += 1
        return kept, dropped

    # ------------------------------------------------------------------
    # Severity filter
    # ------------------------------------------------------------------
    def _meets_severity(self, severity: str) -> bool:
        rank = _SEVERITY_RANK.get(severity, -1)
        return rank >= _SEVERITY_RANK[self.min_severity]

    # ------------------------------------------------------------------
    # Review one file
    # ------------------------------------------------------------------
    def review_file(self, path: str, code: str | None = None) -> VibeReviewReport:
        if code is None:
            with open(path, "rb") as fh:
                raw = fh.read()
            code = raw.decode("utf-8", errors="replace")

        lines = code.splitlines()
        total_lines = len(lines)
        warning: str | None = None

        t0 = time.perf_counter()
        if not code.strip():
            raw_findings: list[Any] = []
        else:
            try:
                raw_findings = list(self.judge_fn(code, path) or [])
            except Exception as exc:  # noqa: BLE001
                warning = f"judge_fn raised {type(exc).__name__}: {exc}"
                raw_findings = []

        kept, dropped = self.filter_low_quality(raw_findings)

        vibe: list[VibeFinding] = []
        for f in kept:
            severity = str(_get(f, "severity", "info") or "info").lower()
            if severity not in _VALID_SEVERITIES:
                severity = "info"
            if not self._meets_severity(severity):
                continue
            try:
                line_no = int(_get(f, "line", 1) or 1)
            except (TypeError, ValueError):
                line_no = 1
            if line_no < 1:
                line_no = 1
            snippet = _get(f, "snippet", "") or _extract_snippet(lines, line_no)
            vibe.append(
                VibeFinding(
                    why_real=str(_get(f, "why_real", "")),
                    fix_hint=str(_get(f, "fix_hint", "")),
                    cwe=str(_get(f, "cwe", "")),
                    severity=severity,
                    line=line_no,
                    snippet=snippet,
                    file_path=path,
                )
            )

        elapsed_s = max(0.0, time.perf_counter() - t0)

        return VibeReviewReport(
            file_path=path,
            findings=vibe,
            total_lines=total_lines,
            elapsed_s=elapsed_s,
            dropped=dropped,
            warning=warning,
        )

    # ------------------------------------------------------------------
    # Review many files
    # ------------------------------------------------------------------
    def review_corpus(self, paths: list[str]) -> dict[str, VibeReviewReport]:
        return {p: self.review_file(p) for p in paths}


# ----------------------------------------------------------------------
# Deterministic stub judge for tests
# ----------------------------------------------------------------------
# Marker -> (cwe, severity, why_real, fix_hint)
_STUB_MARKERS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        r"VIBE_SSRF_SINK",
        "CWE-918",
        "high",
        "Unauthenticated remote attacker controls the URL passed into "
        "the outbound HTTP client on this line, turning the request into "
        "an SSRF pivot against internal services.",
        "Validate the URL against an allowlist of hosts before the outbound request.",
    ),
    (
        r"VIBE_IDOR_SINK",
        "CWE-639",
        "high",
        "An authenticated tenant supplies the object id in the request "
        "path and the handler loads the row without any ownership check, "
        "allowing cross-tenant reads.",
        "Scope the query by current_user.tenant_id before returning the row.",
    ),
    (
        r"VIBE_CRYPTO_SINK",
        "CWE-327",
        "medium",
        "Any caller who can submit ciphertext can rely on ECB mode on "
        "this line to leak plaintext structure, enabling chosen-ciphertext "
        "distinguishers.",
        "Use AES-GCM with a random 96-bit nonce instead of ECB.",
    ),
    (
        r"VIBE_LOW_QUALITY",
        "CWE-000",
        "medium",
        "",  # intentionally missing why_real → must be dropped
        "",  # intentionally missing fix_hint → must be dropped
    ),
    (
        r"VIBE_LOW_SEVERITY",
        "CWE-710",
        "low",
        "A local user can cause a noisy log line here but no "
        "confidentiality or integrity impact results.",
        "Downgrade the log level to debug.",
    ),
)


def stub_judge_fn(code: str, file_path: str) -> list[dict[str, Any]]:
    """Deterministic regex-based judge used by tests.

    Scans ``code`` for marker tokens; each match yields a Finding-shaped
    dict. Deterministic: same input → same output, every call.
    """
    findings: list[dict[str, Any]] = []
    lines = code.splitlines()
    for lineno, line in enumerate(lines, start=1):
        for marker, cwe, severity, why_real, fix_hint in _STUB_MARKERS:
            if re.search(marker, line):
                findings.append(
                    {
                        "why_real": why_real,
                        "fix_hint": fix_hint,
                        "cwe": cwe,
                        "severity": severity,
                        "line": lineno,
                    }
                )
    return findings


__all__ = [
    "VULNHUNTER_SYSTEM_PROMPT",
    "NEGATIVE_EXAMPLES",
    "REQUIRED_FIELDS",
    "VibeFinding",
    "VibeReviewReport",
    "VibeCodeReviewer",
    "stub_judge_fn",
]
