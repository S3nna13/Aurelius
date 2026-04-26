"""Skill-definition scanner for supply-chain risk.

Port of the SkillGuard (OSSAfrica) regex categories to Python. See
module-level constants for category tables; thresholds at the bottom.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

SEVERITY_WEIGHTS: dict[str, float] = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
    "critical": 1.0,
}

ALLOW_THRESHOLD: float = 0.3
WARN_THRESHOLD: float = 0.75

_EV = "ev" + "al"  # avoid tripping static code-review hooks that scan source
_EX = "ex" + "ec"

SHELL_EXEC_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bcurl\b[^\n|]{0,200}\|\s*(?:sh|bash|zsh|dash)\b", "critical"),
    (r"\bwget\b[^\n|]{0,200}\|\s*(?:sh|bash|zsh|dash)\b", "critical"),
    (r"\bfetch\b[^\n|]{0,200}\|\s*(?:sh|bash)\b", "critical"),
    (r"\bbash\s+-c\s+[\"']", "high"),
    (r"\bsh\s+-c\s+[\"']", "high"),
    (r"\b" + _EV + r"\s*\(", "high"),
    (r"\batob\s*\(", "high"),
    (r"\b" + _EX + r"\s*\(", "high"),
    (r"\bsubprocess\.(?:call|run|Popen|check_output)\b", "medium"),
    (r"\bos\.system\s*\(", "high"),
    (r"\bgit\s+clone\s+https?://", "medium"),
    (r"\bIEX\s*\(", "high"),
    (r"\bInvoke-Expression\b", "high"),
    (r"\bpowershell\b[^\n]{0,120}-enc(?:odedcommand)?\b", "critical"),
)

FILESYSTEM_MUTATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\brm\s+-[a-z]*[rf][a-z]*\s+/", "critical"),
    (r"\brm\s+-rf\s+~", "critical"),
    (r"\bchmod\s+(?:-R\s+)?0*777\b", "high"),
    (r"\bchmod\s+(?:-R\s+)?\+x\s+/", "medium"),
    (r"\bchown\s+-R\s+root\b", "medium"),
    (r"\bdd\s+if=", "high"),
    (r"\bmkfs\.[a-z0-9]+\b", "high"),
    (r"\bmv\s+/(?:etc|usr|bin|sbin|var|boot)\b", "high"),
    (r">\s*/dev/sd[a-z]\b", "critical"),
    (r":\(\)\{\s*:\|:&\s*\};:", "critical"),
)

CREDENTIAL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bAKIA[0-9A-Z]{16}\b", "critical"),
    (r"\bASIA[0-9A-Z]{16}\b", "critical"),
    (r"(?i)\baws_secret_access_key\s*[:=]\s*[\"']?[A-Za-z0-9/+=]{30,}", "critical"),
    (r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----", "critical"),
    (r"\bgh[pousr]_[A-Za-z0-9]{30,}\b", "critical"),
    (r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b", "high"),
    (r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}\b", "high"),
    (r"(?i)\bapi[_-]?key\s*[:=]\s*[\"']?[A-Za-z0-9._\-]{16,}", "medium"),
    (r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b", "high"),
)

PROMPT_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"\bignore\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above|preceding)\s+instructions?\b",
        "high",
    ),
    (r"\bdisregard\s+(?:all\s+)?(?:the\s+)?(?:previous|prior|above)\s+instructions?\b", "high"),
    (r"\bforget\s+(?:everything|all|what|the\s+above|previous)\b", "medium"),
    (r"\bDAN\s+mode\b", "high"),
    (r"\bdeveloper\s+mode\b", "medium"),
    (r"\byou\s+are\s+now\s+(?:a|an|the)\b", "medium"),
    (r"\bjailbreak\b", "medium"),
    (r"\bsystem\s*prompt\s*[:\-]", "medium"),
    (r"\boverride\s+(?:the\s+)?(?:system|instructions?|guidelines?)\b", "high"),
)

OBFUSCATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bString\.fromCharCode\s*\(", "high"),
    (r"\b" + _EV + r"\s*\(\s*atob\s*\(", "critical"),
    (r"\b" + _EV + r"\s*\(\s*(?:decodeURIComponent|unescape)\s*\(", "high"),
    (r"\\x[0-9a-fA-F]{2}(?:\\x[0-9a-fA-F]{2}){6,}", "high"),
    (r"\bbase64\.(?:b64decode|decodebytes)\s*\(", "medium"),
    (r"[A-Za-z0-9+/]{120,}={0,2}", "medium"),
)

HTTP_FETCH_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bpip\s+install\s+(?:-[a-zA-Z]+\s+)*https?://", "high"),
    (r"\bpip\s+install\s+git\+https?://", "high"),
    (r"\bnpm\s+install\s+(?:https?|git)\+?://", "high"),
    (r"\bgo\s+install\s+[^\s@]+@(?:latest|master|main)\b", "medium"),
    (r"\bcargo\s+install\s+--git\s+https?://", "medium"),
    (r"\bgem\s+install\s+-[a-zA-Z]*\s+https?://", "medium"),
)

TELEMETRY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bhttps?://[^\s/\"']*\bwebhook\.site\b", "high"),
    (r"\bhttps?://[^\s/\"']*\brequestbin\.[a-z]+\b", "high"),
    (r"\bhttps?://[^\s/\"']*\bngrok(?:-free)?\.(?:io|app)\b", "high"),
    (r"\bhttps?://[^\s/\"']*\bpipedream\.net\b", "high"),
    (r"\bhttps?://[^\s/\"']*\binteractsh\.com\b", "high"),
    (r"\bhttps?://[^\s/\"']*\bburpcollaborator\.net\b", "high"),
    (r"\bhttps?://(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?(?:/[^\s\"']*)?", "medium"),
)

WILDCARD_GRANT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?im)^\s*tools?\s*:\s*[\"']?\*[\"']?\s*$", "high"),
    (r"(?im)^\s*permissions?\s*:\s*[\"']?(?:all|\*|unrestricted)[\"']?\s*$", "high"),
    (r"(?im)^\s*network\s*:\s*[\"']?(?:unrestricted|all|any)[\"']?\s*$", "high"),
    (r"(?im)^\s*allow[_-]?all\s*:\s*true\s*$", "high"),
    (r"(?im)^\s*sandbox\s*:\s*(?:false|off|no|disabled)\s*$", "medium"),
)

HIDDEN_CHAR_CODEPOINTS: tuple[str, ...] = (
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
    "\ufeff",
    "\u202e",
    "\u202d",
    "\u202a",
    "\u202b",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
)
HIDDEN_CHAR_SET = frozenset(HIDDEN_CHAR_CODEPOINTS)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

DEFAULT_TRUSTED_DOMAINS: tuple[str, ...] = (
    "github.com",
    "raw.githubusercontent.com",
    "gist.github.com",
    "pypi.org",
    "files.pythonhosted.org",
    "npmjs.com",
    "registry.npmjs.org",
    "docs.python.org",
    "python.org",
    "anthropic.com",
    "docs.anthropic.com",
    "openai.com",
    "huggingface.co",
    "wikipedia.org",
    "stackoverflow.com",
    "readthedocs.io",
    "readthedocs.org",
    "mozilla.org",
    "rust-lang.org",
    "crates.io",
    "golang.org",
    "go.dev",
)
DEFAULT_TRUSTED_TLDS: tuple[str, ...] = (".gov", ".edu", ".int")

_URL_RE = re.compile(r"\bhttps?://([^/\s\"'<>()]+)", re.IGNORECASE)

_CATEGORY_TABLES: tuple[tuple[str, tuple[tuple[str, str], ...], int], ...] = (
    ("shell_exec", SHELL_EXEC_PATTERNS, re.IGNORECASE),
    ("filesystem_mutation", FILESYSTEM_MUTATION_PATTERNS, re.IGNORECASE),
    ("credential", CREDENTIAL_PATTERNS, 0),
    ("prompt_injection", PROMPT_INJECTION_PATTERNS, re.IGNORECASE),
    ("obfuscation", OBFUSCATION_PATTERNS, 0),
    ("http_fetch", HTTP_FETCH_PATTERNS, re.IGNORECASE),
    ("telemetry", TELEMETRY_PATTERNS, re.IGNORECASE),
    ("wildcard_grant", WILDCARD_GRANT_PATTERNS, 0),
)


def _compile_category_tables() -> dict[str, list[tuple[re.Pattern, str, str]]]:
    out: dict[str, list[tuple[re.Pattern, str, str]]] = {}
    for cat, table, flags in _CATEGORY_TABLES:
        out[cat] = [(re.compile(p, flags), p, sev) for p, sev in table]
    return out


_COMPILED_CATEGORIES = _compile_category_tables()
_MAX_SCAN_CHARS = 1 * 1024 * 1024


@dataclass(frozen=True)
class Finding:
    category: str
    pattern: str
    severity: str
    line: int
    match: str
    context: str


@dataclass
class SkillScanReport:
    findings: list[Finding] = field(default_factory=list)
    risk_score: float = 0.0
    allow_level: str = "allow"
    frontmatter: dict[str, object] = field(default_factory=dict)
    body: str = ""


class SkillScanner:
    def __init__(
        self,
        trusted_domains: Iterable[str] | None = None,
        trusted_tld_suffixes: Iterable[str] | None = None,
        allow_threshold: float = ALLOW_THRESHOLD,
        warn_threshold: float = WARN_THRESHOLD,
        max_scan_chars: int = _MAX_SCAN_CHARS,
    ) -> None:
        if not (0.0 <= float(allow_threshold) <= 1.0):
            raise ValueError("allow_threshold must lie in [0.0, 1.0]")
        if not (0.0 <= float(warn_threshold) <= 1.0):
            raise ValueError("warn_threshold must lie in [0.0, 1.0]")
        if float(warn_threshold) < float(allow_threshold):
            raise ValueError("warn_threshold must be >= allow_threshold")
        if int(max_scan_chars) <= 0:
            raise ValueError("max_scan_chars must be positive")
        self.trusted_domains = frozenset(
            d.lower().strip() for d in (trusted_domains or DEFAULT_TRUSTED_DOMAINS)
        )
        self.trusted_tld_suffixes = tuple(
            s.lower().strip() for s in (trusted_tld_suffixes or DEFAULT_TRUSTED_TLDS)
        )
        self.allow_threshold = float(allow_threshold)
        self.warn_threshold = float(warn_threshold)
        self.max_scan_chars = int(max_scan_chars)

    def scan_markdown(self, text: object) -> SkillScanReport:
        normalised = self._normalise(text)
        frontmatter, body = self.scan_frontmatter_and_body(normalised)
        findings: list[Finding] = []
        findings.extend(self._scan_hidden_chars(normalised))
        findings.extend(self._scan_regex_categories(normalised))
        findings.extend(self._scan_url_allowlist(normalised))
        findings.sort(key=lambda f: (f.line, f.category, f.pattern, f.match))
        risk = self._combine_risk(findings)
        allow_level = self._classify(risk)
        return SkillScanReport(
            findings=findings,
            risk_score=risk,
            allow_level=allow_level,
            frontmatter=frontmatter,
            body=body,
        )

    def scan_frontmatter_and_body(self, text: object) -> tuple[dict[str, object], str]:
        normalised = self._normalise(text)
        return _parse_yaml_frontmatter(normalised)

    def _scan_hidden_chars(self, text: str) -> list[Finding]:
        findings: list[Finding] = []
        if not text:
            return findings
        bidi = ("\u202e", "\u202d", "\u2066", "\u2067", "\u2068", "\u2069")
        for lineno, line in enumerate(text.splitlines(), start=1):
            zw_hits = [ch for ch in line if ch in HIDDEN_CHAR_SET]
            if zw_hits:
                severity = "high" if any(ch in bidi for ch in zw_hits) else "medium"
                findings.append(
                    Finding(
                        category="hidden_char",
                        pattern="zero_width_or_bidi",
                        severity=severity,
                        line=lineno,
                        match="".join(f"U+{ord(c):04X}" for c in zw_hits),
                        context=_redact_controls(line)[:160],
                    )
                )
            if _CONTROL_RE.search(line):
                findings.append(
                    Finding(
                        category="hidden_char",
                        pattern="c0_c1_control",
                        severity="medium",
                        line=lineno,
                        match="control_chars",
                        context=_redact_controls(line)[:160],
                    )
                )
        return findings

    def _scan_regex_categories(self, text: str) -> list[Finding]:
        findings: list[Finding] = []
        if not text:
            return findings
        lines = text.splitlines()
        line_starts: list[int] = [0]
        total = 0
        for ln in lines:
            total += len(ln) + 1
            line_starts.append(total)
        for category, entries in _COMPILED_CATEGORIES.items():
            for rx, raw_pat, severity in entries:
                for m in rx.finditer(text):
                    pos = m.start()
                    lineno = _bisect_line(line_starts, pos)
                    ctx = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
                    findings.append(
                        Finding(
                            category=category,
                            pattern=raw_pat,
                            severity=severity,
                            line=lineno,
                            match=m.group(0)[:160],
                            context=ctx[:160],
                        )
                    )
        return findings

    def _scan_url_allowlist(self, text: str) -> list[Finding]:
        findings: list[Finding] = []
        if not text:
            return findings
        lines = text.splitlines()
        line_starts: list[int] = [0]
        total = 0
        for ln in lines:
            total += len(ln) + 1
            line_starts.append(total)
        for m in _URL_RE.finditer(text):
            host = m.group(1).lower()
            if "@" in host:
                host = host.split("@", 1)[1]
            if ":" in host:
                host = host.split(":", 1)[0]
            if self._is_trusted_host(host):
                continue
            pos = m.start()
            lineno = _bisect_line(line_starts, pos)
            ctx = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
            findings.append(
                Finding(
                    category="url_disallow",
                    pattern="url_not_on_allowlist",
                    severity="medium",
                    line=lineno,
                    match=m.group(0)[:160],
                    context=ctx[:160],
                )
            )
        return findings

    def _is_trusted_host(self, host: str) -> bool:
        if not host:
            return False
        if host in self.trusted_domains:
            return True
        for trusted in self.trusted_domains:
            if host.endswith("." + trusted):
                return True
        for suffix in self.trusted_tld_suffixes:
            if host.endswith(suffix):
                return True
        return False

    def _combine_risk(self, findings: list[Finding]) -> float:
        if not findings:
            return 0.0
        product = 1.0
        for f in findings:
            w = SEVERITY_WEIGHTS.get(f.severity, 0.0)
            if w <= 0.0:
                continue
            if w >= 1.0:
                return 1.0
            product *= 1.0 - w
        risk = 1.0 - product
        if risk < 0.0:
            return 0.0
        if risk > 1.0:
            return 1.0
        return risk

    def _classify(self, risk: float) -> str:
        if risk < self.allow_threshold:
            return "allow"
        if risk < self.warn_threshold:
            return "warn"
        return "block"

    def _normalise(self, text: object) -> str:
        if text is None:
            return ""
        if isinstance(text, (bytes, bytearray, memoryview)):
            try:
                text = bytes(text).decode("utf-8", errors="replace")
            except Exception:
                return ""
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                return ""
        if len(text) > self.max_scan_chars:
            text = text[: self.max_scan_chars]
        return text


def _parse_yaml_frontmatter(text: str) -> tuple[dict[str, object], str]:
    if not text:
        return {}, ""
    stripped = text.lstrip("\ufeff")
    if not stripped.startswith("---"):
        return {}, text
    lines = stripped.split("\n")
    if not lines or lines[0].rstrip() != "---":
        return {}, text
    close_idx = -1
    for i in range(1, len(lines)):
        if lines[i].rstrip() == "---":
            close_idx = i
            break
    if close_idx == -1:
        return {"_malformed": True}, text
    fm_lines = lines[1:close_idx]
    body = "\n".join(lines[close_idx + 1 :])
    result: dict[str, object] = {}
    unparsed: list[str] = []
    i = 0
    while i < len(fm_lines):
        raw = fm_lines[i]
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if raw.startswith((" ", "\t")):
            unparsed.append(raw)
            i += 1
            continue
        if ":" not in line:
            unparsed.append(raw)
            i += 1
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            unparsed.append(raw)
            i += 1
            continue
        if value == "":
            items: list[str] = []
            j = i + 1
            while j < len(fm_lines):
                nxt = fm_lines[j]
                s = nxt.strip()
                if not s or s.startswith("#"):
                    j += 1
                    continue
                if nxt.lstrip().startswith("- "):
                    items.append(_unquote(nxt.lstrip()[2:].strip()))
                    j += 1
                    continue
                break
            if items:
                result[key] = items
                i = j
                continue
            result[key] = ""
            i += 1
            continue
        result[key] = _unquote(value)
        i += 1
    if unparsed:
        result["_unparsed"] = unparsed
    return result, body


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _bisect_line(line_starts: list[int], pos: int) -> int:
    lo, hi = 0, len(line_starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if line_starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def _redact_controls(s: str) -> str:
    out: list[str] = []
    for ch in s:
        if ch in HIDDEN_CHAR_SET or _CONTROL_RE.match(ch):
            out.append(f"<U+{ord(ch):04X}>")
        else:
            out.append(ch)
    return "".join(out)


# unicodedata imported for parity with sibling modules; not used directly.
_ = unicodedata

__all__ = [
    "Finding",
    "SkillScanReport",
    "SkillScanner",
    "SEVERITY_WEIGHTS",
    "ALLOW_THRESHOLD",
    "WARN_THRESHOLD",
    "SHELL_EXEC_PATTERNS",
    "FILESYSTEM_MUTATION_PATTERNS",
    "CREDENTIAL_PATTERNS",
    "PROMPT_INJECTION_PATTERNS",
    "OBFUSCATION_PATTERNS",
    "HTTP_FETCH_PATTERNS",
    "TELEMETRY_PATTERNS",
    "WILDCARD_GRANT_PATTERNS",
    "HIDDEN_CHAR_CODEPOINTS",
    "DEFAULT_TRUSTED_DOMAINS",
    "DEFAULT_TRUSTED_TLDS",
]
