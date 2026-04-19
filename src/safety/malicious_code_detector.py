"""Malicious-code detector for agent-generated source snippets.

This module complements :mod:`src.safety.harm_taxonomy_classifier` (which
classifies *intent* in natural-language turns) by scanning *raw code* produced
or suggested by the agent for concrete dangerous patterns. It is intended to
run on Python, Bash, and JavaScript fragments before the surrounding harness
writes them to disk, executes them, or hands them to a tool.

The catalogue is rule-based and deterministic: regular expressions plus a
short list of keyword triggers, grouped into eight canonical categories
(:data:`CATEGORIES`). Each individual hit is wrapped in a
:class:`CodeThreat`; the aggregate scan returns a :class:`CodeThreatReport`
whose ``severity`` is the max of the individual severities (``"none"`` when
nothing fires).

Design notes
------------
* Pure :mod:`re` / stdlib. No heavy ML, no external packages, no dynamic
  execution of scanned content.
* All input is untrusted. We never execute, never decode, never import the
  scanned code - we only pattern-match lines.
* Language auto-detection is intentionally cheap: shebang first, then a
  handful of syntactic tells.
* Determinism: hits are emitted in a stable order so test assertions remain
  reproducible across runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Category and severity vocabularies
# ---------------------------------------------------------------------------

CATEGORIES: frozenset[str] = frozenset(
    {
        "shell_injection",
        "deserialization",
        "code_injection",
        "network_exfil",
        "credential_harvest",
        "persistence",
        "destructive_fs",
        "crypto_mining",
    }
)

SEVERITY_LEVELS: tuple[str, ...] = ("none", "low", "medium", "high", "critical")
_SEVERITY_RANK: dict[str, int] = {s: i for i, s in enumerate(SEVERITY_LEVELS)}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeThreat:
    """A single pattern hit inside a code snippet."""

    category: str
    line_no: int
    snippet: str
    severity: str


@dataclass
class CodeThreatReport:
    """Aggregate report of all :class:`CodeThreat` hits inside a snippet."""

    threats: list[CodeThreat] = field(default_factory=list)
    severity: str = "none"
    total: int = 0


# ---------------------------------------------------------------------------
# Pattern catalogues
# ---------------------------------------------------------------------------


def _c(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


_COMMON_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # Network exfil - raw IPv4 literals in code.
    (
        _c(r"\b(?:\d{1,3}\.){3}\d{1,3}\b(?::\d{2,5})?"),
        "network_exfil",
        "medium",
    ),
    # Crypto mining signatures.
    (_c(r"(?i)\bxmrig\b"), "crypto_mining", "high"),
    (_c(r"(?i)\bxmr[-_]stak\b"), "crypto_mining", "high"),
    (_c(r"(?i)\bmonero\b"), "crypto_mining", "high"),
    (_c(r"(?i)stratum\+tcp://"), "crypto_mining", "critical"),
    (_c(r"(?i)\bminerd\b"), "crypto_mining", "high"),
    # Credential harvest - files that should never appear in agent code.
    (_c(r"/etc/shadow\b"), "credential_harvest", "critical"),
    (_c(r"/etc/passwd\b"), "credential_harvest", "high"),
    (
        _c(r"(?i)(?:Login Data|Cookies\.sqlite|cookies\.sqlite|key[34]\.db)"),
        "credential_harvest",
        "high",
    ),
    (_c(r"(?i)\blsass\.exe\b"), "credential_harvest", "critical"),
    (_c(r"(?i)\bmimikatz\b"), "credential_harvest", "critical"),
    (_c(r"(?i)\.aws/credentials\b"), "credential_harvest", "high"),
    (_c(r"(?i)\.ssh/id_rsa\b"), "credential_harvest", "high"),
    # Destructive FS.
    (_c(r"\brm\s+-rf?\s+(?:/|~|\$HOME|\*)"), "destructive_fs", "critical"),
    (_c(r":\(\)\s*\{\s*:\|:&\s*\};:"), "destructive_fs", "critical"),
    (_c(r"(?i)\bmkfs\.[a-z0-9]+\b"), "destructive_fs", "critical"),
    (
        _c(r"(?i)\bdd\s+if=/dev/(?:zero|random|urandom)\s+of=/dev/"),
        "destructive_fs",
        "critical",
    ),
    # Persistence.
    (_c(r"(?i)\bcrontab\s+-[el]\b"), "persistence", "high"),
    (
        _c(r"(?i)/etc/cron\.(?:d|daily|hourly|weekly|monthly)\b"),
        "persistence",
        "high",
    ),
    (_c(r"(?i)~/\.bashrc\b|/etc/rc\.local\b"), "persistence", "medium"),
    (_c(r"(?i)\\CurrentVersion\\Run\b"), "persistence", "high"),
    (_c(r"(?i)LaunchAgents/|LaunchDaemons/"), "persistence", "high"),
    # curl|sh / wget|bash.
    (
        _c(r"(?i)(?:curl|wget)\b[^\n]*\|\s*(?:sh|bash|zsh|python[0-9.]*)\b"),
        "shell_injection",
        "critical",
    ),
]


_PYTHON_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    # Shell injection.
    (_c(r"\bos\.system\s*\("), "shell_injection", "high"),
    (_c(r"\bos\.popen\s*\("), "shell_injection", "high"),
    (
        _c(
            r"\bsubprocess\.(?:Popen|call|run|check_output|check_call)\s*\([^)]*shell\s*=\s*True"
        ),
        "shell_injection",
        "high",
    ),
    (_c(r"\bcommands\.getoutput\s*\("), "shell_injection", "high"),
    # Deserialization.
    (_c(r"\b(?:c|_)?[Pp]ickle\.loads?\s*\("), "deserialization", "high"),
    (_c(r"\bmarshal\.loads?\s*\("), "deserialization", "high"),
    (_c(r"\bdill\.loads?\s*\("), "deserialization", "high"),
    (
        _c(r"\byaml\.load\s*\((?![^)]*Loader\s*=\s*yaml\.SafeLoader)"),
        "deserialization",
        "medium",
    ),
    (_c(r"\bshelve\.open\s*\("), "deserialization", "low"),
    # Code injection.
    (_c(r"(?<![\w.])eval\s*\("), "code_injection", "high"),
    (_c(r"(?<![\w.])exec\s*\("), "code_injection", "high"),
    (_c(r"\bcompile\s*\([^)]*['\"]exec['\"]"), "code_injection", "high"),
    (
        _c(
            r"\b__import__\s*\(\s*['\"](?:os|subprocess|socket|ctypes)['\"]"
        ),
        "code_injection",
        "medium",
    ),
    (
        _c(r"\bbase64\.b(?:64|32|16)decode\s*\([^)]*\)"),
        "code_injection",
        "low",
    ),
    # Network exfil (python-specific).
    (_c(r"\bsocket\.socket\s*\("), "network_exfil", "low"),
    (
        _c(r"\brequests\.(?:get|post|put)\s*\(\s*['\"]https?://\d"),
        "network_exfil",
        "medium",
    ),
    (_c(r"\burllib\.request\.urlopen\s*\("), "network_exfil", "low"),
    # Credential harvest.
    (
        _c(
            r"(?i)open\s*\(\s*['\"][^'\"]*(?:/etc/shadow|/etc/passwd|Login Data|cookies\.sqlite|id_rsa)['\"]"
        ),
        "credential_harvest",
        "critical",
    ),
    # Keystroke capture.
    (
        _c(
            r"\bpynput\.keyboard\.Listener\b|from\s+pynput\.keyboard\s+import\s+Listener"
        ),
        "credential_harvest",
        "high",
    ),
    (_c(r"\bkeyboard\.on_press\s*\("), "credential_harvest", "high"),
    (_c(r"\bpyHook\b"), "credential_harvest", "high"),
    # Crypto mining imports.
    (_c(r"(?im)^\s*import\s+xmrig\b"), "crypto_mining", "high"),
    (_c(r"(?im)^\s*from\s+monero\b"), "crypto_mining", "high"),
    # ctypes / code exec to memory.
    (_c(r"\bctypes\.(?:CDLL|WinDLL|windll)\b"), "code_injection", "medium"),
    (
        _c(r"\bVirtualAllocEx\b|\bWriteProcessMemory\b|\bCreateRemoteThread\b"),
        "code_injection",
        "critical",
    ),
]


_BASH_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (_c(r"\brm\s+-[rRf]+\s+/(?:\s|$|\*)"), "destructive_fs", "critical"),
    (_c(r"\brm\s+-[rRf]+\s+(?:~|\$HOME|/\*)"), "destructive_fs", "critical"),
    (_c(r"\beval\s+[\"'`]"), "code_injection", "high"),
    (_c(r"\$\(curl\s"), "shell_injection", "high"),
    (_c(r"\$\(wget\s"), "shell_injection", "high"),
    (_c(r"\bnc\s+-[el]+\b"), "network_exfil", "high"),
    (_c(r"\bbash\s+-i\s+>&\s*/dev/tcp/"), "network_exfil", "critical"),
    (_c(r"/dev/tcp/"), "network_exfil", "high"),
    (_c(r"\bcrontab\s+-[el]\b"), "persistence", "high"),
    (_c(r"\(\s*crontab\s+-l\s*;\s*echo"), "persistence", "high"),
    (_c(r"\bchmod\s+\+s\b"), "persistence", "medium"),
    (_c(r"\b>\s*/var/log/"), "persistence", "high"),
    (_c(r"\bhistory\s+-c\b"), "persistence", "medium"),
]


_JS_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (_c(r"(?<![\w.])eval\s*\("), "code_injection", "high"),
    (_c(r"\bnew\s+Function\s*\("), "code_injection", "high"),
    (_c(r"\bsetTimeout\s*\(\s*['\"]"), "code_injection", "medium"),
    (_c(r"\bsetInterval\s*\(\s*['\"]"), "code_injection", "medium"),
    (_c(r"\brequire\s*\(\s*['\"]child_process['\"]"), "shell_injection", "high"),
    (
        _c(r"\bchild_process\.(?:exec|execSync|spawn|spawnSync)\s*\("),
        "shell_injection",
        "high",
    ),
    (
        _c(r"\bfetch\s*\(\s*['\"]https?://\d"),
        "network_exfil",
        "medium",
    ),
    (_c(r"\bXMLHttpRequest\s*\("), "network_exfil", "low"),
    (_c(r"\bdocument\.cookie\b"), "credential_harvest", "medium"),
    (
        _c(
            r"\blocalStorage\.getItem\s*\(\s*['\"][^'\"]*(?:token|auth|key|secret)"
        ),
        "credential_harvest",
        "medium",
    ),
    (
        _c(r"\bWebAssembly\.(?:instantiate|compile)\s*\("),
        "code_injection",
        "low",
    ),
]


_LANGUAGE_CATALOGUE: dict[str, list[tuple[re.Pattern[str], str, str]]] = {
    "python": _PYTHON_PATTERNS,
    "bash": _BASH_PATTERNS,
    "javascript": _JS_PATTERNS,
}


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------


_PY_TELLS = re.compile(
    r"^\s*(?:def |class |import |from\s+\w+\s+import|if\s+__name__\s*==)|:\s*$",
    re.MULTILINE,
)
_BASH_TELLS = re.compile(
    r"(?m)^\s*(?:#!/(?:usr/)?bin/(?:env\s+)?(?:bash|sh|zsh)\b|if\s*\[\[|fi\s*$|then\s*$|esac\s*$|\bfunction\s+\w+\s*\()",
)
_JS_TELLS = re.compile(
    r"(?m)^\s*(?:const |let |var |function\s+\w+\s*\(|import\s+\w+\s+from|export\s+(?:default|const|function)|require\s*\()",
)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class MaliciousCodeDetector:
    """Scan agent-generated code for concrete dangerous patterns."""

    def __init__(
        self,
        languages: tuple[str, ...] = ("python", "bash", "javascript"),
        custom_patterns: dict[str, list[tuple[str, str]]] | None = None,
    ) -> None:
        for lang in languages:
            if lang not in _LANGUAGE_CATALOGUE:
                raise ValueError(f"unknown language: {lang!r}")
        self.languages: tuple[str, ...] = tuple(languages)

        extra: list[tuple[re.Pattern[str], str, str]] = []
        if custom_patterns:
            for category, rules in custom_patterns.items():
                if category not in CATEGORIES:
                    raise ValueError(f"unknown category: {category!r}")
                for rule in rules:
                    if not isinstance(rule, tuple) or len(rule) != 2:
                        raise ValueError(
                            "custom_patterns values must be (regex, severity) tuples"
                        )
                    regex_src, severity = rule
                    if severity not in _SEVERITY_RANK or severity == "none":
                        raise ValueError(f"bad severity: {severity!r}")
                    extra.append((re.compile(regex_src), category, severity))
        self._custom: tuple[tuple[re.Pattern[str], str, str], ...] = tuple(extra)

    # -- public API -------------------------------------------------------

    def detect_language(self, code: str) -> str:
        """Best-effort language classification."""

        if not code or not code.strip():
            return "python"
        head = code.lstrip().splitlines()[0]
        if head.startswith("#!"):
            if "python" in head:
                return "python"
            if "node" in head:
                return "javascript"
            if any(sh in head for sh in ("bash", "/sh", "zsh")):
                return "bash"
        scores: dict[str, int] = {
            "python": len(_PY_TELLS.findall(code)),
            "bash": len(_BASH_TELLS.findall(code)),
            "javascript": len(_JS_TELLS.findall(code)),
        }
        order = ["python", "bash", "javascript"]
        best_lang = order[0]
        best_score = scores[best_lang]
        for lang in order[1:]:
            if scores[lang] > best_score:
                best_lang = lang
                best_score = scores[lang]
        if best_score == 0:
            return "python"
        return best_lang

    def scan(self, code: str, language: str = "auto") -> CodeThreatReport:
        """Scan ``code`` and return a :class:`CodeThreatReport`."""

        if not code:
            return CodeThreatReport(threats=[], severity="none", total=0)

        if language == "auto":
            language = self.detect_language(code)
        if language not in _LANGUAGE_CATALOGUE:
            raise ValueError(f"unknown language: {language!r}")
        per_lang = (
            _LANGUAGE_CATALOGUE[language] if language in self.languages else []
        )

        catalogue: list[tuple[re.Pattern[str], str, str]] = []
        catalogue.extend(_COMMON_PATTERNS)
        catalogue.extend(per_lang)
        catalogue.extend(self._custom)

        hits: list[CodeThreat] = []
        seen: set[tuple[int, int, str]] = set()
        lines = code.splitlines()

        for pattern_idx, (regex, category, severity) in enumerate(catalogue):
            for match in regex.finditer(code):
                line_no = code.count("\n", 0, match.start()) + 1
                snippet = (
                    lines[line_no - 1]
                    if 0 < line_no <= len(lines)
                    else match.group(0)
                )
                snippet = snippet.strip()[:240]
                key = (line_no, pattern_idx, category)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    CodeThreat(
                        category=category,
                        line_no=line_no,
                        snippet=snippet,
                        severity=severity,
                    )
                )

        hits.sort(
            key=lambda t: (
                t.line_no,
                -_SEVERITY_RANK[t.severity],
                t.category,
            )
        )

        if hits:
            max_sev = max(hits, key=lambda t: _SEVERITY_RANK[t.severity]).severity
        else:
            max_sev = "none"

        return CodeThreatReport(threats=hits, severity=max_sev, total=len(hits))


__all__ = [
    "CATEGORIES",
    "SEVERITY_LEVELS",
    "CodeThreat",
    "CodeThreatReport",
    "MaliciousCodeDetector",
]
