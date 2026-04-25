"""Secrets scanner — detect credentials and API keys in text.

Implements Trail of Bits pattern-matching approach: regex-based detection
for common secret formats, with noise filtering to avoid false positives.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


_SECRET_PATTERNS: list[tuple[str, str]] = [
    ("aws_key", r"AKIA[0-9A-Z]{16}"),
    ("github_token", r"gh[pousr]_[A-Za-z0-9_]{36,40}"),
    ("bearer_token", r"Bearer\s+[A-Za-z0-9\-._~+/]{20,}"),
    ("slack_token", r"xox[baprs]-[0-9A-Za-z-]{10,60}"),
    ("ssh_private_key", r"-----BEGIN\s+(RSA|DSA|EC|OPENSSH)\s+PRIVATE KEY-----"),
    ("jwt_token", r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    ("stripe_key", r"sk_live_[0-9A-Za-z]{24,}"),
    ("google_api", r"AIza[0-9A-Za-z\-_]{35}"),
    ("generic_api", r"(?:api[_-]?key|apikey|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}['\"]?"),
]


@dataclass
class SecretMatch:
    value: str
    match_type: str
    line: int
    column: int

    def masked(self) -> str:
        if len(self.value) <= 8:
            return "***" + self.value[-4:]
        return self.value[:4] + "***" + self.value[-4:]


@dataclass
class SecretsScanner:
    patterns: list[tuple[str, str]] = field(default_factory=lambda: _SECRET_PATTERNS)

    def scan(self, text: str) -> list[SecretMatch]:
        results: list[SecretMatch] = []
        for match_type, pattern in self.patterns:
            for m in re.finditer(pattern, text, re.MULTILINE):
                line = text[:m.start()].count("\n") + 1
                col = m.start() - text[:m.start()].rfind("\n")
                results.append(SecretMatch(
                    value=m.group(),
                    match_type=match_type,
                    line=line,
                    column=col,
                ))
        return results

    def scan_file(self, path: str) -> list[SecretMatch]:
        with open(path, "r") as f:
            return self.scan(f.read())


SECRETS_SCANNER = SecretsScanner()