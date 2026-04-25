"""Content Security Policy validator and header builder."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CSPDirective:
    directive: str
    sources: list[str]

    def to_string(self) -> str:
        return f"{self.directive} {' '.join(self.sources)}"


@dataclass
class CSPBuilder:
    directives: list[CSPDirective] = field(default_factory=list)

    def add(self, directive: CSPDirective) -> None:
        self.directives.append(directive)

    def build(self) -> str:
        return "; ".join(d.to_string() for d in self.directives)

    def to_header(self) -> dict[str, str]:
        return {"Content-Security-Policy": self.build()}

    @classmethod
    def strict_default(cls) -> CSPBuilder:
        return cls(directives=[
            CSPDirective("default-src", ["'self'"]),
            CSPDirective("script-src", ["'self'"]),
            CSPDirective("style-src", ["'self'", "'unsafe-inline'"]),
            CSPDirective("img-src", ["'self'", "data:"]),
            CSPDirective("connect-src", ["'self'"]),
            CSPDirective("frame-ancestors", ["'none'"]),
            CSPDirective("base-uri", ["'self'"]),
            CSPDirective("form-action", ["'self'"]),
        ])


CSP_STRICT = CSPBuilder.strict_default()