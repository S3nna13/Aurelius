"""Honeytoken generator for detecting unauthorized data access."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Honeytoken:
    token: str
    location: str
    created: str
    accessed: bool = False
    access_time: str | None = None
    access_source: str | None = None


@dataclass
class HoneytokenManager:
    _tokens: dict[str, Honeytoken] = field(default_factory=dict, repr=False)

    def generate(self, location: str) -> Honeytoken:
        import uuid
        from datetime import datetime, timezone
        token_str = f"hk_{uuid.uuid4().hex}"
        token = Honeytoken(
            token=token_str,
            location=location,
            created=datetime.now(timezone.utc).isoformat(),
        )
        self._tokens[token_str] = token
        return token

    def check(self, token_str: str, source: str = "") -> Honeytoken | None:
        token = self._tokens.get(token_str)
        if token is None:
            return None
        from datetime import datetime, timezone
        token.accessed = True
        token.access_time = datetime.now(timezone.utc).isoformat()
        token.access_source = source
        return token

    def alerts(self) -> list[Honeytoken]:
        return [t for t in self._tokens.values() if t.accessed]

    def clear(self) -> None:
        self._tokens.clear()


HONEYTOKEN_MANAGER = HoneytokenManager()