"""Secure random token generator for API keys and session tokens."""

from __future__ import annotations

import secrets
from dataclasses import dataclass


@dataclass
class TokenGenerator:
    """Generate cryptographically secure random tokens."""

    byte_length: int = 32

    def generate(self) -> str:
        return secrets.token_hex(self.byte_length)

    def generate_urlsafe(self) -> str:
        return secrets.token_urlsafe(self.byte_length)

    def generate_pin(self, digits: int = 6) -> str:
        return "".join(str(secrets.randbelow(10)) for _ in range(digits))


TOKEN_GENERATOR = TokenGenerator()
