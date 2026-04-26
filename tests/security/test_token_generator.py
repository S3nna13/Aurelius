"""Tests for token generator."""

from __future__ import annotations

from src.security.token_generator import TokenGenerator


class TestTokenGenerator:
    def test_generate_returns_hex(self):
        tg = TokenGenerator(byte_length=16)
        token = tg.generate()
        assert len(token) == 32  # 16 bytes = 32 hex chars

    def test_generate_urlsafe(self):
        tg = TokenGenerator(byte_length=16)
        token = tg.generate_urlsafe()
        assert len(token) > 0
        assert isinstance(token, str)

    def test_generate_pin(self):
        tg = TokenGenerator()
        pin = tg.generate_pin(6)
        assert len(pin) == 6
        assert pin.isdigit()

    def test_unique_tokens(self):
        tg = TokenGenerator()
        tokens = {tg.generate() for _ in range(100)}
        assert len(tokens) == 100
