"""Global test fixtures and configuration."""

import os


def pytest_configure(config):
    """Set env vars before any test modules are collected."""
    os.environ.setdefault(
        "AURELIUS_ENCRYPTION_KEY",
        "dGVzdGluZ19rZXlfZm9yX2F1cmVsaXVzX3Rlc3Rz",
    )
