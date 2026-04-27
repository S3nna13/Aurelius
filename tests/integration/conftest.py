"""Shared fixtures for integration tests."""

from __future__ import annotations

import os
from typing import Any, Generator

import pytest
import requests

BASE_URL = os.environ.get("MIDDLE_URL", "http://localhost:3001")
API_KEY = os.environ.get("AURELIUS_API_KEY", "dev-key")


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session")
def api_key() -> str:
    return API_KEY


@pytest.fixture(scope="session")
def auth_headers(api_key: str) -> dict[str, str]:
    return {"X-API-Key": api_key, "Content-Type": "application/json"}


@pytest.fixture(scope="session")
def api_client(base_url: str, auth_headers: dict[str, str]) -> Generator[Any, None, None]:
    session = requests.Session()
    session.headers.update(auth_headers)
    session.base_url = base_url  # type: ignore[attr-defined]
    yield session
    session.close()


@pytest.fixture(scope="session")
def health_url(base_url: str) -> str:
    return f"{base_url}/health"


@pytest.fixture(scope="session")
def readyz_url(base_url: str) -> str:
    return f"{base_url}/readyz"
