"""Shared fixtures for integration tests."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import requests

API_KEY = os.environ.get("AURELIUS_API_KEY", "dev-key")
MIDDLE_DIR = Path(__file__).resolve().parents[2] / "middle"


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_ready(base_url: str, timeout: float = 45.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/readyz", timeout=1)
            if resp.status_code == 200:
                return
        except Exception as exc:  # pragma: no cover - retry loop
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Middle server did not become ready at {base_url}") from last_error


@pytest.fixture(scope="session")
def middle_server() -> Generator[str, None, None]:
    existing_url = os.environ.get("MIDDLE_URL")
    if existing_url:
        _wait_for_ready(existing_url)
        yield existing_url
        return

    port = _pick_free_port()
    env = os.environ.copy()
    env["MIDDLE_HOST"] = "127.0.0.1"
    env["MIDDLE_PORT"] = str(port)
    env.setdefault("AURELIUS_API_KEY", API_KEY)
    env.setdefault("AURELIUS_SERVICE_KEY", API_KEY)
    env.setdefault("ALLOW_PUBLIC_REGISTRATION", "true")

    # Start the repo-local middle server with a fixed command and controlled env.
    proc = subprocess.Popen(  # noqa: S603
        [str(MIDDLE_DIR / "node_modules" / ".bin" / "tsx"), "src/index.ts"],
        cwd=MIDDLE_DIR,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_for_ready(base_url)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="session")
def base_url(middle_server: str) -> str:
    return middle_server


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
