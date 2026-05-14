"""Shared fixtures for integration tests."""

@pytest.mark.integration
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pytest
import requests

BASE_URL = os.environ.get("MIDDLE_URL", "http://localhost:3001")
API_KEY = os.environ.get("AURELIUS_API_KEY", "dev-key")
ROOT = Path(__file__).resolve().parents[2]
MIDDLE_DIR = ROOT / "middle"
UPSTREAM_URL = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8080")
UPSTREAM_HOST = os.environ.get("AURELIUS_API_HOST", "127.0.0.1")
UPSTREAM_PORT = int(os.environ.get("AURELIUS_API_PORT", "8080"))


def _is_local_default_base(url: str) -> bool:
    parsed = urlparse(url)
    return (
        parsed.scheme == "http"
        and parsed.hostname in {"localhost", "127.0.0.1"}
        and parsed.port == 3001
    )


def _healthcheck(url: str) -> bool:
    try:
        return requests.get(url, timeout=0.5).ok
    except requests.RequestException:
        return False


def _wait_for_health(
    url: str,
    proc: subprocess.Popen[str],
    log_path: Path,
    timeout_s: int = 60,
) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"Process exited with code {proc.returncode} while waiting for {url}\n"
                f"{log_path.read_text(errors='replace')}"
            )
        if _healthcheck(url):
            return
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {url}\n{log_path.read_text(errors='replace')}")


def _start_logged_process(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
) -> tuple[subprocess.Popen[str], Path]:
    log_file = tempfile.NamedTemporaryFile(
        prefix="aurelius-integration-",
        suffix=".log",
        delete=False,
    )
    log_path = Path(log_file.name)
    proc = subprocess.Popen(  # noqa: S603 - fixed command list for local test harness
        command,
        cwd=cwd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_file.close()
    return proc, log_path


def _stop_process(proc: subprocess.Popen[str], log_path: Path) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    try:
        log_path.unlink(missing_ok=True)
    except OSError:
        pass


@pytest.fixture(scope="session", autouse=True)
def integration_stack() -> Generator[None, None, None]:
    """Start the local BFF and mock upstream if no external service is configured."""

    if not _is_local_default_base(BASE_URL):
        yield
        return

    processes: list[tuple[subprocess.Popen[str], Path]] = []
    try:
        base_env = os.environ.copy()
        base_env.setdefault("AURELIUS_API_KEY", API_KEY)
        base_env.setdefault("AURELIUS_SERVICE_KEY", API_KEY)

        upstream_url = UPSTREAM_URL.rstrip("/")
        if not _healthcheck(f"{upstream_url}/health"):
            upstream_env = base_env.copy()
            upstream_env.setdefault("AURELIUS_API_HOST", UPSTREAM_HOST)
            upstream_env.setdefault("AURELIUS_API_PORT", str(UPSTREAM_PORT))
            upstream_cmd = [
                sys.executable,
                "-m",
                "src.backend",
                "api",
                "--host",
                UPSTREAM_HOST,
                "--port",
                str(UPSTREAM_PORT),
            ]
            upstream_proc, upstream_log = _start_logged_process(upstream_cmd, ROOT, upstream_env)
            processes.append((upstream_proc, upstream_log))
            _wait_for_health(f"{upstream_url}/health", upstream_proc, upstream_log)

        if not _healthcheck(f"{BASE_URL}/health"):
            middle_env = base_env.copy()
            middle_env.setdefault("MIDDLE_HOST", "127.0.0.1")
            middle_env.setdefault("MIDDLE_PORT", "3001")
            middle_env.setdefault("UPSTREAM_URL", upstream_url)
            middle_env["ALLOW_PUBLIC_REGISTRATION"] = "true"
            middle_cmd = ["npm", "run", "dev"]
            middle_proc, middle_log = _start_logged_process(middle_cmd, MIDDLE_DIR, middle_env)
            processes.append((middle_proc, middle_log))
            _wait_for_health(f"{BASE_URL}/health", middle_proc, middle_log)

        yield
    finally:
        for proc, log_path in reversed(processes):
            _stop_process(proc, log_path)


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
