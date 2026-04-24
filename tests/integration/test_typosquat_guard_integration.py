"""Integration test for typosquat_guard against the real uv.lock.

Finding AUR-SEC-2026-0023 (typosquat); CWE-494.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.security.typosquat_guard import check_typosquats

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK = REPO_ROOT / "uv.lock"


@pytest.mark.skipif(not LOCK.exists(), reason="no uv.lock in repo root")
def test_real_uv_lock_parses() -> None:
    result = check_typosquats(LOCK)
    # We always expect to parse *some* packages from the real lock.
    assert result.total_checked > 0


@pytest.mark.skipif(not LOCK.exists(), reason="no uv.lock in repo root")
def test_real_uv_lock_report(capsys: pytest.CaptureFixture[str]) -> None:
    result = check_typosquats(LOCK)
    if result.suspicious:
        print(f"[typosquat] {len(result.suspicious)} suspicious packages in uv.lock:")
        for hit in result.suspicious:
            print(
                f"  - {hit.package}=={hit.version} near '{hit.nearest_popular}' "
                f"(dist={hit.distance}, severity={hit.severity})"
            )
    else:
        print(f"[typosquat] 0 suspicious packages across {result.total_checked} deps")
    # Don't fail the test — just report.
    assert result.total_checked > 0
