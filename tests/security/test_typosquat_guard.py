"""Tests for src.security.typosquat_guard (AUR-SEC-2026-0023)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.security.typosquat_guard import (
    TyposquatHit,
    TyposquatResult,
    check_typosquats,
    damerau_levenshtein,
    normalize_package_name,
)


def _write_lock(tmp_path: Path, packages: list[tuple[str, str]]) -> Path:
    parts = ["version = 1", "revision = 3", ""]
    for name, ver in packages:
        parts.append("[[package]]")
        parts.append(f'name = "{name}"')
        parts.append(f'version = "{ver}"')
        parts.append("")
    path = tmp_path / "uv.lock"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def test_exact_match_not_flagged(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path, [("numpy", "1.26.0"), ("torch", "2.5.0")])
    result = check_typosquats(lock)
    assert isinstance(result, TyposquatResult)
    assert result.suspicious == []
    assert result.total_checked == 2


def test_distance_1_flagged_high(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path, [("nupmy", "0.0.1")])  # transpose
    result = check_typosquats(lock)
    assert len(result.suspicious) == 1
    hit = result.suspicious[0]
    assert hit.package == "nupmy"
    assert hit.nearest_popular == "numpy"
    assert hit.distance == 1
    assert hit.severity == "high"


def test_distance_2_flagged_medium(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path, [("trnsch", "0.0.1")])  # from "torch" distance 2-3
    result = check_typosquats(lock)
    # At least one medium-ish. Try more reliable example:
    lock2 = _write_lock(tmp_path, [("nunpi", "0.0.1")])  # numpy -> nunpi: m->n, y->i dist=2
    r2 = check_typosquats(lock2)
    assert any(h.severity == "medium" and h.nearest_popular == "numpy" for h in r2.suspicious)


def test_distance_3_plus_not_flagged(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path, [("completelyunrelatedpkg", "1.0")])
    result = check_typosquats(lock)
    assert result.suspicious == []


def test_allowlist_skips(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path, [("nupmy", "0.0.1")])
    result = check_typosquats(lock, allowlist={"nupmy"})
    assert result.suspicious == []


def test_empty_lockfile(tmp_path: Path) -> None:
    path = tmp_path / "uv.lock"
    path.write_text("", encoding="utf-8")
    result = check_typosquats(path)
    assert result.total_checked == 0
    assert result.suspicious == []


def test_malformed_lockfile_raises(tmp_path: Path) -> None:
    path = tmp_path / "uv.lock"
    path.write_text("[[package]\nname = broken", encoding="utf-8")
    with pytest.raises(ValueError):
        check_typosquats(path)


def test_missing_lockfile_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        check_typosquats(tmp_path / "nope.lock")


def test_unicode_homoglyph_cyrillic(tmp_path: Path) -> None:
    # 'torсh' with cyrillic с (U+0441) looks like torch but isn't
    lock = _write_lock(tmp_path, [("torсh", "0.0.1")])
    result = check_typosquats(lock)
    assert len(result.suspicious) >= 1
    hit = result.suspicious[0]
    assert hit.nearest_popular == "torch"
    assert hit.severity == "high"


def test_dashes_underscores_normalized(tmp_path: Path) -> None:
    # scikit_learn should be treated same as scikit-learn (PEP 503)
    lock = _write_lock(tmp_path, [("scikit_learn", "1.0")])
    result = check_typosquats(lock)
    assert result.suspicious == []


def test_dash_typo_detected(tmp_path: Path) -> None:
    # "scikit-lern" (dist 1 from scikit-learn)
    lock = _write_lock(tmp_path, [("scikit-lern", "1.0")])
    result = check_typosquats(lock)
    assert any(h.nearest_popular == "scikit-learn" and h.severity == "high" for h in result.suspicious)


def test_damerau_levenshtein_basics() -> None:
    assert damerau_levenshtein("kitten", "kitten") == 0
    assert damerau_levenshtein("kitten", "sitten") == 1
    # transposition is distance 1 under Damerau (not 2 like plain Levenshtein)
    assert damerau_levenshtein("ab", "ba") == 1
    assert damerau_levenshtein("abcd", "acbd") == 1
    assert damerau_levenshtein("", "abc") == 3
    assert damerau_levenshtein("abc", "") == 3


def test_normalize_pep503() -> None:
    assert normalize_package_name("Scikit_Learn") == "scikit-learn"
    assert normalize_package_name("HuggingFace.Hub") == "huggingface-hub"
    assert normalize_package_name("Torch") == "torch"


def test_short_package_names_still_checked(tmp_path: Path) -> None:
    # "clack" vs "click" (distance 1) should flag
    lock = _write_lock(tmp_path, [("clack", "1.0")])
    result = check_typosquats(lock)
    assert any(h.nearest_popular == "click" and h.severity == "high" for h in result.suspicious)


def test_multiple_packages_mixed(tmp_path: Path) -> None:
    lock = _write_lock(
        tmp_path,
        [
            ("numpy", "1.26.0"),
            ("nupmy", "0.0.1"),
            ("requests", "2.32.0"),
            ("requesst", "0.0.1"),
            ("my-internal-pkg", "1.0"),
        ],
    )
    result = check_typosquats(lock)
    flagged = {h.package for h in result.suspicious}
    assert "nupmy" in flagged
    assert "requesst" in flagged
    assert "numpy" not in flagged
    assert "requests" not in flagged
    assert "my-internal-pkg" not in flagged
    assert result.total_checked == 5
