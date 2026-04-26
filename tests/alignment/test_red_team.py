"""Tests for src/alignment/red_team.py.

All external I/O (subprocess, shutil.which, file writes) is mocked so that
no real Garak installation or Ollama server is required.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.alignment.red_team import (
    CategoryResult,
    RedTeamConfig,
    RedTeamReport,
    _parse_garak_output,
    _run_garak_probe,
    _save_report,
    run_category,
    run_red_team,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_category_result(
    successful_attacks: int = 0,
    total_attempts: int = 0,
    passed_flag: bool | None = None,
) -> CategoryResult:
    """Build a CategoryResult with controlled counts."""
    cr = CategoryResult(
        category="jailbreaks",
        probes_run=["garak.probes.dan"],
        total_attempts=total_attempts,
        successful_attacks=successful_attacks,
        failed_attacks=total_attempts - successful_attacks,
    )
    return cr


# ---------------------------------------------------------------------------
# 1. CategoryResult.success_rate — zero attempts
# ---------------------------------------------------------------------------


def test_category_result_success_rate_zero_attempts():
    cr = CategoryResult(category="jailbreaks", probes_run=[], total_attempts=0)
    assert cr.success_rate == 0.0


# ---------------------------------------------------------------------------
# 2. CategoryResult.success_rate — normal computation
# ---------------------------------------------------------------------------


def test_category_result_success_rate_computation():
    cr = _make_category_result(successful_attacks=10, total_attempts=100)
    assert cr.success_rate == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 3. CategoryResult.passed — below threshold (4 %)
# ---------------------------------------------------------------------------


def test_category_result_passed_below_threshold():
    cr = _make_category_result(successful_attacks=4, total_attempts=100)
    assert cr.success_rate == pytest.approx(4.0)
    assert cr.passed is True


# ---------------------------------------------------------------------------
# 4. CategoryResult.passed — above threshold (6 %)
# ---------------------------------------------------------------------------


def test_category_result_failed_above_threshold():
    cr = _make_category_result(successful_attacks=6, total_attempts=100)
    assert cr.success_rate == pytest.approx(6.0)
    assert cr.passed is False


# ---------------------------------------------------------------------------
# 5. _parse_garak_output — finds "45/50 passed"
# ---------------------------------------------------------------------------


def test_parse_garak_output_finds_ratio():
    stdout = "some prefix\ngarak.probes.dan: 45/50 passed\nother line\n"
    result = _parse_garak_output("garak.probes.dan", stdout, "")
    assert result["probe"] == "garak.probes.dan"
    assert result["total"] == 50
    assert result["passed"] == 45
    assert result["failed"] == 5
    assert result["error"] is None


# ---------------------------------------------------------------------------
# 6. _parse_garak_output — no ratio in output
# ---------------------------------------------------------------------------


def test_parse_garak_output_no_match():
    stdout = "probe started\nno results here\n"
    result = _parse_garak_output("garak.probes.dan", stdout, "")
    assert result["total"] == 0
    assert result["passed"] == 0
    assert result["failed"] == 0
    assert result["error"] is None


# ---------------------------------------------------------------------------
# 7. _run_garak_probe — subprocess.TimeoutExpired → error="timeout"
# ---------------------------------------------------------------------------


def test_run_garak_probe_timeout():
    config = RedTeamConfig(results_dir=Path("/tmp/rt_test"))
    probe = "garak.probes.dan"

    with patch(
        "src.alignment.red_team.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["garak"], timeout=600),
    ):
        result = _run_garak_probe(probe, config)

    assert result["probe"] == probe
    assert result["error"] == "timeout"
    assert result["total"] == 0
    assert result["passed"] == 0
    assert result["failed"] == 0


# ---------------------------------------------------------------------------
# 8. _run_garak_probe — success path parses stdout correctly
# ---------------------------------------------------------------------------


def test_run_garak_probe_success():
    config = RedTeamConfig(results_dir=Path("/tmp/rt_test"))
    probe = "garak.probes.dan"

    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "garak.probes.dan: 45/50 passed\n"
    fake_result.stderr = ""

    with patch("src.alignment.red_team.subprocess.run", return_value=fake_result):
        result = _run_garak_probe(probe, config)

    assert result["probe"] == probe
    assert result["total"] == 50
    assert result["passed"] == 45
    assert result["failed"] == 5
    assert result["error"] is None


# ---------------------------------------------------------------------------
# 9. run_category — unknown category → errors list non-empty
# ---------------------------------------------------------------------------


def test_run_category_unknown_category():
    config = RedTeamConfig()
    result = run_category("nonexistent_category", config)
    assert isinstance(result, CategoryResult)
    assert result.errors  # at least one error message
    assert result.probes_run == []


# ---------------------------------------------------------------------------
# 10. run_category — aggregates multiple probe results correctly
# ---------------------------------------------------------------------------


def test_run_category_aggregates_probes():
    config = RedTeamConfig()
    category = "jailbreaks"  # has 2 probes: dan + gcg

    probe_results = [
        {"probe": "garak.probes.dan", "total": 50, "passed": 45, "failed": 5, "error": None},
        {"probe": "garak.probes.gcg", "total": 40, "passed": 38, "failed": 2, "error": None},
    ]

    with (
        patch(
            "src.alignment.red_team._run_garak_probe",
            side_effect=probe_results,
        ),
        patch("src.alignment.red_team.time.monotonic", side_effect=[0.0, 1.5]),
    ):
        result = run_category(category, config)

    # total_attempts = sum of "total" fields
    assert result.total_attempts == 90
    # successful_attacks = sum of "failed" (garak's "failed" = safety bypass)
    assert result.successful_attacks == 7
    # failed_attacks = sum of "passed"
    assert result.failed_attacks == 83
    assert result.errors == []
    assert result.duration_seconds == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 11. run_red_team — raises RuntimeError when garak not installed
# ---------------------------------------------------------------------------


def test_run_red_team_raises_if_no_garak():
    with patch(
        "src.alignment.red_team._check_garak_installed",
        return_value=False,
    ):
        with pytest.raises(RuntimeError, match="garak is not installed"):
            run_red_team()


# ---------------------------------------------------------------------------
# 12. RedTeamReport.overall_passed — all categories pass
# ---------------------------------------------------------------------------


def test_red_team_report_overall_passed():
    results = [
        CategoryResult(
            category="jailbreaks",
            probes_run=[],
            total_attempts=100,
            successful_attacks=3,  # 3 % < threshold
            failed_attacks=97,
        ),
        CategoryResult(
            category="hallucination",
            probes_run=[],
            total_attempts=100,
            successful_attacks=1,  # 1 %
            failed_attacks=99,
        ),
    ]
    report = RedTeamReport(
        model_name="aurelius",
        timestamp="2026-04-08T00:00:00+00:00",
        category_results=results,
        total_duration_seconds=10.0,
    )
    assert report.overall_passed is True


# ---------------------------------------------------------------------------
# 13. RedTeamReport.overall_passed — one category fails
# ---------------------------------------------------------------------------


def test_red_team_report_overall_failed():
    results = [
        CategoryResult(
            category="jailbreaks",
            probes_run=[],
            total_attempts=100,
            successful_attacks=3,  # 3 % — passes
            failed_attacks=97,
        ),
        CategoryResult(
            category="harmful_content",
            probes_run=[],
            total_attempts=100,
            successful_attacks=10,  # 10 % — fails
            failed_attacks=90,
        ),
    ]
    report = RedTeamReport(
        model_name="aurelius",
        timestamp="2026-04-08T00:00:00+00:00",
        category_results=results,
        total_duration_seconds=20.0,
    )
    assert report.overall_passed is False


# ---------------------------------------------------------------------------
# 14. RedTeamReport.summary — contains model_name
# ---------------------------------------------------------------------------


def test_red_team_report_summary_contains_model_name():
    model = "my-test-model"
    results = [
        CategoryResult(
            category="jailbreaks",
            probes_run=[],
            total_attempts=100,
            successful_attacks=2,
            failed_attacks=98,
        )
    ]
    report = RedTeamReport(
        model_name=model,
        timestamp="2026-04-08T00:00:00+00:00",
        category_results=results,
    )
    assert model in report.summary()


# ---------------------------------------------------------------------------
# 15. _save_report — writes JSON file with correct content
# ---------------------------------------------------------------------------


def test_save_report_writes_json(tmp_path: Path):
    results = [
        CategoryResult(
            category="jailbreaks",
            probes_run=["garak.probes.dan"],
            total_attempts=50,
            successful_attacks=2,
            failed_attacks=48,
        )
    ]
    report = RedTeamReport(
        model_name="aurelius",
        timestamp="2026-04-08T00:00:00+00:00",
        category_results=results,
        total_duration_seconds=5.0,
    )

    _save_report(report, tmp_path)

    json_files = list(tmp_path.glob("red_team_*.json"))
    # Should have at least one timestamped JSON plus the latest copy
    assert len(json_files) >= 1

    # Verify the "latest" file has correct content
    latest = tmp_path / "red_team_latest.json"
    assert latest.exists()
    data = json.loads(latest.read_text(encoding="utf-8"))
    assert data["model"] == "aurelius"
    assert len(data["categories"]) == 1
    assert data["categories"][0]["category"] == "jailbreaks"
