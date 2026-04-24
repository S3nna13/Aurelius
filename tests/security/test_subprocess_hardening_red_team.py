"""Regression tests: red_team probe dispatch uses run_safe (AUR-SEC-2026-0021)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

from src.alignment import red_team
from src.security.safe_subprocess import SafeRunResult


def test_python_abs_resolved_at_import() -> None:
    assert os.path.isabs(red_team._PYTHON_ABS)


def test_run_garak_probe_uses_run_safe(tmp_path: Path) -> None:
    config = red_team.RedTeamConfig(
        model_name="test-model",
        results_dir=tmp_path,
        timeout_per_category=1.0,
    )
    fake = SafeRunResult(
        returncode=0,
        stdout="attempts: 10\npassed: 10\n",
        stderr="",
        duration_s=0.01,
        killed_on_timeout=False,
    )
    with mock.patch.object(red_team, "run_safe", return_value=fake) as m:
        result = red_team._run_garak_probe("garak.probes.dan", config)
        assert m.called
        argv, = m.call_args.args if m.call_args.args else ([],)
        # argv[0] is the absolute interpreter path.
        assert os.path.isabs(argv[0])
        assert argv[0] == red_team._PYTHON_ABS
    assert isinstance(result, dict)


def test_run_garak_probe_timeout_path(tmp_path: Path) -> None:
    config = red_team.RedTeamConfig(
        model_name="test-model",
        results_dir=tmp_path,
        timeout_per_category=1.0,
    )
    fake = SafeRunResult(
        returncode=124,
        stdout="",
        stderr="",
        duration_s=1.0,
        killed_on_timeout=True,
    )
    with mock.patch.object(red_team, "run_safe", return_value=fake):
        result = red_team._run_garak_probe("garak.probes.dan", config)
    assert result["error"] == "timeout"
