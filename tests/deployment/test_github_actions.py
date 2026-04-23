"""Tests for src.deployment.github_actions — GitHub Actions CI YAML generator.

WebArena benchmark (Zhou et al. 2307.13854, Apache-2.0), GitHub Actions (MIT spec),
clean-room implementation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.deployment.github_actions import (
    GHActionsError,
    GHActionsGenerator,
    GHActionsJob,
    GHActionsTrigger,
    GHActionsWorkflow,
)


# ---------------------------------------------------------------------------
# Dataclass defaults
# ---------------------------------------------------------------------------


def test_trigger_default_push_includes_main() -> None:
    """GHActionsTrigger defaults include 'main' in on_push_branches."""
    trigger = GHActionsTrigger()
    assert "main" in trigger.on_push_branches


def test_trigger_default_push_includes_codex_glob() -> None:
    """GHActionsTrigger defaults include 'codex/*' in on_push_branches."""
    trigger = GHActionsTrigger()
    assert "codex/*" in trigger.on_push_branches


def test_trigger_default_pull_request_includes_main() -> None:
    """GHActionsTrigger defaults include 'main' in on_pull_request_branches."""
    trigger = GHActionsTrigger()
    assert "main" in trigger.on_pull_request_branches


def test_trigger_default_schedule_is_none() -> None:
    """GHActionsTrigger defaults on_schedule_cron to None."""
    trigger = GHActionsTrigger()
    assert trigger.on_schedule_cron is None


def test_job_default_runs_on() -> None:
    """GHActionsJob defaults runs_on to 'ubuntu-latest'."""
    job = GHActionsJob(job_id="test", name="Test")
    assert job.runs_on == "ubuntu-latest"


def test_job_default_python_version() -> None:
    """GHActionsJob defaults python_version to '3.14'."""
    job = GHActionsJob(job_id="test", name="Test")
    assert job.python_version == "3.14"


def test_job_default_steps_empty() -> None:
    """GHActionsJob defaults steps to an empty list."""
    job = GHActionsJob(job_id="test", name="Test")
    assert job.steps == []


# ---------------------------------------------------------------------------
# generate_yaml
# ---------------------------------------------------------------------------


def _make_simple_workflow(steps: list[str] | None = None) -> GHActionsWorkflow:
    trigger = GHActionsTrigger()
    job = GHActionsJob(
        job_id="build",
        name="Build and Test",
        steps=steps or ["pytest --tb=short -q tests/"],
    )
    return GHActionsWorkflow(name="ci", trigger=trigger, jobs=[job])


def test_generate_yaml_contains_name_key() -> None:
    """generate_yaml output contains a 'name:' key."""
    gen = GHActionsGenerator()
    yaml = gen.generate_yaml(_make_simple_workflow())
    assert "name:" in yaml


def test_generate_yaml_contains_on_key() -> None:
    """generate_yaml output contains an 'on:' section."""
    gen = GHActionsGenerator()
    yaml = gen.generate_yaml(_make_simple_workflow())
    assert "\non:" in yaml or yaml.startswith("on:")


def test_generate_yaml_contains_jobs_key() -> None:
    """generate_yaml output contains a 'jobs:' section."""
    gen = GHActionsGenerator()
    yaml = gen.generate_yaml(_make_simple_workflow())
    assert "jobs:" in yaml


def test_generate_yaml_contains_checkout_action() -> None:
    """generate_yaml includes 'actions/checkout' step."""
    gen = GHActionsGenerator()
    yaml = gen.generate_yaml(_make_simple_workflow())
    assert "actions/checkout" in yaml


def test_generate_yaml_contains_setup_python_action() -> None:
    """generate_yaml includes 'actions/setup-python' step."""
    gen = GHActionsGenerator()
    yaml = gen.generate_yaml(_make_simple_workflow())
    assert "actions/setup-python" in yaml


def test_generate_yaml_contains_pip_install() -> None:
    """generate_yaml includes a pip install step."""
    gen = GHActionsGenerator()
    yaml = gen.generate_yaml(_make_simple_workflow())
    assert "pip install" in yaml


def test_generate_yaml_contains_custom_steps() -> None:
    """generate_yaml embeds the custom steps from GHActionsJob.steps."""
    gen = GHActionsGenerator()
    custom_step = "pytest --tb=short -q tests/"
    yaml = gen.generate_yaml(_make_simple_workflow(steps=[custom_step]))
    assert "pytest" in yaml
    assert "--tb=short" in yaml


def test_generate_yaml_raises_on_empty_jobs() -> None:
    """generate_yaml raises GHActionsError when the workflow has no jobs."""
    gen = GHActionsGenerator()
    workflow = GHActionsWorkflow(
        name="empty",
        trigger=GHActionsTrigger(),
        jobs=[],
    )
    with pytest.raises(GHActionsError):
        gen.generate_yaml(workflow)


# ---------------------------------------------------------------------------
# write_workflow
# ---------------------------------------------------------------------------


def test_write_workflow_creates_file(tmp_path: Path) -> None:
    """write_workflow writes the YAML file under .github/workflows/."""
    gen = GHActionsGenerator()
    workflow = _make_simple_workflow()
    out_path = gen.write_workflow(workflow, tmp_path)
    assert out_path.exists()


def test_write_workflow_correct_path(tmp_path: Path) -> None:
    """write_workflow places file at .github/workflows/<name>.yml."""
    gen = GHActionsGenerator()
    workflow = _make_simple_workflow()
    out_path = gen.write_workflow(workflow, tmp_path)
    expected = tmp_path / ".github" / "workflows" / "ci.yml"
    assert out_path == expected


def test_write_workflow_file_contains_yaml(tmp_path: Path) -> None:
    """Written file contains valid YAML content (name: and jobs:)."""
    gen = GHActionsGenerator()
    workflow = _make_simple_workflow()
    out_path = gen.write_workflow(workflow, tmp_path)
    content = out_path.read_text()
    assert "name:" in content
    assert "jobs:" in content


# ---------------------------------------------------------------------------
# default_ci_workflow
# ---------------------------------------------------------------------------


def test_default_ci_workflow_returns_workflow() -> None:
    """default_ci_workflow returns a GHActionsWorkflow instance."""
    gen = GHActionsGenerator()
    workflow = gen.default_ci_workflow()
    assert isinstance(workflow, GHActionsWorkflow)


def test_default_ci_workflow_has_pytest_step() -> None:
    """default_ci_workflow includes a pytest step in at least one job."""
    gen = GHActionsGenerator()
    workflow = gen.default_ci_workflow()
    all_steps = [s for job in workflow.jobs for s in job.steps]
    assert any("pytest" in s for s in all_steps)


def test_default_ci_workflow_python_version_override() -> None:
    """default_ci_workflow respects the python_version parameter."""
    gen = GHActionsGenerator()
    workflow = gen.default_ci_workflow(python_version="3.13")
    assert all(job.python_version == "3.13" for job in workflow.jobs)


# ---------------------------------------------------------------------------
# GHActionsError
# ---------------------------------------------------------------------------


def test_ghactions_error_is_exception() -> None:
    """GHActionsError is a subclass of Exception."""
    assert issubclass(GHActionsError, Exception)


def test_ghactions_error_can_be_raised() -> None:
    """GHActionsError can be raised and caught."""
    with pytest.raises(GHActionsError, match="test error"):
        raise GHActionsError("test error")


# ---------------------------------------------------------------------------
# Registry check
# ---------------------------------------------------------------------------


def test_artifact_builder_registry_contains_github_actions() -> None:
    """ARTIFACT_BUILDER_REGISTRY contains 'github_actions' after import."""
    import src.deployment as deployment_module

    registry = deployment_module.ARTIFACT_BUILDER_REGISTRY
    assert "github_actions" in registry
