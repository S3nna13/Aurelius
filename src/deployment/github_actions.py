"""GitHub Actions CI YAML generator for Aurelius deployment surface.

WebArena benchmark (Zhou et al. 2307.13854, Apache-2.0), GitHub Actions (MIT spec),
clean-room implementation.

Generates valid GitHub Actions workflow YAML using stdlib string formatting only
(no external YAML library).
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GHActionsError(Exception):
    """Raised when GitHub Actions YAML generation encounters an error."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GHActionsTrigger:
    """Defines when a GitHub Actions workflow is triggered."""

    on_push_branches: list[str] = field(
        default_factory=lambda: ["main", "codex/*"]
    )
    on_pull_request_branches: list[str] = field(
        default_factory=lambda: ["main"]
    )
    on_schedule_cron: str | None = None


@dataclass
class GHActionsJob:
    """A single job in a GitHub Actions workflow."""

    job_id: str
    name: str
    runs_on: str = "ubuntu-latest"
    python_version: str = "3.14"
    steps: list[str] = field(default_factory=list)


@dataclass
class GHActionsWorkflow:
    """A complete GitHub Actions workflow definition."""

    name: str
    trigger: GHActionsTrigger
    jobs: list[GHActionsJob]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _yaml_branch_list(branches: list[str]) -> str:
    """Render a list of branches as a YAML sequence (indented 8 spaces)."""
    return "\n".join(f"        - {b}" for b in branches)


def _sanitize_workflow_name(name: str) -> str:
    """Convert workflow name to a safe filename stem (lowercase, hyphens)."""
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name.lower())


class GHActionsGenerator:
    """Generates GitHub Actions workflow YAML files from structured definitions."""

    def generate_yaml(self, workflow: GHActionsWorkflow) -> str:
        """Render a valid GitHub Actions YAML string.

        Includes: name, on (push/pull_request/schedule), jobs with
        actions/checkout@v4, actions/setup-python@v5, pip install, and
        any custom steps.

        Args:
            workflow: The workflow definition to render.

        Returns:
            A YAML string suitable for writing to .github/workflows/<name>.yml.

        Raises:
            GHActionsError: If the workflow has no jobs.
        """
        if not workflow.jobs:
            raise GHActionsError("Workflow must contain at least one job.")

        trigger = workflow.trigger
        lines: list[str] = [
            f"name: {workflow.name}",
            "",
            "on:",
        ]

        # Push trigger
        if trigger.on_push_branches:
            lines.append("  push:")
            lines.append("    branches:")
            for b in trigger.on_push_branches:
                lines.append(f"      - '{b}'")

        # Pull request trigger
        if trigger.on_pull_request_branches:
            lines.append("  pull_request:")
            lines.append("    branches:")
            for b in trigger.on_pull_request_branches:
                lines.append(f"      - '{b}'")

        # Schedule trigger
        if trigger.on_schedule_cron:
            lines.append("  schedule:")
            lines.append(f"    - cron: '{trigger.on_schedule_cron}'")

        lines.append("")
        lines.append("jobs:")

        for job in workflow.jobs:
            lines.append(f"  {job.job_id}:")
            lines.append(f"    name: {job.name}")
            lines.append(f"    runs-on: {job.runs_on}")
            lines.append("    steps:")

            # Mandatory: checkout
            lines.append("      - name: Checkout repository")
            lines.append("        uses: actions/checkout@v4")

            # Mandatory: setup-python
            lines.append("      - name: Set up Python")
            lines.append("        uses: actions/setup-python@v5")
            lines.append("        with:")
            lines.append(f"          python-version: '{job.python_version}'")

            # Mandatory: pip install
            lines.append("      - name: Install dependencies")
            lines.append("        run: |")
            lines.append("          python -m pip install --upgrade pip")
            lines.append("          pip install -e .")

            # Custom steps
            for step_cmd in job.steps:
                # Use the first line as the step name, rest as run body
                step_lines = step_cmd.strip().splitlines()
                step_name = step_lines[0].lstrip("#").strip() if step_lines else "Run step"
                lines.append(f"      - name: {step_name}")
                lines.append("        run: |")
                for sl in step_lines:
                    lines.append(f"          {sl}")

        lines.append("")
        return "\n".join(lines)

    def write_workflow(
        self, workflow: GHActionsWorkflow, output_dir: Path
    ) -> Path:
        """Write workflow YAML to output_dir/.github/workflows/<name>.yml.

        Creates intermediate directories if needed.

        Args:
            workflow: The workflow to write.
            output_dir: Root directory (e.g. the repo root).

        Returns:
            Path to the written YAML file.
        """
        workflows_dir = output_dir / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        safe_name = _sanitize_workflow_name(workflow.name)
        output_path = workflows_dir / f"{safe_name}.yml"
        output_path.write_text(self.generate_yaml(workflow))
        return output_path

    def default_ci_workflow(self, python_version: str = "3.14") -> GHActionsWorkflow:
        """Build a pre-configured CI workflow with test and lint steps.

        Args:
            python_version: Python version string to use in setup-python.

        Returns:
            A GHActionsWorkflow ready to generate YAML from.
        """
        trigger = GHActionsTrigger(
            on_push_branches=["main", "codex/*"],
            on_pull_request_branches=["main"],
        )
        test_job = GHActionsJob(
            job_id="test",
            name="Run tests",
            runs_on="ubuntu-latest",
            python_version=python_version,
            steps=[
                "pytest --tb=short -q tests/",
            ],
        )
        lint_job = GHActionsJob(
            job_id="lint",
            name="Lint",
            runs_on="ubuntu-latest",
            python_version=python_version,
            steps=[
                "python -m py_compile $(find src -name '*.py' | head -20)",
            ],
        )
        return GHActionsWorkflow(
            name="ci",
            trigger=trigger,
            jobs=[test_job, lint_job],
        )
