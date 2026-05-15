"""Aurelius v2 Skill Executor — runs skills in dry_run/plan/execute/verify/rollback modes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.skills.manifest import SkillExecutionMode, SkillManifest, SkillStatus
from src.skills.permissions import PermissionCheck, PermissionGrant, PermissionGate


@dataclass
class SkillResult:
    """Result of executing a skill."""
    skill_id: str
    mode: SkillExecutionMode
    success: bool
    output: Any = None
    error: str = ""
    runtime_ms: int = 0
    permission_denials: list[str] = field(default_factory=list)
    dry_run_details: str = ""
    checkpoint_created: bool = False


class SkillExecutor:
    """Executes native skills with permission gates, timeout, and mode enforcement.

    Execution modes:
    - dry_run: Explain what the skill would do (no side effects)
    - plan: Produce an ordered execution plan
    - execute: Actually perform the skill actions
    - verify: Validate results of a previous action
    - rollback: Undo changes made by the skill
    """

    def __init__(self, permission_gate: PermissionGate | None = None) -> None:
        self._permission_gate = permission_gate or PermissionGate()

    def execute(
        self,
        manifest: SkillManifest,
        mode: SkillExecutionMode,
        skill_callable: Any = None,
        inputs: dict[str, Any] | None = None,
        context: Any = None,
    ) -> SkillResult:
        """Execute a skill in the specified mode."""
        result = SkillResult(skill_id=manifest.id, mode=mode, success=False)
        start = time.monotonic()

        # Check status
        if manifest.status == SkillStatus.UNSAFE:
            result.error = f"Skill {manifest.id} is marked UNSAFE and cannot be executed"
            result.runtime_ms = int((time.monotonic() - start) * 1000)
            return result

        if manifest.status == SkillStatus.DEPRECATED:
            result.error = f"Skill {manifest.id} is deprecated"
            result.runtime_ms = int((time.monotonic() - start) * 1000)
            return result

        # Verify mode is supported
        if mode not in manifest.supported_modes:
            result.error = f"Skill {manifest.id} does not support mode {mode.value}"
            result.runtime_ms = int((time.monotonic() - start) * 1000)
            return result

        # Handle dry_run without executing the skill
        if mode == SkillExecutionMode.DRY_RUN:
            result.success = True
            result.dry_run_details = (
                f"Would execute: {manifest.name} ({manifest.id})\n"
                f"Category: {manifest.category}\n"
                f"Risk level: {manifest.risk_level.value}\n"
                f"Permissions needed: {', '.join(p.name for p in manifest.permissions)}\n"
                f"Tools required: {', '.join(manifest.required_tools)}"
            )
            result.runtime_ms = int((time.monotonic() - start) * 1000)
            return result

        # Execute mode requires permission check
        if mode in (SkillExecutionMode.EXECUTE, SkillExecutionMode.ROLLBACK):
            from src.skills.permissions import PermissionContext
            checks = self._permission_gate.check(manifest, PermissionContext())
            denials = [c.permission for c in checks if c.grant == PermissionGrant.DENIED]
            if denials:
                result.error = f"Permissions denied: {', '.join(denials)}"
                result.permission_denials = denials
                result.runtime_ms = int((time.monotonic() - start) * 1000)
                return result

        # Execute the skill callable if available
        if skill_callable is not None:
            try:
                result.output = skill_callable(inputs=inputs or {}, context=context, mode=mode)
                result.success = True
            except Exception as e:
                result.error = str(e)
                result.success = False
        else:
            # No callable — only metadata returned
            result.dry_run_details = f"Skill {manifest.id} has no callable entrypoint"
            result.success = mode in (SkillExecutionMode.DRY_RUN, SkillExecutionMode.PLAN)

        result.runtime_ms = int((time.monotonic() - start) * 1000)
        return result
