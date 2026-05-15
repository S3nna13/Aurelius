"""Aurelius v2 Skill Permissions — permission gate and enforcement."""

from __future__ import annotations

import enum
from dataclasses import dataclass

from src.skills.manifest import RiskLevel, SkillManifest, SkillPermission


class PermissionGrant(enum.StrEnum):
    GRANTED = "granted"
    DENIED = "denied"
    REQUIRES_APPROVAL = "requires_approval"


@dataclass
class PermissionCheck:
    """Result of checking a single permission."""
    permission: str
    grant: PermissionGrant
    reason: str = ""
    scope: str = ""


@dataclass
class PermissionContext:
    """Context used when evaluating permissions."""
    safety_mode: str = "balanced"  # safe, balanced, unrestricted
    user_approval_mode: str = "inline"  # inline, batch, always, never
    hardware_profile_id: str = ""
    task_risk: str = "low"
    skill_trust_tier: int = 1  # 0=untrusted, 1=first-party, 2=battle-tested


# Known permission types
KNOWN_PERMISSIONS: set[str] = {
    "file_read", "file_write", "terminal", "network", "browser",
    "cua", "memory_read", "memory_write", "secrets_access",
    "external_service", "background_job",
}


class PermissionGate:
    """Evaluates and enforces skill permissions.

    Permission grants depend on:
    - Current safety mode
    - User profile approval settings
    - Hardware profile capabilities
    - Task risk assessment
    - Skill trust tier
    """

    def check(self, manifest: SkillManifest, context: PermissionContext | None = None) -> list[PermissionCheck]:
        """Check all permissions for a skill against the current context."""
        ctx = context or PermissionContext()
        results: list[PermissionCheck] = []

        for perm in manifest.permissions:
            grant = self._evaluate(perm, ctx, manifest.risk_level)
            results.append(PermissionCheck(
                permission=perm.name,
                grant=grant,
                scope=perm.scope,
            ))
        return results

    def check_all_allowed(self, checks: list[PermissionCheck]) -> bool:
        """Return True if all permissions are granted (no denials)."""
        return all(c.grant != PermissionGrant.DENIED for c in checks)

    def has_denials(self, checks: list[PermissionCheck]) -> list[PermissionCheck]:
        """Return list of denied permissions."""
        return [c for c in checks if c.grant == PermissionGrant.DENIED]

    def _evaluate(
        self, perm: SkillPermission, ctx: PermissionContext, risk: RiskLevel,
    ) -> PermissionGrant:
        """Evaluate a single permission."""
        name = perm.name

        # Secrets access always requires explicit approval
        if name == "secrets_access":
            return PermissionGrant.REQUIRES_APPROVAL

        # Safe mode blocks most permissions
        if ctx.safety_mode == "safe":
            if name in ("file_write", "terminal", "network", "cua", "external_service"):
                return PermissionGrant.DENIED
            return PermissionGrant.GRANTED

        # High risk skills require approval for write/cua/terminal
        if risk in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            if name in ("file_write", "terminal", "cua"):
                if ctx.user_approval_mode == "never":
                    return PermissionGrant.DENIED
                if ctx.user_approval_mode == "inline":
                    return PermissionGrant.REQUIRES_APPROVAL
                return PermissionGrant.GRANTED

        # CUDA/TensorRT checker — terminal is fine for operator skills
        if name == "terminal" and perm.scope == "limited":
            return PermissionGrant.GRANTED

        # File read, memory read are always granted for first-party
        if name in ("file_read", "memory_read"):
            return PermissionGrant.GRANTED

        # CUA on verifier-only hardware is denied
        if name == "cua":
            if "verifier" in ctx.hardware_profile_id:
                return PermissionGrant.REQUIRES_APPROVAL
            return PermissionGrant.GRANTED

        # Network: granted unless safe mode
        if name in ("network", "external_service"):
            return PermissionGrant.GRANTED if ctx.safety_mode != "safe" else PermissionGrant.DENIED

        # Background jobs: require approval only in restrictive mode
        if name == "background_job":
            if ctx.user_approval_mode == "always":
                return PermissionGrant.REQUIRES_APPROVAL
            return PermissionGrant.GRANTED

        # Default: grant for first-party skills
        return PermissionGrant.GRANTED
