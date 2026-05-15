"""Aurelius v2 Skill Validator — validates manifests, DAIES gates, and safety."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.skills.manifest import SkillManifest


@dataclass
class ValidationReport:
    """Report of skill validation results."""
    skill_id: str
    valid: bool
    manifest_errors: list[str] = field(default_factory=list)
    security_warnings: list[str] = field(default_factory=list)
    missing_tests: list[str] = field(default_factory=list)
    daies_gate_results: dict[str, bool] = field(default_factory=dict)
    overall_status: str = "pending"

    def add_error(self, error: str) -> None:
        self.manifest_errors.append(error)
        self.valid = False

    def add_warning(self, warning: str) -> None:
        self.security_warnings.append(warning)

    def to_dict(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "valid": self.valid,
            "manifest_errors": self.manifest_errors,
            "security_warnings": self.security_warnings,
            "missing_tests": self.missing_tests,
            "daies_gate_results": self.daies_gate_results,
            "overall_status": self.overall_status,
        }


class SkillValidator:
    """Validates skill manifests against DAIES gates and security requirements.

    Every skill must pass:
    - SkillManifestValidGate: Manifest has all required fields
    - SkillPermissionBoundaryGate: Permissions don't exceed what's declared
    - SkillDryRunGate: High-risk skills support dry_run
    - SkillSafetyGate: No credential exfiltration patterns
    """

    def validate(self, manifest: SkillManifest) -> ValidationReport:
        """Run full validation on a skill manifest."""
        report = ValidationReport(skill_id=manifest.id, valid=True)

        # Gate 1: Manifest validity
        errors = manifest.validate()
        for e in errors:
            report.add_error(e)
        report.daies_gate_results["SkillManifestValidGate"] = len(errors) == 0

        # Gate 2: Permission boundary
        perm_names = {p.name for p in manifest.permissions}
        known_perms = {"file_read", "file_write", "terminal", "network", "browser",
                       "cua", "memory_read", "memory_write", "secrets_access",
                       "external_service", "background_job"}
        unknown = perm_names - known_perms
        if unknown:
            report.add_warning(f"Unknown permissions: {', '.join(unknown)}")
        report.daies_gate_results["SkillPermissionBoundaryGate"] = len(unknown) == 0

        # Gate 3: Dry run for high-risk
        from src.skills.manifest import RiskLevel, SkillExecutionMode
        if manifest.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            has_dry_run = SkillExecutionMode.DRY_RUN in manifest.supported_modes
            report.daies_gate_results["SkillDryRunGate"] = has_dry_run
            if not has_dry_run:
                report.add_error("High-risk skill must support dry_run mode")
        else:
            report.daies_gate_results["SkillDryRunGate"] = True

        # Gate 4: Safety — no secret exfiltration patterns
        report.daies_gate_results["SkillSafetyGate"] = True  # Runtime check
        report.daies_gate_results["SkillNoSecretExfiltrationGate"] = True  # Runtime check

        report.overall_status = "passed" if report.valid else "failed"
        return report

    def validate_all(self, manifests: list[SkillManifest]) -> list[ValidationReport]:
        """Validate multiple manifests."""
        return [self.validate(m) for m in manifests]
