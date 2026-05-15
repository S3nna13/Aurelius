"""Tests for Aurelius v2 Runtime foundation modules."""
import unittest

from src.runtime.capability_report import CapabilityMode, CapabilityReport
from src.runtime.memory_budget import MemoryBudgetConfig, MemoryBudgetManager, PressureLevel
from src.skills.manifest import RiskLevel, SkillExecutionMode, SkillManifest, SkillPermission
from src.skills.permissions import PermissionContext, PermissionGate, PermissionGrant


class TestMemoryBudget(unittest.TestCase):
    def test_low_pressure(self):
        config = MemoryBudgetConfig(total_memory_gb=32.0, reserved_for_os_gb=4.0)
        mgr = MemoryBudgetManager(config)
        mgr.update_consumer("weights_gb", 3.0)
        mgr.update_consumer("kv_cache_gb", 2.0)
        report = mgr.generate_report()
        self.assertEqual(report.pressure_level, PressureLevel.LOW)
        self.assertTrue(report.free_gb > 0)

    def test_critical_pressure_triggers_degradation(self):
        config = MemoryBudgetConfig(total_memory_gb=8.0, reserved_for_os_gb=2.0)
        mgr = MemoryBudgetManager(config)
        mgr.update_consumer("weights_gb", 5.0)
        mgr.update_consumer("kv_cache_gb", 1.5)
        mgr.update_consumer("runtime_memory_gb", 1.0)
        mgr.update_consumer("vision_gb", 1.0)
        self.assertTrue(mgr.should_degrade())
        step = mgr.apply_next_degradation()
        self.assertIsNotNone(step)
        self.assertEqual(step.name, "unload_inactive_skills")

class TestCapabilityReport(unittest.TestCase):
    def test_full_local_report(self):
        report = CapabilityReport.create_full_local(model="forge", backend="mlx", artifact="forge-q4-mlx", context=32768, hardware="mac_silicon_32gb")
        self.assertTrue(report.is_live())

    def test_silent_fallback_detection(self):
        report = CapabilityReport(requested_model="atlas", actual_model="swift", execution_mode=CapabilityMode.VERIFIER_ONLY, backend="remote_aurelius", artifact="swift-q3-gguf", quantization="q3", context_budget=4096, local_or_remote="remote", hardware_profile="jetson_nano_4gb")
        self.assertTrue(report.has_silent_fallback())

class TestSkillManifest(unittest.TestCase):
    def test_valid_manifest(self):
        m = SkillManifest(id="coding.test", name="Test", version="1.0.0", category="coding", summary="test", entrypoint="skills.builtin.coding.test:run")
        self.assertEqual(m.validate(), [])

    def test_high_risk_must_support_dry_run(self):
        m = SkillManifest(id="security.audit", name="Audit", version="1.0.0", category="security", summary="test", risk_level=RiskLevel.CRITICAL, supported_modes=[SkillExecutionMode.EXECUTE])
        errors = m.validate()
        self.assertTrue(any("dry_run" in e for e in errors))

class TestPermissionGate(unittest.TestCase):
    def test_safe_mode_blocks_writes(self):
        gate = PermissionGate()
        m = SkillManifest(id="test.s", name="Test", version="1.0", category="test", summary="t", permissions=[SkillPermission(name="file_write")], risk_level=RiskLevel.LOW)
        checks = gate.check(m, PermissionContext(safety_mode="safe"))
        self.assertTrue(any(c.grant == PermissionGrant.DENIED for c in checks))

    def test_secrets_require_approval(self):
        gate = PermissionGate()
        m = SkillManifest(id="test.s", name="Test", version="1.0", category="test", summary="t", permissions=[SkillPermission(name="secrets_access")], risk_level=RiskLevel.LOW)
        checks = gate.check(m, PermissionContext())
        self.assertTrue(any(c.grant == PermissionGrant.REQUIRES_APPROVAL for c in checks))

if __name__ == "__main__":
    unittest.main()
