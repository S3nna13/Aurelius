"""
test_rollback_manager.py — Tests for src/deployment/rollback_manager.py
Aurelius LLM Project — stdlib only.
"""

import dataclasses
import unittest

from src.deployment.rollback_manager import (
    REGISTRY,
    ROLLBACK_MANAGER_REGISTRY,
    DeploymentSnapshot,
    RollbackManager,
    RollbackReason,
)


class TestRollbackReasonEnum(unittest.TestCase):
    def test_manual_value(self):
        self.assertEqual(RollbackReason.MANUAL.value, "manual")

    def test_auto_health_fail_value(self):
        self.assertEqual(RollbackReason.AUTO_HEALTH_FAIL.value, "auto_health_fail")

    def test_auto_error_rate_value(self):
        self.assertEqual(RollbackReason.AUTO_ERROR_RATE.value, "auto_error_rate")

    def test_auto_latency_value(self):
        self.assertEqual(RollbackReason.AUTO_LATENCY.value, "auto_latency")

    def test_canary_fail_value(self):
        self.assertEqual(RollbackReason.CANARY_FAIL.value, "canary_fail")

    def test_five_members(self):
        self.assertEqual(len(RollbackReason), 5)


class TestDeploymentSnapshot(unittest.TestCase):
    def test_fields(self):
        snap = DeploymentSnapshot(
            snapshot_id="abc12345",
            version="v1.2.3",
            config={"key": "value"},
            created_at=1.0,
        )
        self.assertEqual(snap.snapshot_id, "abc12345")
        self.assertEqual(snap.version, "v1.2.3")
        self.assertEqual(snap.config["key"], "value")
        self.assertEqual(snap.created_at, 1.0)

    def test_frozen(self):
        snap = DeploymentSnapshot(
            snapshot_id="abc12345",
            version="v1",
            config={},
            created_at=0.0,
        )
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            snap.version = "v2"  # type: ignore[misc]


class TestRollbackManagerSnapshot(unittest.TestCase):
    def test_snapshot_returns_deployment_snapshot(self):
        mgr = RollbackManager()
        snap = mgr.snapshot("v1", {"a": 1})
        self.assertIsInstance(snap, DeploymentSnapshot)

    def test_snapshot_id_auto_assigned(self):
        mgr = RollbackManager()
        snap = mgr.snapshot("v1", {})
        self.assertIsNotNone(snap.snapshot_id)
        self.assertEqual(len(snap.snapshot_id), 8)

    def test_snapshot_id_unique(self):
        mgr = RollbackManager()
        s1 = mgr.snapshot("v1", {})
        s2 = mgr.snapshot("v2", {})
        self.assertNotEqual(s1.snapshot_id, s2.snapshot_id)

    def test_snapshot_version_stored(self):
        mgr = RollbackManager()
        snap = mgr.snapshot("v3.0.0", {"model": "aurelius"})
        self.assertEqual(snap.version, "v3.0.0")

    def test_snapshot_config_stored(self):
        mgr = RollbackManager()
        cfg = {"lr": 0.001, "layers": 48}
        snap = mgr.snapshot("v1", cfg)
        self.assertEqual(snap.config["lr"], 0.001)

    def test_snapshot_created_at_positive(self):
        mgr = RollbackManager()
        snap = mgr.snapshot("v1", {})
        self.assertGreater(snap.created_at, 0)

    def test_snapshot_increments_len(self):
        mgr = RollbackManager()
        mgr.snapshot("v1", {})
        mgr.snapshot("v2", {})
        self.assertEqual(len(mgr), 2)


class TestRollbackManagerMaxSnapshots(unittest.TestCase):
    def test_max_snapshots_eviction(self):
        mgr = RollbackManager(max_snapshots=3)
        s1 = mgr.snapshot("v1", {})
        mgr.snapshot("v2", {})
        mgr.snapshot("v3", {})
        mgr.snapshot("v4", {})
        # v1 should have been evicted
        self.assertEqual(len(mgr), 3)
        ids = [s.snapshot_id for s in mgr.history()]
        self.assertNotIn(s1.snapshot_id, ids)

    def test_max_snapshots_len_capped(self):
        mgr = RollbackManager(max_snapshots=2)
        for i in range(5):
            mgr.snapshot(f"v{i}", {})
        self.assertEqual(len(mgr), 2)


class TestRollbackManagerRollbackTo(unittest.TestCase):
    def test_rollback_to_returns_dict(self):
        mgr = RollbackManager()
        snap = mgr.snapshot("v1", {})
        result = mgr.rollback_to(snap.snapshot_id, RollbackReason.MANUAL)
        self.assertIsInstance(result, dict)

    def test_rollback_to_snapshot_id_key(self):
        mgr = RollbackManager()
        snap = mgr.snapshot("v2", {})
        result = mgr.rollback_to(snap.snapshot_id, RollbackReason.MANUAL)
        self.assertEqual(result["snapshot_id"], snap.snapshot_id)

    def test_rollback_to_version(self):
        mgr = RollbackManager()
        snap = mgr.snapshot("v5.0", {})
        result = mgr.rollback_to(snap.snapshot_id, RollbackReason.AUTO_ERROR_RATE)
        self.assertEqual(result["version"], "v5.0")

    def test_rollback_to_reason(self):
        mgr = RollbackManager()
        snap = mgr.snapshot("v1", {})
        result = mgr.rollback_to(snap.snapshot_id, RollbackReason.CANARY_FAIL)
        self.assertEqual(result["reason"], "canary_fail")

    def test_rollback_to_rolled_back_at(self):
        mgr = RollbackManager()
        snap = mgr.snapshot("v1", {})
        result = mgr.rollback_to(snap.snapshot_id, RollbackReason.MANUAL)
        self.assertIn("rolled_back_at", result)
        self.assertIsInstance(result["rolled_back_at"], float)

    def test_rollback_to_missing_raises_key_error(self):
        mgr = RollbackManager()
        with self.assertRaises(KeyError):
            mgr.rollback_to("deadbeef", RollbackReason.MANUAL)


class TestRollbackManagerLatestHistory(unittest.TestCase):
    def test_latest_none_when_empty(self):
        mgr = RollbackManager()
        self.assertIsNone(mgr.latest())

    def test_latest_returns_newest(self):
        mgr = RollbackManager()
        mgr.snapshot("v1", {})
        s2 = mgr.snapshot("v2", {})
        self.assertEqual(mgr.latest().snapshot_id, s2.snapshot_id)

    def test_history_newest_first(self):
        mgr = RollbackManager()
        s1 = mgr.snapshot("v1", {})
        s2 = mgr.snapshot("v2", {})
        s3 = mgr.snapshot("v3", {})
        hist = mgr.history()
        self.assertEqual(hist[0].snapshot_id, s3.snapshot_id)
        self.assertEqual(hist[-1].snapshot_id, s1.snapshot_id)

    def test_history_returns_list(self):
        mgr = RollbackManager()
        self.assertIsInstance(mgr.history(), list)

    def test_len_empty(self):
        mgr = RollbackManager()
        self.assertEqual(len(mgr), 0)

    def test_len_after_snapshots(self):
        mgr = RollbackManager()
        for i in range(4):
            mgr.snapshot(f"v{i}", {})
        self.assertEqual(len(mgr), 4)


class TestRegistry(unittest.TestCase):
    def test_registry_key(self):
        self.assertIn("default", ROLLBACK_MANAGER_REGISTRY)

    def test_registry_value(self):
        self.assertIs(ROLLBACK_MANAGER_REGISTRY["default"], RollbackManager)

    def test_registry_alias(self):
        self.assertIs(REGISTRY, ROLLBACK_MANAGER_REGISTRY)


if __name__ == "__main__":
    unittest.main()
