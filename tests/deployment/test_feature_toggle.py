"""
test_feature_toggle.py — Tests for src/deployment/feature_toggle.py
Aurelius LLM Project — stdlib only.
"""

import hashlib
import unittest

from src.deployment.feature_toggle import (
    FEATURE_TOGGLE_REGISTRY,
    REGISTRY,
    FeatureToggle,
    FeatureToggleManager,
    ToggleState,
)


class TestToggleStateEnum(unittest.TestCase):
    def test_enabled_value(self):
        self.assertEqual(ToggleState.ENABLED.value, "enabled")

    def test_disabled_value(self):
        self.assertEqual(ToggleState.DISABLED.value, "disabled")

    def test_percentage_value(self):
        self.assertEqual(ToggleState.PERCENTAGE.value, "percentage")

    def test_allowlist_value(self):
        self.assertEqual(ToggleState.ALLOWLIST.value, "allowlist")

    def test_four_members(self):
        self.assertEqual(len(ToggleState), 4)


class TestFeatureToggleDataclass(unittest.TestCase):
    def test_defaults(self):
        ft = FeatureToggle(name="x", state=ToggleState.ENABLED)
        self.assertEqual(ft.percentage, 0.0)
        self.assertEqual(ft.allowlist, [])
        self.assertEqual(ft.metadata, {})

    def test_stores_name_state(self):
        ft = FeatureToggle(name="flag", state=ToggleState.DISABLED)
        self.assertEqual(ft.name, "flag")
        self.assertEqual(ft.state, ToggleState.DISABLED)

    def test_mutable_allowlist_independent(self):
        ft1 = FeatureToggle(name="a", state=ToggleState.ALLOWLIST)
        ft2 = FeatureToggle(name="b", state=ToggleState.ALLOWLIST)
        ft1.allowlist.append("user1")
        self.assertNotIn("user1", ft2.allowlist)


class TestFeatureToggleManagerEnabled(unittest.TestCase):
    def setUp(self):
        self.mgr = FeatureToggleManager()
        self.mgr.register(FeatureToggle(name="flag_on", state=ToggleState.ENABLED))

    def test_enabled_no_user(self):
        self.assertTrue(self.mgr.is_enabled("flag_on"))

    def test_enabled_with_user(self):
        self.assertTrue(self.mgr.is_enabled("flag_on", user_id="alice"))

    def test_enabled_with_empty_user(self):
        self.assertTrue(self.mgr.is_enabled("flag_on", user_id=""))


class TestFeatureToggleManagerDisabled(unittest.TestCase):
    def setUp(self):
        self.mgr = FeatureToggleManager()
        self.mgr.register(FeatureToggle(name="flag_off", state=ToggleState.DISABLED))

    def test_disabled_no_user(self):
        self.assertFalse(self.mgr.is_enabled("flag_off"))

    def test_disabled_with_user(self):
        self.assertFalse(self.mgr.is_enabled("flag_off", user_id="bob"))


class TestFeatureToggleManagerPercentage(unittest.TestCase):
    def _bucket(self, name: str, user_id: str) -> int:
        return int(hashlib.md5((name + user_id).encode()).hexdigest(), 16) % 100

    def test_percentage_deterministic(self):
        mgr = FeatureToggleManager()
        mgr.register(
            FeatureToggle(name="pct", state=ToggleState.PERCENTAGE, percentage=50.0)
        )
        result1 = mgr.is_enabled("pct", "userA")
        result2 = mgr.is_enabled("pct", "userA")
        self.assertEqual(result1, result2)

    def test_percentage_zero_always_false(self):
        mgr = FeatureToggleManager()
        mgr.register(
            FeatureToggle(name="p0", state=ToggleState.PERCENTAGE, percentage=0.0)
        )
        for uid in ["u1", "u2", "u3", "u4"]:
            self.assertFalse(mgr.is_enabled("p0", uid))

    def test_percentage_100_always_true(self):
        mgr = FeatureToggleManager()
        mgr.register(
            FeatureToggle(name="p100", state=ToggleState.PERCENTAGE, percentage=100.0)
        )
        for uid in ["u1", "u2", "u3", "u4"]:
            self.assertTrue(mgr.is_enabled("p100", uid))

    def test_percentage_bucket_boundary(self):
        """Manually verify one user against the hash formula."""
        mgr = FeatureToggleManager()
        pct = 50.0
        mgr.register(
            FeatureToggle(name="btest", state=ToggleState.PERCENTAGE, percentage=pct)
        )
        uid = "boundary_user"
        bucket = self._bucket("btest", uid)
        expected = bucket < pct
        self.assertEqual(mgr.is_enabled("btest", uid), expected)


class TestFeatureToggleManagerAllowlist(unittest.TestCase):
    def setUp(self):
        self.mgr = FeatureToggleManager()
        ft = FeatureToggle(
            name="allow",
            state=ToggleState.ALLOWLIST,
            allowlist=["alice", "bob"],
        )
        self.mgr.register(ft)

    def test_allowlist_user_in_list(self):
        self.assertTrue(self.mgr.is_enabled("allow", "alice"))

    def test_allowlist_another_user_in_list(self):
        self.assertTrue(self.mgr.is_enabled("allow", "bob"))

    def test_allowlist_user_not_in_list(self):
        self.assertFalse(self.mgr.is_enabled("allow", "charlie"))

    def test_allowlist_empty_user_not_in_list(self):
        self.assertFalse(self.mgr.is_enabled("allow", ""))


class TestFeatureToggleManagerRegister(unittest.TestCase):
    def test_register_duplicate_raises_value_error(self):
        mgr = FeatureToggleManager()
        mgr.register(FeatureToggle(name="dup", state=ToggleState.ENABLED))
        with self.assertRaises(ValueError):
            mgr.register(FeatureToggle(name="dup", state=ToggleState.DISABLED))


class TestFeatureToggleManagerOverride(unittest.TestCase):
    def test_override_changes_state(self):
        mgr = FeatureToggleManager()
        mgr.register(FeatureToggle(name="tog", state=ToggleState.ENABLED))
        result = mgr.override("tog", ToggleState.DISABLED)
        self.assertTrue(result)
        self.assertFalse(mgr.is_enabled("tog"))

    def test_override_missing_returns_false(self):
        mgr = FeatureToggleManager()
        result = mgr.override("nonexistent", ToggleState.ENABLED)
        self.assertFalse(result)


class TestFeatureToggleManagerListGet(unittest.TestCase):
    def test_list_toggles_sorted(self):
        mgr = FeatureToggleManager()
        mgr.register(FeatureToggle(name="z_flag", state=ToggleState.ENABLED))
        mgr.register(FeatureToggle(name="a_flag", state=ToggleState.DISABLED))
        mgr.register(FeatureToggle(name="m_flag", state=ToggleState.ENABLED))
        self.assertEqual(mgr.list_toggles(), ["a_flag", "m_flag", "z_flag"])

    def test_list_toggles_empty(self):
        mgr = FeatureToggleManager()
        self.assertEqual(mgr.list_toggles(), [])

    def test_get_existing(self):
        mgr = FeatureToggleManager()
        ft = FeatureToggle(name="found", state=ToggleState.ENABLED)
        mgr.register(ft)
        self.assertIs(mgr.get("found"), ft)

    def test_get_missing_returns_none(self):
        mgr = FeatureToggleManager()
        self.assertIsNone(mgr.get("ghost"))

    def test_is_enabled_missing_name_returns_false(self):
        mgr = FeatureToggleManager()
        self.assertFalse(mgr.is_enabled("no_such_flag", "user"))


class TestRegistry(unittest.TestCase):
    def test_feature_toggle_registry_key(self):
        self.assertIn("default", FEATURE_TOGGLE_REGISTRY)

    def test_feature_toggle_registry_value(self):
        self.assertIs(FEATURE_TOGGLE_REGISTRY["default"], FeatureToggleManager)

    def test_registry_alias(self):
        self.assertIs(REGISTRY, FEATURE_TOGGLE_REGISTRY)


if __name__ == "__main__":
    unittest.main()
