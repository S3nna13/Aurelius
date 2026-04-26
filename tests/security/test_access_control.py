"""Tests for access control."""

from __future__ import annotations

import pytest

from src.security.access_control import AccessControl, Permission, Role


class TestAccessControl:
    def test_simple_check(self):
        ac = AccessControl()
        ac.add_role(Role("reader", [Permission("log", "read")]))
        ac.assign_role("alice", "reader")
        assert ac.check("alice", "log", "read") is True
        assert ac.check("alice", "log", "write") is False

    def test_unknown_role_raises(self):
        ac = AccessControl()
        with pytest.raises(ValueError, match="unknown role"):
            ac.assign_role("bob", "nonexistent")

    def test_unknown_user_no_permissions(self):
        ac = AccessControl()
        assert ac.check("stranger", "resource", "action") is False

    def test_revoke_role(self):
        ac = AccessControl()
        ac.add_role(Role("admin", [Permission("*", "admin")]))
        ac.assign_role("carol", "admin")
        assert ac.check("carol", "*", "admin") is True
        ac.revoke_role("carol", "admin")
        assert ac.check("carol", "*", "admin") is False

    def test_list_roles(self):
        ac = AccessControl()
        ac.add_role(Role("a"))
        ac.add_role(Role("b"))
        assert sorted(ac.list_roles()) == ["a", "b"]
