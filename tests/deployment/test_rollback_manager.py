"""Tests for rollback manager."""
from __future__ import annotations

import json
import os
import tempfile
import time

import pytest

from src.deployment.rollback_manager import RollbackManager, ROLLBACK_MANAGER_REGISTRY


class TestRollbackManager:
    def test_snapshot_creates_file_and_returns_path(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = RollbackManager(td, max_revisions=5)
            path = mgr.snapshot("v1", {"key": "val"})
            assert os.path.isfile(path)
            assert path == os.path.join(td, "v1", "snapshot.json")
            with open(path) as f:
                assert json.load(f) == {"key": "val"}

    def test_list_revisions_sorted_newest_first(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = RollbackManager(td, max_revisions=5)
            mgr.snapshot("v1", {"n": 1})
            time.sleep(0.01)
            mgr.snapshot("v2", {"n": 2})
            time.sleep(0.01)
            mgr.snapshot("v3", {"n": 3})
            revs = mgr.list_revisions()
            assert revs == ["v3", "v2", "v1"]

    def test_get_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = RollbackManager(td, max_revisions=5)
            mgr.snapshot("v1", {"key": "value"})
            assert mgr.get_manifest("v1") == {"key": "value"}

    def test_rollback_returns_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = RollbackManager(td, max_revisions=5)
            mgr.snapshot("v2", {"model": "a"})
            manifest = mgr.rollback("v2")
            assert manifest == {"model": "a"}

    def test_rollback_missing_raises(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = RollbackManager(td, max_revisions=5)
            with pytest.raises(FileNotFoundError):
                mgr.rollback("missing")

    def test_prune_old_revisions(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = RollbackManager(td, max_revisions=2)
            mgr.snapshot("v1", {"n": 1})
            time.sleep(0.01)
            mgr.snapshot("v2", {"n": 2})
            time.sleep(0.01)
            mgr.snapshot("v3", {"n": 3})
            mgr.prune_old_revisions()
            revs = mgr.list_revisions()
            assert revs == ["v3", "v2"]
            assert not os.path.exists(os.path.join(td, "v1", "snapshot.json"))

    def test_invalid_version_empty(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = RollbackManager(td)
            with pytest.raises(ValueError):
                mgr.snapshot("", {})

    def test_invalid_version_too_long(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = RollbackManager(td)
            with pytest.raises(ValueError):
                mgr.snapshot("x" * 65, {})

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            mgr = RollbackManager(td)
            with pytest.raises(ValueError):
                mgr.snapshot("../../../etc/passwd", {})
            with pytest.raises(ValueError):
                mgr.snapshot("foo..bar", {})


class TestRegistry:
    def test_rollback_manager_registry(self):
        assert ROLLBACK_MANAGER_REGISTRY["default"] is RollbackManager
