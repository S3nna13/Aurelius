"""Tests for deployment drainer."""
from __future__ import annotations

import pytest

from src.deployment.drainer import DeploymentDrainer


class TestDeploymentDrainer:
    def test_track_and_finish(self):
        dd = DeploymentDrainer()
        dd.track("conn1")
        assert dd.active_count() == 1
        dd.finish("conn1")
        assert dd.active_count() == 0

    def test_finish_unknown_no_error(self):
        dd = DeploymentDrainer()
        dd.finish("unknown")  # should not raise

    def test_drain_returns_true_when_empty(self):
        dd = DeploymentDrainer()
        assert dd.drain() is True