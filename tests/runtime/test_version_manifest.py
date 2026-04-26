"""Tests for version manifest."""

from __future__ import annotations

from src.runtime.version_manifest import VersionManifest


class TestVersionManifest:
    def test_detect(self):
        vm = VersionManifest.detect()
        assert vm.python_version.startswith("3.")

    def test_to_dict(self):
        vm = VersionManifest()
        d = vm.to_dict()
        assert "python" in d
        assert "aurelius" in d
