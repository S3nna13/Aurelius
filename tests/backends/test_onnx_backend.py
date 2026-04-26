"""Tests for onnx_backend."""

from __future__ import annotations

from src.backends.onnx_backend import OnnxBackend


class TestOnnxBackend:
    def test_name(self):
        b = OnnxBackend()
        assert b.name == "onnx"

    def test_health(self):
        b = OnnxBackend()
        assert not b.health()

    def test_capabilities(self):
        c = OnnxBackend().capabilities()
        assert isinstance(c, list)
