"""Tests for the TGI backend adapter."""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from src.backends.tgi_backend_adapter import (
    TGI_BACKEND_REGISTRY,
    TGIBackendAdapter,
    TGIBackendAdapterError,
)


def _make_response(data: dict, status: int = 200):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_sse_stream(lines: list[str]):
    """Return a mock response that yields SSE lines on iteration."""
    resp = MagicMock()
    resp.status = 200
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    encoded = [line.encode("utf-8") for line in lines]
    resp.__iter__ = lambda s: iter(encoded)
    return resp


class TestTGIBackendAdapterInit:
    def test_init_accepts_valid_http_url(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "model")
        assert adapter._base_url == "http://localhost:8080"
        assert adapter._model_name == "model"

    def test_init_accepts_valid_https_url(self):
        adapter = TGIBackendAdapter("https://tgi.example.com", "model")
        assert adapter._base_url == "https://tgi.example.com"

    def test_init_strips_trailing_slash(self):
        adapter = TGIBackendAdapter("http://localhost:8080/", "model")
        assert adapter._base_url == "http://localhost:8080"

    def test_init_rejects_file_url(self):
        with pytest.raises(ValueError, match="not allowed"):
            TGIBackendAdapter("file:///etc/passwd", "model")

    def test_init_rejects_ftp_url(self):
        with pytest.raises(ValueError, match="not allowed"):
            TGIBackendAdapter("ftp://example.com", "model")

    def test_init_rejects_url_without_scheme(self):
        with pytest.raises(ValueError):
            TGIBackendAdapter("localhost:8080", "model")

    def test_init_rejects_empty_url(self):
        with pytest.raises(ValueError):
            TGIBackendAdapter("", "model")


class TestTGIBackendAdapterBuildPayload:
    def test_build_payload_default(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "model")
        payload = adapter._build_payload("hello", 64, 0.7)
        assert payload["inputs"] == "hello"
        assert payload["parameters"]["max_new_tokens"] == 64
        assert payload["parameters"]["temperature"] == 0.7
        assert "stream" not in payload

    def test_build_payload_stream(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "model")
        payload = adapter._build_payload("hello", 128, 0.5, stream=True)
        assert payload["stream"] is True


class TestTGIBackendAdapterHealthCheck:
    def test_health_check_true_on_200(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "model")
        mock_resp = _make_response({}, status=200)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert adapter.health_check() is True

    def test_health_check_false_on_network_error(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "model")
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert adapter.health_check() is False

    def test_health_check_false_on_http_error(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "model")
        err = urllib.error.HTTPError(
            "http://localhost:8080/health", 503, "Service Unavailable", None, None
        )
        with patch("urllib.request.urlopen", side_effect=err):
            assert adapter.health_check() is False


class TestTGIBackendAdapterGenerate:
    def test_generate_returns_generated_text(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "model")
        mock_resp = _make_response({"generated_text": "hello world"})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = adapter.generate("say hi")
        assert result == "hello world"

    def test_generate_sends_correct_payload(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "mistral")
        mock_resp = _make_response({"generated_text": "ok"})
        captured = []

        def fake_urlopen(req, timeout=None):
            captured.append(json.loads(req.data))
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            adapter.generate("test", max_tokens=128, temperature=0.5)

        body = captured[0]
        assert body["inputs"] == "test"
        assert body["parameters"]["max_new_tokens"] == 128
        assert body["parameters"]["temperature"] == 0.5

    def test_generate_raises_on_http_error(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "model")
        err = urllib.error.HTTPError(
            "http://localhost:8080/generate",
            500,
            "Internal Server Error",
            None,
            None,
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(TGIBackendAdapterError, match="HTTP error 500"):
                adapter.generate("hello")

    def test_generate_raises_on_invalid_json(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "model")
        resp = MagicMock()
        resp.status = 200
        resp.read.return_value = b"not json"
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=resp):
            with pytest.raises(TGIBackendAdapterError, match="invalid JSON"):
                adapter.generate("hello")

    def test_generate_raises_on_missing_generated_text(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "model")
        mock_resp = _make_response({"other_field": "value"})
        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(TGIBackendAdapterError, match="missing 'generated_text'"):
                adapter.generate("hello")


class TestTGIBackendAdapterStreamGenerate:
    def test_stream_generate_yields_tokens(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "model")
        sse_lines = [
            'data: {"token": {"text": " Hello"}}',
            'data: {"token": {"text": " world"}}',
            'data: {"generated_text": " Hello world"}',
        ]
        mock_resp = _make_sse_stream(sse_lines)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            tokens = list(adapter.stream_generate("say hi"))
        assert tokens == [" Hello", " world"]

    def test_stream_generate_skips_malformed_lines(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "model")
        sse_lines = [
            "event: ping",
            "data: not json",
            'data: {"token": {"text": "ok"}}',
        ]
        mock_resp = _make_sse_stream(sse_lines)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            tokens = list(adapter.stream_generate("test"))
        assert tokens == ["ok"]

    def test_stream_generate_handles_done_marker(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "model")
        sse_lines = [
            'data: {"token": {"text": "a"}}',
            "data: [DONE]",
            'data: {"token": {"text": "b"}}',
        ]
        mock_resp = _make_sse_stream(sse_lines)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            tokens = list(adapter.stream_generate("test"))
        assert tokens == ["a"]

    def test_stream_generate_raises_on_http_error(self):
        adapter = TGIBackendAdapter("http://localhost:8080", "model")
        err = urllib.error.HTTPError(
            "http://localhost:8080/generate_stream",
            503,
            "Unavailable",
            None,
            None,
        )
        with patch("urllib.request.urlopen", side_effect=err):
            with pytest.raises(TGIBackendAdapterError, match="HTTP error 503"):
                list(adapter.stream_generate("hello"))


class TestTGIBackendAdapterRegistry:
    def test_registry_has_tgi_key(self):
        assert "tgi" in TGI_BACKEND_REGISTRY

    def test_registry_value_is_tgi_backend_adapter_class(self):
        assert TGI_BACKEND_REGISTRY["tgi"] is TGIBackendAdapter
