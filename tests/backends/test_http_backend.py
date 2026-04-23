from __future__ import annotations

import json
import unittest
import urllib.error
from unittest.mock import MagicMock, patch, call

from src.backends.http_backend import HTTPBackend, HTTPBackendConfig, HTTP_BACKEND_REGISTRY


def _make_response(data: dict, status: int = 200):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(data).encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_http_error(code: int, reason: str = "Error"):
    err = urllib.error.HTTPError(url="http://x", code=code, msg=reason, hdrs=None, fp=None)
    return err


class TestHTTPBackendConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = HTTPBackendConfig()
        assert cfg.base_url == "http://localhost:8000"
        assert cfg.api_key == ""
        assert cfg.model == "aurelius"
        assert cfg.timeout_s == 30.0
        assert cfg.max_retries == 3


class TestHTTPBackendChat(unittest.TestCase):
    def test_chat_parses_choices_message_content(self):
        backend = HTTPBackend()
        data = {"choices": [{"message": {"content": "hello"}}]}
        mock_resp = _make_response(data)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = backend.chat([{"role": "user", "content": "hi"}])
        assert result == "hello"

    def test_chat_sends_auth_header_when_api_key_set(self):
        cfg = HTTPBackendConfig(api_key="sk-test")
        backend = HTTPBackend(cfg)
        data = {"choices": [{"message": {"content": "ok"}}]}
        mock_resp = _make_response(data)
        captured_req = []

        def fake_urlopen(req, timeout=None):
            captured_req.append(req)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            backend.chat([{"role": "user", "content": "test"}])

        assert captured_req[0].get_header("Authorization") == "Bearer sk-test"

    def test_chat_no_auth_header_when_no_api_key(self):
        backend = HTTPBackend()
        data = {"choices": [{"message": {"content": "ok"}}]}
        mock_resp = _make_response(data)
        captured_req = []

        def fake_urlopen(req, timeout=None):
            captured_req.append(req)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            backend.chat([])

        assert captured_req[0].get_header("Authorization") is None

    def test_chat_retries_on_5xx(self):
        cfg = HTTPBackendConfig(max_retries=3)
        backend = HTTPBackend(cfg)
        call_count = [0]

        def fake_urlopen(req, timeout=None):
            call_count[0] += 1
            raise _make_http_error(503, "Service Unavailable")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(RuntimeError):
                backend.chat([])

        assert call_count[0] == 3

    def test_chat_raises_immediately_on_4xx(self):
        backend = HTTPBackend()
        call_count = [0]

        def fake_urlopen(req, timeout=None):
            call_count[0] += 1
            raise _make_http_error(400, "Bad Request")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with self.assertRaises(RuntimeError) as ctx:
                backend.chat([])

        assert call_count[0] == 1
        assert "Client error 400" in str(ctx.exception)

    def test_chat_succeeds_after_retry(self):
        cfg = HTTPBackendConfig(max_retries=3)
        backend = HTTPBackend(cfg)
        call_count = [0]
        data = {"choices": [{"message": {"content": "retried ok"}}]}
        mock_resp = _make_response(data)

        def fake_urlopen(req, timeout=None):
            call_count[0] += 1
            if call_count[0] < 2:
                raise _make_http_error(503)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = backend.chat([])

        assert result == "retried ok"
        assert call_count[0] == 2


class TestHTTPBackendComplete(unittest.TestCase):
    def test_complete_returns_choices_text(self):
        backend = HTTPBackend()
        data = {"choices": [{"text": "completed text"}]}
        mock_resp = _make_response(data)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = backend.complete("once upon a time")
        assert result == "completed text"


class TestHTTPBackendHealth(unittest.TestCase):
    def test_health_true_on_200(self):
        backend = HTTPBackend()
        mock_resp = _make_response({}, status=200)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            assert backend.health() is True

    def test_health_false_on_network_error(self):
        backend = HTTPBackend()
        with patch("urllib.request.urlopen", side_effect=OSError("down")):
            assert backend.health() is False

    def test_health_false_on_http_error(self):
        backend = HTTPBackend()
        with patch("urllib.request.urlopen", side_effect=_make_http_error(500)):
            assert backend.health() is False


class TestHTTPBackendRegistry(unittest.TestCase):
    def test_registry_has_http_key(self):
        assert "http" in HTTP_BACKEND_REGISTRY

    def test_registry_value_is_http_backend_class(self):
        assert HTTP_BACKEND_REGISTRY["http"] is HTTPBackend


if __name__ == "__main__":
    unittest.main()
