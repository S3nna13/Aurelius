"""SSRF regression tests for HTTP + Ollama backends.

Finding AUR-SEC-2026-0020; CWE-918 (SSRF), CWE-284.

Asserts that :func:`urllib.request.urlopen` is *never* invoked with a banned
scheme: the validator must raise :class:`UnsafeURLSchemeError` before any
network call is made.
"""
from __future__ import annotations

import pytest

from src.backends.http_backend import HTTPBackend, HTTPBackendConfig
from src.backends.ollama_adapter import OllamaAdapter, OllamaConfig
from src.security.url_scheme_validator import UnsafeURLSchemeError


class _UrlopenSpy:
    """Records every urlopen invocation and fails hard on banned schemes."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, req, timeout=None):  # noqa: ANN001
        url = getattr(req, "full_url", req)
        self.calls.append(url)
        raise AssertionError(
            f"urlopen was invoked with {url!r} — validator should have blocked it"
        )


@pytest.fixture
def urlopen_spy(monkeypatch: pytest.MonkeyPatch) -> _UrlopenSpy:
    spy = _UrlopenSpy()
    # Patch both backend call sites.
    monkeypatch.setattr("src.backends.http_backend.urllib.request.urlopen", spy)
    monkeypatch.setattr("src.backends.ollama_adapter.urllib.request.urlopen", spy)
    return spy


class TestHTTPBackendSSRF:
    def test_chat_rejects_file_scheme(self, urlopen_spy: _UrlopenSpy) -> None:
        backend = HTTPBackend(HTTPBackendConfig(base_url="file:///etc"))
        with pytest.raises(UnsafeURLSchemeError):
            backend.chat([{"role": "user", "content": "hi"}])
        assert urlopen_spy.calls == []

    def test_complete_rejects_gopher_scheme(self, urlopen_spy: _UrlopenSpy) -> None:
        backend = HTTPBackend(HTTPBackendConfig(base_url="gopher://evil"))
        with pytest.raises(UnsafeURLSchemeError):
            backend.complete("prompt")
        assert urlopen_spy.calls == []

    def test_health_rejects_javascript_scheme(
        self, urlopen_spy: _UrlopenSpy
    ) -> None:
        backend = HTTPBackend(HTTPBackendConfig(base_url="javascript:alert(1)"))
        # health() currently swallows exceptions and returns False — but
        # crucially must not reach urlopen.
        result = backend.health()
        assert result is False
        assert urlopen_spy.calls == []

    def test_chat_rejects_ftp_scheme(self, urlopen_spy: _UrlopenSpy) -> None:
        backend = HTTPBackend(HTTPBackendConfig(base_url="ftp://example.com"))
        with pytest.raises(UnsafeURLSchemeError):
            backend.chat([{"role": "user", "content": "hi"}])
        assert urlopen_spy.calls == []


class TestOllamaAdapterSSRF:
    def test_generate_rejects_file_scheme(self, urlopen_spy: _UrlopenSpy) -> None:
        adapter = OllamaAdapter(OllamaConfig(host="file:///tmp"))
        with pytest.raises(UnsafeURLSchemeError):
            adapter.generate("hi")
        assert urlopen_spy.calls == []

    def test_list_models_rejects_data_scheme(
        self, urlopen_spy: _UrlopenSpy
    ) -> None:
        adapter = OllamaAdapter(OllamaConfig(host="data:text/plain,abc"))
        with pytest.raises(UnsafeURLSchemeError):
            adapter.list_models()
        assert urlopen_spy.calls == []

    def test_is_available_rejects_jar_scheme(
        self, urlopen_spy: _UrlopenSpy
    ) -> None:
        adapter = OllamaAdapter(OllamaConfig(host="jar:http://x/y.jar!/z"))
        # is_available swallows errors for ergonomics but must still block.
        assert adapter.is_available() is False
        assert urlopen_spy.calls == []
