from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

_ALLOWED_SCHEMES = frozenset(["http", "https"])


def _validate_ollama_url(url: str) -> None:
    """Validate Ollama host URL: only http/https schemes allowed.

    SSRF mitigation for bandit B310 / CWE-22. The host is config-supplied
    and defaults to loopback; we enforce the scheme allowlist to prevent
    file://, gopher://, ftp:// etc.
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"malformed ollama URL: {url!r}") from exc
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"ollama URL scheme {parsed.scheme!r} not allowed; must be http or https"
        )
    if not parsed.hostname:
        raise ValueError(f"ollama URL missing host: {url!r}")

__all__ = [
    "OllamaConfig",
    "OllamaAdapter",
    "OLLAMA_REGISTRY",
]


@dataclass
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "llama3"
    timeout_s: float = 60.0
    stream: bool = False


class OllamaAdapter:
    def __init__(self, config: OllamaConfig | None = None) -> None:
        self._config = config if config is not None else OllamaConfig()

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        cfg = self._config
        body = json.dumps({
            "model": cfg.model,
            "prompt": prompt,
            "options": {"num_predict": max_tokens, "temperature": temperature},
            "stream": False,
        }).encode()
        generate_url = f"{cfg.host}/api/generate"
        _validate_ollama_url(generate_url)
        req = urllib.request.Request(
            generate_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:  # nosec B310 - scheme validated by _validate_ollama_url
                data = json.loads(resp.read())
            return data["response"]
        except Exception as exc:
            raise RuntimeError(f"Ollama unavailable: {exc}") from exc

    def list_models(self) -> list[str]:
        cfg = self._config
        tags_url = f"{cfg.host}/api/tags"
        _validate_ollama_url(tags_url)
        req = urllib.request.Request(tags_url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:  # nosec B310 - scheme validated by _validate_ollama_url
                data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            raise RuntimeError(f"Ollama unavailable: {exc}") from exc

    def is_available(self) -> bool:
        cfg = self._config
        tags_url = f"{cfg.host}/api/tags"
        try:
            _validate_ollama_url(tags_url)
        except ValueError:
            return False
        req = urllib.request.Request(tags_url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:  # nosec B310 - scheme validated by _validate_ollama_url
                return resp.status == 200
        except Exception:
            return False


OLLAMA_REGISTRY: dict[str, type[OllamaAdapter]] = {"ollama": OllamaAdapter}
