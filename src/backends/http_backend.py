from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.security.url_scheme_validator import validate_url

__all__ = [
    "HTTPBackendConfig",
    "HTTPBackend",
    "HTTP_BACKEND_REGISTRY",
]


@dataclass
class HTTPBackendConfig:
    base_url: str = "http://localhost:8000"
    api_key: str = ""
    model: str = "aurelius"
    timeout_s: float = 30.0
    max_retries: int = 3


class HTTPBackend:
    def __init__(self, config: HTTPBackendConfig | None = None) -> None:
        self._config = config if config is not None else HTTPBackendConfig()

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._config.api_key:
            h["Authorization"] = f"Bearer {self._config.api_key}"
        return h

    def _post(self, url: str, body: dict) -> dict:
        cfg = self._config
        # AUR-SEC-2026-0020: gate URL scheme before any urlopen call.
        validate_url(url)
        encoded = json.dumps(body).encode()
        req = urllib.request.Request(url, data=encoded, headers=self._headers(), method="POST")  # noqa: S310 — scheme validated above (AUR-SEC-2026-0020)
        last_exc: Exception | None = None
        for attempt in range(cfg.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:  # noqa: S310 — scheme validated above (AUR-SEC-2026-0020)
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    raise RuntimeError(f"Client error {exc.code}: {exc.reason}") from exc
                last_exc = exc
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(f"HTTP backend request failed after {cfg.max_retries} retries: {last_exc}") from last_exc

    def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.7) -> str:
        cfg = self._config
        url = f"{cfg.base_url}/v1/chat/completions"
        body = {
            "model": cfg.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data = self._post(url, body)
        return data["choices"][0]["message"]["content"]

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        cfg = self._config
        url = f"{cfg.base_url}/v1/completions"
        body = {"model": cfg.model, "prompt": prompt, "max_tokens": max_tokens}
        data = self._post(url, body)
        return data["choices"][0]["text"]

    def health(self) -> bool:
        cfg = self._config
        health_url = f"{cfg.base_url}/health"
        # AUR-SEC-2026-0020: block unsafe schemes before urlopen. health() is
        # intentionally forgiving (returns False on any failure) but must
        # still refuse to dispatch banned-scheme requests.
        try:
            validate_url(health_url)
        except Exception:
            return False
        req = urllib.request.Request(health_url, method="GET")  # noqa: S310 — scheme validated above (AUR-SEC-2026-0020)
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:  # noqa: S310 — scheme validated above (AUR-SEC-2026-0020)
                return resp.status == 200
        except Exception:
            return False


HTTP_BACKEND_REGISTRY: dict[str, type[HTTPBackend]] = {"http": HTTPBackend}
