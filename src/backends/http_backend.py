from __future__ import annotations

import ipaddress
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

_ALLOWED_SCHEMES = frozenset(["http", "https"])

# SSRF: private/reserved IP ranges that should not be reachable via backend.
_BLOCKED_IP_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),   # Link-local
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),          # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),         # IPv6 unique local
    ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
]


def _validate_backend_url(url: str) -> None:
    """Validate backend URL: scheme allowlist + SSRF IP blocklist.

    Mitigates bandit B310 / CWE-918. Blocks private/reserved IPs and
    non-HTTP schemes.
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:  # pragma: no cover - urlparse rarely raises
        raise ValueError(f"malformed backend URL: {url!r}") from exc
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"backend URL scheme {parsed.scheme!r} not allowed; must be http or https"
        )
    if not parsed.hostname:
        raise ValueError(f"backend URL missing host: {url!r}")

    # SSRF: block private/reserved IPs
    hostname = parsed.hostname
    try:
        addr = ipaddress.ip_address(hostname)
    except ValueError:
        # Not an IP literal — allow (DNS resolution happens later at the
        # transport layer, which is acceptable for backend URLs that are
        # config-supplied rather than user-supplied).
        return
    for network in _BLOCKED_IP_NETWORKS:
        if addr in network:
            raise ValueError(
                f"backend URL resolves to a private/reserved IP ({hostname}); "
                "this is blocked to prevent SSRF."
            )

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
        encoded = json.dumps(body).encode()
        req = urllib.request.Request(url, data=encoded, headers=self._headers(), method="POST")
        last_exc: Exception | None = None
        for attempt in range(cfg.max_retries):
            try:
                _validate_backend_url(url)
                with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:  # nosec B310 - scheme validated by _validate_backend_url
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
        try:
            _validate_backend_url(health_url)
        except ValueError:
            return False
        req = urllib.request.Request(health_url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:  # nosec B310 - scheme validated by _validate_backend_url
                return resp.status == 200
        except Exception:
            return False


HTTP_BACKEND_REGISTRY: dict[str, type[HTTPBackend]] = {"http": HTTPBackend}
