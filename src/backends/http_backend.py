from __future__ import annotations

import ipaddress
import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

_ALLOWED_SCHEMES = frozenset(["http", "https"])
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("fc00::/7"),
]


def _is_private_or_reserved(host: str) -> bool:
    if host.lower() == "localhost":
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        try:
            ip = ipaddress.ip_address(socket.gethostbyname(host))
        except (socket.gaierror, ValueError):
            return True
    if ip.is_loopback:
        return False
    return ip.is_private or ip.is_reserved or ip.is_link_local or ip.is_unspecified


def _validate_backend_url(url: str) -> None:
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise ValueError(f"malformed backend URL: {url!r}") from exc
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"backend URL scheme {parsed.scheme!r} not allowed; must be http or https")
    if not parsed.hostname:
        raise ValueError(f"backend URL missing host: {url!r}")
    if os.environ.get("AURELIUS_ALLOW_PRIVATE_URLS") == "1":
        return
    if _is_private_or_reserved(parsed.hostname):
        raise ValueError(
            f"backend URL host {parsed.hostname!r} resolves to a private/reserved address"
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
        _validate_backend_url(url)
        encoded = json.dumps(body).encode()
        req = urllib.request.Request(url, data=encoded, headers=self._headers(), method="POST")  # noqa: S310
        last_exc: Exception | None = None
        for attempt in range(cfg.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:  # nosec B310 - scheme validated by _validate_backend_url  # noqa: S310
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    raise RuntimeError(f"Client error {exc.code}: {exc.reason}") from exc
                last_exc = exc
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
                last_exc = exc
                if attempt < cfg.max_retries - 1:
                    sleep_s = min(2.0**attempt + 0.1, 8.0)
                    logging.getLogger(__name__).debug(
                        "Request attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt + 1,
                        cfg.max_retries,
                        exc.__class__.__name__,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
            except Exception as exc:
                last_exc = exc
        raise RuntimeError(
            f"HTTP backend request failed after {cfg.max_retries} retries: {last_exc}"
        ) from last_exc

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
        req = urllib.request.Request(health_url, method="GET")  # noqa: S310
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:  # nosec B310 - scheme validated by _validate_backend_url  # noqa: S310
                return resp.status == 200
        except Exception as exc:
            logging.getLogger(__name__).debug("Health check failed for %s: %s", health_url, exc)
            return False


HTTP_BACKEND_REGISTRY: dict[str, type[HTTPBackend]] = {"http": HTTPBackend}
