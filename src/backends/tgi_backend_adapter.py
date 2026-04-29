"""HuggingFace Text Generation Inference (TGI) backend adapter for Aurelius.

Clean-room adapter that speaks to a TGI server over HTTP using only stdlib.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from urllib.parse import urlparse

from src.backends.base import BackendAdapterError

__all__ = [
    "TGIBackendAdapterError",
    "TGIBackendAdapter",
    "TGI_BACKEND_REGISTRY",
]

_ALLOWED_SCHEMES = frozenset(["http", "https"])


class TGIBackendAdapterError(BackendAdapterError):
    """Raised for any TGI backend adapter contract or runtime failure."""


def _validate_tgi_url(url: str) -> None:
    """Validate TGI base URL: only http/https schemes allowed.

    SSRF mitigation for bandit B310 / CWE-22. The base URL is config-supplied
    and typically points at a loopback inference server, but we still enforce
    the scheme allowlist to prevent file://, gopher://, ftp:// etc.
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise ValueError(f"malformed TGI URL: {url!r}") from exc
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"TGI URL scheme {parsed.scheme!r} not allowed; must be http or https")
    if not parsed.hostname:
        raise ValueError(f"TGI URL missing host: {url!r}")


class TGIBackendAdapter:
    """HTTP adapter for a HuggingFace Text Generation Inference server."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        timeout_s: float = 30.0,
    ) -> None:
        _validate_tgi_url(base_url)
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._timeout_s = timeout_s

    def _build_payload(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        *,
        stream: bool = False,
    ) -> dict:
        """Construct the JSON payload for a TGI generate request."""
        payload: dict = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
            },
        }
        if stream:
            payload["stream"] = True
        return payload

    def health_check(self) -> bool:
        """Return True iff the TGI server reports healthy."""
        health_url = f"{self._base_url}/health"
        try:
            _validate_tgi_url(health_url)
        except ValueError:
            return False
        req = urllib.request.Request(health_url, method="GET")  # noqa: S310  # nosec
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310  # nosec
                return resp.status == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.7,
    ) -> str:
        """Run a single generation via TGI and return the generated text."""
        url = f"{self._base_url}/generate"
        _validate_tgi_url(url)
        payload = self._build_payload(prompt, max_tokens, temperature)
        body = json.dumps(payload).encode()
        req = urllib.request.Request(  # noqa: S310  # nosec
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310  # nosec
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise TGIBackendAdapterError(f"TGI HTTP error {exc.code}: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise TGIBackendAdapterError(f"TGI returned invalid JSON: {exc}") from exc
        except Exception as exc:
            raise TGIBackendAdapterError(f"TGI request failed: {exc}") from exc

        if not isinstance(data, dict):
            raise TGIBackendAdapterError(
                f"TGI returned unexpected response type: {type(data).__name__}"
            )
        generated_text = data.get("generated_text")
        if generated_text is None:
            raise TGIBackendAdapterError("TGI response missing 'generated_text' field")
        return str(generated_text)

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Stream tokens from TGI via Server-Sent Events."""
        url = f"{self._base_url}/generate_stream"
        _validate_tgi_url(url)
        payload = self._build_payload(prompt, max_tokens, temperature, stream=True)
        body = json.dumps(payload).encode()
        req = urllib.request.Request(  # noqa: S310  # nosec
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310  # nosec
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    json_str = line[len("data:") :].strip()
                    if json_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    token = chunk.get("token")
                    if isinstance(token, dict):
                        text = token.get("text")
                        if text is not None:
                            yield str(text)
                    generated_text = chunk.get("generated_text")
                    if generated_text is not None:
                        break
        except urllib.error.HTTPError as exc:
            raise TGIBackendAdapterError(f"TGI HTTP error {exc.code}: {exc.reason}") from exc
        except Exception as exc:
            raise TGIBackendAdapterError(f"TGI stream request failed: {exc}") from exc


TGI_BACKEND_REGISTRY: dict[str, type[TGIBackendAdapter]] = {
    "tgi": TGIBackendAdapter,
}
