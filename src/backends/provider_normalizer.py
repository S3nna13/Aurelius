from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "NormalizedRequest",
    "NormalizedResponse",
    "ProviderNormalizer",
    "ProviderNormalizerError",
]

_SUPPORTED_PROVIDERS = frozenset({"openai", "anthropic", "local", "generic"})


class ProviderNormalizerError(Exception):
    """Raised when a provider format cannot be normalized or denormalized."""


@dataclass
class NormalizedRequest:
    prompt: str
    max_tokens: int
    temperature: float
    stop_sequences: list[str]
    stream: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class NormalizedResponse:
    text: str
    finish_reason: str
    token_count: int
    latency_ms: float
    raw: dict = field(default_factory=dict)


def _require_provider(provider: str) -> None:
    if provider not in _SUPPORTED_PROVIDERS:
        raise ProviderNormalizerError(
            f"Unsupported provider {provider!r}. "
            f"Supported: {sorted(_SUPPORTED_PROVIDERS)}"
        )


class ProviderNormalizer:
    """4-stage pipeline: normalize -> route -> call -> denormalize."""

    def normalize(self, raw_request: dict, provider: str) -> NormalizedRequest:
        _require_provider(provider)
        if provider == "openai":
            return NormalizedRequest(
                prompt=raw_request.get("messages", [{}])[-1].get("content", "")
                if "messages" in raw_request
                else raw_request.get("prompt", ""),
                max_tokens=int(raw_request.get("max_tokens", 256)),
                temperature=float(raw_request.get("temperature", 1.0)),
                stop_sequences=raw_request.get("stop", []) or [],
                stream=bool(raw_request.get("stream", False)),
                metadata={
                    "model": raw_request.get("model", ""),
                    "provider": "openai",
                },
            )
        if provider == "anthropic":
            messages = raw_request.get("messages", [])
            prompt = messages[-1].get("content", "") if messages else raw_request.get("prompt", "")
            return NormalizedRequest(
                prompt=prompt,
                max_tokens=int(raw_request.get("max_tokens", 1024)),
                temperature=float(raw_request.get("temperature", 1.0)),
                stop_sequences=raw_request.get("stop_sequences", []) or [],
                stream=bool(raw_request.get("stream", False)),
                metadata={
                    "model": raw_request.get("model", ""),
                    "provider": "anthropic",
                    "system": raw_request.get("system", ""),
                },
            )
        if provider == "local":
            return NormalizedRequest(
                prompt=raw_request.get("prompt", ""),
                max_tokens=int(raw_request.get("max_new_tokens", raw_request.get("max_tokens", 256))),
                temperature=float(raw_request.get("temperature", 0.8)),
                stop_sequences=raw_request.get("stop", []) or [],
                stream=bool(raw_request.get("stream", False)),
                metadata={
                    "model_path": raw_request.get("model_path", ""),
                    "provider": "local",
                },
            )
        return NormalizedRequest(
            prompt=raw_request.get("prompt", ""),
            max_tokens=int(raw_request.get("max_tokens", 256)),
            temperature=float(raw_request.get("temperature", 1.0)),
            stop_sequences=raw_request.get("stop_sequences", raw_request.get("stop", [])) or [],
            stream=bool(raw_request.get("stream", False)),
            metadata={"provider": "generic"},
        )

    def denormalize(self, response: NormalizedResponse, provider: str) -> dict:
        _require_provider(provider)
        if provider == "openai":
            return {
                "id": response.raw.get("id", "chatcmpl-stub"),
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": response.text},
                        "finish_reason": response.finish_reason,
                        "index": 0,
                    }
                ],
                "usage": {"completion_tokens": response.token_count},
            }
        if provider == "anthropic":
            return {
                "id": response.raw.get("id", "msg_stub"),
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": response.text}],
                "stop_reason": response.finish_reason,
                "usage": {"output_tokens": response.token_count},
            }
        if provider == "local":
            return {
                "generated_text": response.text,
                "finish_reason": response.finish_reason,
                "tokens_generated": response.token_count,
                "latency_ms": response.latency_ms,
            }
        return {
            "text": response.text,
            "finish_reason": response.finish_reason,
            "token_count": response.token_count,
            "latency_ms": response.latency_ms,
        }

    async def call(self, req: NormalizedRequest, backend: str) -> NormalizedResponse:
        t0 = time.monotonic()
        await asyncio.sleep(0)
        latency_ms = (time.monotonic() - t0) * 1000.0
        stub_text = f"[stub:{backend}] {req.prompt[:80]}"
        token_count = len(stub_text.split())
        return NormalizedResponse(
            text=stub_text,
            finish_reason="stop",
            token_count=token_count,
            latency_ms=latency_ms,
            raw={"backend": backend},
        )

    async def pipeline(self, raw_request: dict, provider: str, backend: str) -> dict:
        req = self.normalize(raw_request, provider)
        response = await self.call(req, backend)
        return self.denormalize(response, provider)
