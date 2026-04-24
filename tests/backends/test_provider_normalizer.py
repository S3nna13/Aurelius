from __future__ import annotations

import asyncio

import pytest

from src.backends.provider_normalizer import (
    NormalizedRequest,
    NormalizedResponse,
    ProviderNormalizer,
    ProviderNormalizerError,
)


@pytest.fixture()
def normalizer() -> ProviderNormalizer:
    return ProviderNormalizer()


# ---------------------------------------------------------------------------
# normalize – openai
# ---------------------------------------------------------------------------


def test_normalize_openai_messages(normalizer: ProviderNormalizer) -> None:
    raw = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        "temperature": 0.5,
        "stop": ["\n"],
        "stream": False,
    }
    req = normalizer.normalize(raw, "openai")
    assert isinstance(req, NormalizedRequest)
    assert req.prompt == "Hello"
    assert req.max_tokens == 100
    assert req.temperature == 0.5
    assert req.stop_sequences == ["\n"]
    assert req.stream is False
    assert req.metadata["provider"] == "openai"


def test_normalize_openai_prompt_fallback(normalizer: ProviderNormalizer) -> None:
    raw = {"prompt": "Hi there", "max_tokens": 50}
    req = normalizer.normalize(raw, "openai")
    assert req.prompt == "Hi there"


def test_normalize_openai_defaults(normalizer: ProviderNormalizer) -> None:
    req = normalizer.normalize({}, "openai")
    assert req.max_tokens == 256
    assert req.temperature == 1.0
    assert req.stop_sequences == []
    assert req.stream is False


# ---------------------------------------------------------------------------
# normalize – anthropic
# ---------------------------------------------------------------------------


def test_normalize_anthropic_messages(normalizer: ProviderNormalizer) -> None:
    raw = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Explain RoPE"}],
        "max_tokens": 512,
        "temperature": 0.3,
        "stop_sequences": ["END"],
        "system": "You are a helpful assistant.",
    }
    req = normalizer.normalize(raw, "anthropic")
    assert req.prompt == "Explain RoPE"
    assert req.max_tokens == 512
    assert req.stop_sequences == ["END"]
    assert req.metadata["system"] == "You are a helpful assistant."
    assert req.metadata["provider"] == "anthropic"


def test_normalize_anthropic_defaults(normalizer: ProviderNormalizer) -> None:
    req = normalizer.normalize({}, "anthropic")
    assert req.max_tokens == 1024
    assert req.temperature == 1.0


# ---------------------------------------------------------------------------
# normalize – local
# ---------------------------------------------------------------------------


def test_normalize_local_max_new_tokens(normalizer: ProviderNormalizer) -> None:
    raw = {"prompt": "Generate code", "max_new_tokens": 300, "temperature": 0.9}
    req = normalizer.normalize(raw, "local")
    assert req.max_tokens == 300
    assert req.temperature == 0.9
    assert req.metadata["provider"] == "local"


def test_normalize_local_defaults(normalizer: ProviderNormalizer) -> None:
    req = normalizer.normalize({}, "local")
    assert req.max_tokens == 256
    assert req.temperature == 0.8


# ---------------------------------------------------------------------------
# normalize – generic
# ---------------------------------------------------------------------------


def test_normalize_generic(normalizer: ProviderNormalizer) -> None:
    raw = {"prompt": "test", "max_tokens": 64, "stop_sequences": ["STOP"]}
    req = normalizer.normalize(raw, "generic")
    assert req.prompt == "test"
    assert req.max_tokens == 64
    assert req.stop_sequences == ["STOP"]


def test_normalize_unsupported_provider_raises(normalizer: ProviderNormalizer) -> None:
    with pytest.raises(ProviderNormalizerError, match="Unsupported provider"):
        normalizer.normalize({}, "vertex")


# ---------------------------------------------------------------------------
# denormalize
# ---------------------------------------------------------------------------


def _make_response(**kwargs) -> NormalizedResponse:
    defaults = dict(
        text="output text",
        finish_reason="stop",
        token_count=10,
        latency_ms=42.0,
        raw={},
    )
    defaults.update(kwargs)
    return NormalizedResponse(**defaults)


def test_denormalize_openai_shape(normalizer: ProviderNormalizer) -> None:
    resp = _make_response()
    out = normalizer.denormalize(resp, "openai")
    assert "choices" in out
    assert out["choices"][0]["message"]["content"] == "output text"
    assert out["choices"][0]["finish_reason"] == "stop"


def test_denormalize_anthropic_shape(normalizer: ProviderNormalizer) -> None:
    resp = _make_response()
    out = normalizer.denormalize(resp, "anthropic")
    assert out["type"] == "message"
    assert out["content"][0]["text"] == "output text"
    assert out["stop_reason"] == "stop"


def test_denormalize_local_shape(normalizer: ProviderNormalizer) -> None:
    resp = _make_response(latency_ms=5.0)
    out = normalizer.denormalize(resp, "local")
    assert out["generated_text"] == "output text"
    assert out["latency_ms"] == 5.0


def test_denormalize_generic_shape(normalizer: ProviderNormalizer) -> None:
    resp = _make_response()
    out = normalizer.denormalize(resp, "generic")
    assert out["text"] == "output text"
    assert "finish_reason" in out


def test_denormalize_unsupported_provider_raises(normalizer: ProviderNormalizer) -> None:
    resp = _make_response()
    with pytest.raises(ProviderNormalizerError, match="Unsupported provider"):
        normalizer.denormalize(resp, "bedrock")


# ---------------------------------------------------------------------------
# async call + pipeline
# ---------------------------------------------------------------------------


def test_call_returns_normalized_response(normalizer: ProviderNormalizer) -> None:
    req = NormalizedRequest(
        prompt="hello",
        max_tokens=50,
        temperature=0.7,
        stop_sequences=[],
        stream=False,
    )
    resp = asyncio.run(normalizer.call(req, "pytorch"))
    assert isinstance(resp, NormalizedResponse)
    assert resp.finish_reason == "stop"
    assert resp.token_count > 0
    assert resp.latency_ms >= 0.0
    assert "pytorch" in resp.text


def test_pipeline_openai_end_to_end(normalizer: ProviderNormalizer) -> None:
    raw = {
        "messages": [{"role": "user", "content": "Hello world"}],
        "max_tokens": 32,
    }
    out = asyncio.run(normalizer.pipeline(raw, "openai", "vllm"))
    assert "choices" in out
    assert "vllm" in out["choices"][0]["message"]["content"]


def test_pipeline_anthropic_end_to_end(normalizer: ProviderNormalizer) -> None:
    raw = {
        "messages": [{"role": "user", "content": "Summarize"}],
        "max_tokens": 64,
    }
    out = asyncio.run(normalizer.pipeline(raw, "anthropic", "sglang"))
    assert out["type"] == "message"
    assert len(out["content"]) == 1
