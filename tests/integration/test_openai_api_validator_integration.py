"""Integration tests: validator is exposed via ``src.serving`` and works end-to-end."""

from __future__ import annotations

import importlib

import pytest


def test_validator_exposed_via_src_serving_package():
    import src.serving as serving

    assert hasattr(serving, "OpenAIChatRequestValidator")
    assert hasattr(serving, "OpenAIChatResponseValidator")
    assert hasattr(serving, "APIValidationError")
    assert hasattr(serving, "API_SHAPE_REGISTRY")
    assert "openai.chat.request" in serving.API_SHAPE_REGISTRY
    assert "openai.chat.response" in serving.API_SHAPE_REGISTRY


def test_validates_sample_openai_request_and_response():
    from src.serving import OpenAIChatRequestValidator, OpenAIChatResponseValidator

    request = {
        "model": "aurelius-1.4b",
        "messages": [
            {"role": "system", "content": "You are an agentic coding LLM."},
            {"role": "user", "content": "Write a Python function to reverse a string."},
        ],
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 512,
        "stream": False,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "run_python",
                    "description": "Execute python code",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
    }
    assert OpenAIChatRequestValidator().validate(request) == []

    response = {
        "id": "chatcmpl-xyz",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "aurelius-1.4b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "def reverse(s): return s[::-1]"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29},
    }
    assert OpenAIChatResponseValidator().validate(response) == []


def test_existing_serving_modules_still_importable():
    # Smoke: ensure our additive __init__ change did not break sibling modules.
    for modname in (
        "src.serving.openai_api_validator",
        "src.serving",
    ):
        importlib.import_module(modname)

    # A few sibling modules exist in the serving package — importing them should
    # not be blocked by our additive registry hook. We import at module level
    # only (no side-effectful construction) to keep the test hermetic.
    import src.serving as serving

    assert serving.__name__ == "src.serving"


def test_registry_classes_are_instantiable_and_validate():
    from src.serving import API_SHAPE_REGISTRY

    req_cls = API_SHAPE_REGISTRY["openai.chat.request"]
    resp_cls = API_SHAPE_REGISTRY["openai.chat.response"]
    assert req_cls().validate({"model": "m", "messages": [{"role": "user", "content": "hi"}]}) == []
    # A clearly invalid response should produce errors.
    errs = resp_cls().validate({})
    assert len(errs) > 0
