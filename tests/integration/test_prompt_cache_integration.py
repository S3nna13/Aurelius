"""Integration tests for the prompt cache exposed via ``src.serving``.

Ensures the cache is re-exported from the package and that adding it
does not break importability of sibling serving modules.
"""

from __future__ import annotations

import importlib

import pytest


def test_prompt_cache_exposed_via_src_serving():
    mod = importlib.import_module("src.serving")
    assert hasattr(mod, "PromptCache")
    assert hasattr(mod, "CachedResponse")


def test_put_get_round_trip_on_ten_prompts():
    from src.serving import PromptCache

    cache = PromptCache(max_entries=64)
    prompts = [f"prompt-{i}" for i in range(10)]
    for i, p in enumerate(prompts):
        cache.put(p, f"completion-{i}", params={"temperature": 0.2})

    for i, p in enumerate(prompts):
        entry = cache.get(p, params={"temperature": 0.2})
        assert entry is not None
        assert entry.completion == f"completion-{i}"

    s = cache.stats()
    assert s["hits"] == 10
    assert s["misses"] == 0
    assert s["size"] == 10


@pytest.mark.parametrize(
    "mod_name",
    [
        "src.serving.openai_api_validator",
        "src.serving.prompt_cache",
    ],
)
def test_existing_serving_modules_still_importable(mod_name):
    importlib.import_module(mod_name)
