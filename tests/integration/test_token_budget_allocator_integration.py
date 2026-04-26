"""Integration: token budget allocator chat registry."""

from __future__ import annotations

import src.chat as chat
from src.model.config import AureliusConfig
from src.runtime.feature_flags import FeatureFlag, FEATURE_FLAG_REGISTRY


def test_allocator_registry():
    assert chat.CHAT_TOKEN_ALLOCATOR_REGISTRY["token_budget"] is chat.TokenBudgetAllocator


def test_config_default_off():
    assert AureliusConfig().chat_token_budget_allocator_enabled is False


def test_chatml_still_registered():
    assert "chatml" in chat.CHAT_TEMPLATE_REGISTRY


def test_smoke_allocate():
    FEATURE_FLAG_REGISTRY.register(FeatureFlag(name="chat.token_budget_allocator", enabled=True))
    cfg = AureliusConfig()
    assert cfg.chat_token_budget_allocator_enabled is True
    alloc = chat.TokenBudgetAllocator()
    out = alloc.allocate(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        [10, 20],
        total_budget=100,
    )
    assert sum(out.caps) == 100
