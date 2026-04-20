"""Tests for token_budget_allocator."""

from __future__ import annotations

import pytest

from src.chat.token_budget_allocator import TokenBudgetAllocator


def test_sum_matches_budget():
    alloc = TokenBudgetAllocator()
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    lens = [100, 200]
    out = alloc.allocate(msgs, lens, total_budget=256)
    assert sum(out.caps) == 256


def test_floors_respected():
    alloc = TokenBudgetAllocator(role_floors={"system": 50, "user": 10})
    msgs = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    out = alloc.allocate(msgs, [1, 1], total_budget=80)
    assert out.caps[0] >= 50
    assert out.caps[1] >= 10


def test_floor_exceeds_budget_raises():
    alloc = TokenBudgetAllocator(role_floors={"system": 100})
    msgs = [{"role": "system", "content": "x"}]
    with pytest.raises(RuntimeError):
        alloc.allocate(msgs, [50], total_budget=50)


def test_empty_messages():
    out = TokenBudgetAllocator().allocate([], [], 10)
    assert out.caps == ()


def test_length_mismatch():
    with pytest.raises(ValueError):
        TokenBudgetAllocator().allocate([{"role": "u", "content": "c"}], [1, 2], 3)


def test_bad_message():
    with pytest.raises(ValueError):
        TokenBudgetAllocator().allocate([{"content": "no role"}], [1], 5)


def test_negative_budget():
    with pytest.raises(ValueError):
        TokenBudgetAllocator().allocate([{"role": "u", "content": "c"}], [1], -1)


def test_negative_measured():
    with pytest.raises(ValueError):
        TokenBudgetAllocator().allocate([{"role": "u", "content": "c"}], [-1], 5)


def test_three_roles():
    alloc = TokenBudgetAllocator()
    msgs = [
        {"role": "system", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "c"},
    ]
    out = alloc.allocate(msgs, [10, 10, 10], total_budget=60)
    assert len(out.caps) == 3
    assert sum(out.caps) == 60
