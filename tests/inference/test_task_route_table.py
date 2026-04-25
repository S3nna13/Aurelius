"""Tests for src/inference/task_route_table.py"""

from __future__ import annotations

import pytest

from src.inference.request_classifier import ComplexityTier, TaskType
from src.inference.task_route_table import TASK_ROUTE_TABLE, RouteEntry, TaskRouteTable


def test_route_entry_defaults():
    entry = RouteEntry(task_type=TaskType.CHAT, complexity=ComplexityTier.LOW, backend="cached")
    assert entry.priority == 5
    assert entry.timeout_s == 30.0


def test_default_low_routes_to_cached():
    table = TaskRouteTable()
    entry = table.lookup(TaskType.CHAT, ComplexityTier.LOW)
    assert entry.backend == "cached"


def test_default_medium_routes_to_local():
    table = TaskRouteTable()
    entry = table.lookup(TaskType.CODE, ComplexityTier.MEDIUM)
    assert entry.backend == "local"


def test_default_high_routes_to_api():
    table = TaskRouteTable()
    entry = table.lookup(TaskType.MATH, ComplexityTier.HIGH)
    assert entry.backend == "api"


def test_register_overrides_default():
    table = TaskRouteTable()
    custom = RouteEntry(task_type=TaskType.CODE, complexity=ComplexityTier.HIGH, backend="gpu-cluster", priority=1)
    table.register(custom)
    result = table.lookup(TaskType.CODE, ComplexityTier.HIGH)
    assert result.backend == "gpu-cluster"


def test_list_routes_returns_all_entries():
    table = TaskRouteTable()
    routes = table.list_routes()
    assert len(routes) >= len(TaskType) * len(ComplexityTier)


def test_custom_init_routes_applied():
    custom = RouteEntry(task_type=TaskType.REASONING, complexity=ComplexityTier.MEDIUM, backend="tpu")
    table = TaskRouteTable(routes=[custom])
    result = table.lookup(TaskType.REASONING, ComplexityTier.MEDIUM)
    assert result.backend == "tpu"


def test_lookup_unknown_task_type_low():
    table = TaskRouteTable()
    result = table.lookup(TaskType.UNKNOWN, ComplexityTier.LOW)
    assert result.backend == "cached"


def test_lookup_unknown_task_type_high():
    table = TaskRouteTable()
    result = table.lookup(TaskType.UNKNOWN, ComplexityTier.HIGH)
    assert result.backend == "api"


def test_module_singleton_is_task_route_table():
    assert isinstance(TASK_ROUTE_TABLE, TaskRouteTable)


def test_singleton_lookup_works():
    result = TASK_ROUTE_TABLE.lookup(TaskType.CHAT, ComplexityTier.LOW)
    assert result.backend == "cached"


def test_register_preserves_other_routes():
    table = TaskRouteTable()
    table.register(RouteEntry(task_type=TaskType.MATH, complexity=ComplexityTier.LOW, backend="fast"))
    assert table.lookup(TaskType.CODE, ComplexityTier.LOW).backend == "cached"


def test_route_entry_custom_timeout():
    entry = RouteEntry(task_type=TaskType.TRANSLATE, complexity=ComplexityTier.HIGH, backend="api", timeout_s=120.0)
    assert entry.timeout_s == 120.0
