"""Unit tests for src/agent/plugin_hook.py — PluginHookRegistry."""
from __future__ import annotations

import pytest

from src.agent.plugin_hook import HOOK_POINTS, HOOK_REGISTRY, PluginHookRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh() -> PluginHookRegistry:
    """Return a brand-new registry to avoid test pollution."""
    return PluginHookRegistry()


# ---------------------------------------------------------------------------
# 1. test_register_and_fire
# ---------------------------------------------------------------------------

def test_register_and_fire():
    reg = _fresh()
    called = []
    reg.register("pre_tool_call", lambda **kw: called.append(True))
    reg.fire("pre_tool_call")
    assert called == [True]


# ---------------------------------------------------------------------------
# 2. test_multiple_hooks_called_in_order
# ---------------------------------------------------------------------------

def test_multiple_hooks_called_in_order():
    reg = _fresh()
    order = []
    reg.register("pre_tool_call", lambda **kw: order.append(1))
    reg.register("pre_tool_call", lambda **kw: order.append(2))
    reg.register("pre_tool_call", lambda **kw: order.append(3))
    reg.fire("pre_tool_call")
    assert order == [1, 2, 3]


# ---------------------------------------------------------------------------
# 3. test_pre_tool_call_receives_kwargs
# ---------------------------------------------------------------------------

def test_pre_tool_call_receives_kwargs():
    reg = _fresh()
    received = {}

    def hook(**kw):
        received.update(kw)

    reg.register("pre_tool_call", hook)
    reg.fire("pre_tool_call", tool_name="bash", args={"cmd": "ls"})
    assert received["tool_name"] == "bash"
    assert received["args"] == {"cmd": "ls"}


# ---------------------------------------------------------------------------
# 4. test_post_tool_call_receives_result
# ---------------------------------------------------------------------------

def test_post_tool_call_receives_result():
    reg = _fresh()
    received = {}

    def hook(**kw):
        received.update(kw)

    reg.register("post_tool_call", hook)
    reg.fire("post_tool_call", tool_name="read_file", result="file contents")
    assert received["result"] == "file contents"
    assert received["tool_name"] == "read_file"


# ---------------------------------------------------------------------------
# 5. test_on_error_receives_exception
# ---------------------------------------------------------------------------

def test_on_error_receives_exception():
    reg = _fresh()
    received = {}

    def hook(**kw):
        received.update(kw)

    reg.register("on_error", hook)
    exc = ValueError("something went wrong")
    reg.fire("on_error", exception=exc)
    assert received["exception"] is exc


# ---------------------------------------------------------------------------
# 6. test_unknown_point_register_raises
# ---------------------------------------------------------------------------

def test_unknown_point_register_raises():
    reg = _fresh()
    with pytest.raises(ValueError, match="Unknown hook point"):
        reg.register("bad_point", lambda **kw: None)


# ---------------------------------------------------------------------------
# 7. test_unknown_point_fire_raises
# ---------------------------------------------------------------------------

def test_unknown_point_fire_raises():
    reg = _fresh()
    with pytest.raises(ValueError, match="Unknown hook point"):
        reg.fire("bad_point")


# ---------------------------------------------------------------------------
# 8. test_hook_exception_propagates
# ---------------------------------------------------------------------------

def test_hook_exception_propagates():
    reg = _fresh()

    def boom(**kw):
        raise RuntimeError("hook failure")

    reg.register("post_generation", boom)
    with pytest.raises(RuntimeError, match="hook failure"):
        reg.fire("post_generation", response="hello")


# ---------------------------------------------------------------------------
# 9. test_clear_specific_point
# ---------------------------------------------------------------------------

def test_clear_specific_point():
    reg = _fresh()
    reg.register("pre_tool_call", lambda **kw: None)
    reg.register("post_tool_call", lambda **kw: None)
    reg.clear("pre_tool_call")
    assert reg.hook_count("pre_tool_call") == 0
    assert reg.hook_count("post_tool_call") == 1


# ---------------------------------------------------------------------------
# 10. test_clear_all
# ---------------------------------------------------------------------------

def test_clear_all():
    reg = _fresh()
    for point in HOOK_POINTS:
        reg.register(point, lambda **kw: None)
    reg.clear()
    for point in HOOK_POINTS:
        assert reg.hook_count(point) == 0


# ---------------------------------------------------------------------------
# 11. test_hook_count
# ---------------------------------------------------------------------------

def test_hook_count():
    reg = _fresh()
    reg.register("pre_generation", lambda **kw: None)
    reg.register("pre_generation", lambda **kw: None)
    reg.register("pre_generation", lambda **kw: None)
    assert reg.hook_count("pre_generation") == 3


# ---------------------------------------------------------------------------
# 12. test_all_points_returns_tuple
# ---------------------------------------------------------------------------

def test_all_points_returns_tuple():
    reg = _fresh()
    points = reg.all_points()
    assert isinstance(points, tuple)
    assert set(points) == set(HOOK_POINTS)
    assert len(points) == 5


# ---------------------------------------------------------------------------
# 13. test_empty_hook_list_no_crash
# ---------------------------------------------------------------------------

def test_empty_hook_list_no_crash():
    reg = _fresh()
    # No hooks registered — should not raise
    reg.fire("pre_generation", prompt="Hello")


# ---------------------------------------------------------------------------
# 14. test_second_hook_not_called_if_first_raises
# ---------------------------------------------------------------------------

def test_second_hook_not_called_if_first_raises():
    reg = _fresh()
    second_called = []

    def first_hook(**kw):
        raise RuntimeError("first hook blows up")

    def second_hook(**kw):
        second_called.append(True)

    reg.register("on_error", first_hook)
    reg.register("on_error", second_hook)

    with pytest.raises(RuntimeError):
        reg.fire("on_error", exception=Exception("original"))

    assert second_called == [], "second hook must not be called when first raises"


# ---------------------------------------------------------------------------
# 15. test_singleton_exists
# ---------------------------------------------------------------------------

def test_singleton_exists():
    assert isinstance(HOOK_REGISTRY, PluginHookRegistry)
