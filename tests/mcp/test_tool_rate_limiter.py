"""Tests for tool_rate_limiter."""
from __future__ import annotations
import time
from src.mcp.tool_rate_limiter import ToolRateLimiter, RateLimitResult
class TestRateLimitResult:
    def test_allowed(self): r=RateLimitResult(allowed=True);assert r.allowed;assert r.retry_after==0
    def test_denied(self): r=RateLimitResult(allowed=False,retry_after=5.0);assert not r.allowed;assert r.retry_after==5.0
class TestToolRateLimiter:
    def test_first_call_allowed(self): r=ToolRateLimiter(max_calls=5,window=1.0);assert r.check_call("tool1").allowed
    def test_exceeds_limit(self): r=ToolRateLimiter(max_calls=2,window=10.0);r.check_call("t");r.check_call("t");result=r.check_call("t");assert not result.allowed
    def test_window_reset(self): r=ToolRateLimiter(max_calls=1,window=0.01);r.check_call("t");time.sleep(0.02);assert r.check_call("t").allowed
    def test_different_tools_separate(self): r=ToolRateLimiter(max_calls=1,window=10);r.check_call("a");assert r.check_call("b").allowed
    def test_get_stats(self): r=ToolRateLimiter(5,10);r.check_call("x");s=r.get_stats("x");assert s["calls"]==1;assert s["remaining"]==4
