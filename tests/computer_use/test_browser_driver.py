"""Tests for src/computer_use/browser_driver.py.

Inspired by OpenDevin/OpenDevin (Apache-2.0, browser tool), MoonshotAI/Kimi-Dev
(Apache-2.0, patch synthesis), WebArena trajectory replay, clean-room reimplementation.
"""

from __future__ import annotations

import pytest

from src.computer_use.browser_driver import (
    BROWSER_DRIVER_REGISTRY,
    BrowserAction,
    BrowserDriver,
    BrowserDriverError,
    BrowserState,
    StubBrowserDriver,
)


# ---------------------------------------------------------------------------
# BrowserAction dataclass
# ---------------------------------------------------------------------------

class TestBrowserAction:
    def test_default_wait_ms(self):
        action = BrowserAction(action_type="click")
        assert action.wait_ms == 500

    def test_explicit_fields(self):
        action = BrowserAction(
            action_type="navigate",
            selector="#btn",
            value="hello",
            url="https://example.com",
            wait_ms=1000,
        )
        assert action.action_type == "navigate"
        assert action.selector == "#btn"
        assert action.value == "hello"
        assert action.url == "https://example.com"
        assert action.wait_ms == 1000

    def test_optional_fields_default_none(self):
        action = BrowserAction(action_type="screenshot")
        assert action.selector is None
        assert action.value is None
        assert action.url is None


# ---------------------------------------------------------------------------
# BrowserState dataclass
# ---------------------------------------------------------------------------

class TestBrowserState:
    def test_default_ready_true(self):
        state = BrowserState(url="https://example.com", title="Example")
        assert state.ready is True

    def test_optional_fields_default_none(self):
        state = BrowserState(url="https://example.com", title="Example")
        assert state.html_snapshot is None
        assert state.screenshot_path is None


# ---------------------------------------------------------------------------
# BrowserDriver abstract class
# ---------------------------------------------------------------------------

class TestBrowserDriverAbstract:
    def test_cannot_instantiate_abstract_driver(self):
        with pytest.raises(TypeError):
            BrowserDriver()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# StubBrowserDriver — happy paths
# ---------------------------------------------------------------------------

class TestStubBrowserDriverHappyPath:
    def test_navigate_updates_url(self):
        driver = StubBrowserDriver()
        state = driver.navigate("https://example.com")
        assert state.url == "https://example.com"

    def test_navigate_returns_browser_state(self):
        driver = StubBrowserDriver()
        state = driver.navigate("https://example.com")
        assert isinstance(state, BrowserState)

    def test_navigate_sets_ready_true(self):
        driver = StubBrowserDriver()
        state = driver.navigate("https://example.com")
        assert state.ready is True

    def test_navigate_sets_title(self):
        driver = StubBrowserDriver()
        state = driver.navigate("https://example.com")
        assert isinstance(state.title, str)
        assert len(state.title) > 0

    def test_click_returns_browser_state(self):
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        state = driver.click("#btn")
        assert isinstance(state, BrowserState)

    def test_click_returns_ready_true(self):
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        state = driver.click("#btn")
        assert state.ready is True

    def test_type_text_updates_state(self):
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        state = driver.type_text("#input", "hello")
        assert isinstance(state, BrowserState)
        assert state.ready is True

    def test_type_text_embeds_value_in_html(self):
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        state = driver.type_text("#input", "hello")
        assert state.html_snapshot is not None
        assert "hello" in state.html_snapshot

    def test_get_state_returns_current_state(self):
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        state = driver.get_state()
        assert state.url == "https://example.com"

    def test_get_state_before_navigate_returns_state(self):
        driver = StubBrowserDriver()
        state = driver.get_state()
        assert isinstance(state, BrowserState)

    def test_close_prevents_further_use(self):
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        driver.close()
        with pytest.raises(BrowserDriverError):
            driver.get_state()

    def test_multiple_navigations_update_url(self):
        driver = StubBrowserDriver()
        driver.navigate("https://first.com")
        state = driver.navigate("https://second.com")
        assert state.url == "https://second.com"


# ---------------------------------------------------------------------------
# StubBrowserDriver — error paths
# ---------------------------------------------------------------------------

class TestStubBrowserDriverErrors:
    def test_navigate_empty_url_raises(self):
        driver = StubBrowserDriver()
        with pytest.raises(BrowserDriverError):
            driver.navigate("")

    def test_navigate_whitespace_url_raises(self):
        driver = StubBrowserDriver()
        with pytest.raises(BrowserDriverError):
            driver.navigate("   ")

    def test_click_empty_selector_raises(self):
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        with pytest.raises(BrowserDriverError):
            driver.click("")

    def test_click_whitespace_selector_raises(self):
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        with pytest.raises(BrowserDriverError):
            driver.click("   ")

    def test_type_text_empty_selector_raises(self):
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")
        with pytest.raises(BrowserDriverError):
            driver.type_text("", "text")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestBrowserDriverRegistry:
    def test_registry_contains_stub(self):
        assert "stub" in BROWSER_DRIVER_REGISTRY

    def test_registry_stub_value_is_stub_class(self):
        assert BROWSER_DRIVER_REGISTRY["stub"] is StubBrowserDriver

    def test_registry_is_dict(self):
        assert isinstance(BROWSER_DRIVER_REGISTRY, dict)

    def test_stub_from_registry_is_instantiable(self):
        cls = BROWSER_DRIVER_REGISTRY["stub"]
        driver = cls()
        assert isinstance(driver, BrowserDriver)
