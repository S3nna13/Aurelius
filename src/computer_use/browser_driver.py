"""Browser automation abstraction for the Aurelius computer_use surface.

Inspired by OpenDevin/OpenDevin (Apache-2.0, browser tool), MoonshotAI/Kimi-Dev
(Apache-2.0, patch synthesis), WebArena trajectory replay, clean-room reimplementation.

No playwright, pyautogui, selenium, or OS accessibility API imports anywhere.
All browser interactions are behind abstract interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BrowserDriverError(Exception):
    """Raised when a browser driver operation fails or receives invalid input."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BrowserAction:
    """A single browser automation action."""

    action_type: str
    selector: str | None = None
    value: str | None = None
    url: str | None = None
    wait_ms: int = 500


@dataclass
class BrowserState:
    """Current state of the browser."""

    url: str
    title: str
    html_snapshot: str | None = None
    screenshot_path: str | None = None
    ready: bool = True


# ---------------------------------------------------------------------------
# Abstract driver interface
# ---------------------------------------------------------------------------


class BrowserDriver(ABC):
    """Abstract interface for browser automation drivers.

    Concrete implementations must not import playwright, pyautogui, selenium,
    or any OS accessibility API at module level.
    """

    @abstractmethod
    def navigate(self, url: str) -> BrowserState:
        """Navigate to *url* and return updated state.

        Parameters
        ----------
        url:
            Fully-qualified URL to navigate to.

        Returns
        -------
        BrowserState

        Raises
        ------
        BrowserDriverError
            If *url* is empty or navigation fails.
        """
        ...

    @abstractmethod
    def click(self, selector: str) -> BrowserState:
        """Click the element identified by *selector*.

        Parameters
        ----------
        selector:
            CSS selector or similar identifier string.

        Returns
        -------
        BrowserState

        Raises
        ------
        BrowserDriverError
            If *selector* is empty or the element cannot be found.
        """
        ...

    @abstractmethod
    def type_text(self, selector: str, text: str) -> BrowserState:
        """Type *text* into the element identified by *selector*.

        Parameters
        ----------
        selector:
            CSS selector or similar identifier string.
        text:
            Text to type.

        Returns
        -------
        BrowserState

        Raises
        ------
        BrowserDriverError
            If *selector* is empty.
        """
        ...

    @abstractmethod
    def get_state(self) -> BrowserState:
        """Return the current browser state without performing any action."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by this driver instance."""
        ...


# ---------------------------------------------------------------------------
# Stub driver — pure in-memory, no real browser
# ---------------------------------------------------------------------------


class StubBrowserDriver(BrowserDriver):
    """In-memory stub driver for testing and offline environments.

    Maintains a mutable ``_state: BrowserState``.  All operations mutate
    that state and return it without touching any real browser process.

    Raises
    ------
    BrowserDriverError
        On empty url or selector inputs.
    """

    def __init__(self) -> None:
        self._state: BrowserState = BrowserState(url="", title="", ready=False)
        self._closed: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_open(self) -> None:
        if self._closed:
            raise BrowserDriverError("StubBrowserDriver has been closed.")

    # ------------------------------------------------------------------
    # BrowserDriver implementation
    # ------------------------------------------------------------------

    def navigate(self, url: str) -> BrowserState:
        """Navigate to *url*; updates internal state url and title."""
        self._assert_open()
        if not url or not url.strip():
            raise BrowserDriverError("navigate() requires a non-empty url.")
        # Derive a naive title from the url (stub behaviour only).
        title = url.rstrip("/").split("/")[-1] or url
        self._state = BrowserState(
            url=url,
            title=title,
            html_snapshot=f"<html><head><title>{title}</title></head><body/></html>",
            screenshot_path=None,
            ready=True,
        )
        return self._state

    def click(self, selector: str) -> BrowserState:
        """Record a click on *selector*; returns current state."""
        self._assert_open()
        if not selector or not selector.strip():
            raise BrowserDriverError("click() requires a non-empty selector.")
        # Update html_snapshot to reflect the click in the stub.
        self._state = BrowserState(
            url=self._state.url,
            title=self._state.title,
            html_snapshot=self._state.html_snapshot,
            screenshot_path=self._state.screenshot_path,
            ready=True,
        )
        return self._state

    def type_text(self, selector: str, text: str) -> BrowserState:
        """Record typed *text* into *selector*; returns updated state."""
        self._assert_open()
        if not selector or not selector.strip():
            raise BrowserDriverError("type_text() requires a non-empty selector.")
        # Embed the typed value into the stub html_snapshot for traceability.
        # Handle both self-closing <body/> and open/close <body>…</body> forms.
        html = self._state.html_snapshot or "<html><body></body></html>"
        insert = f'<input data-selector="{selector}" value="{text}"/>'
        if "</body>" in html:
            updated_html = html.replace("</body>", f"{insert}</body>")
        elif "<body/>" in html:
            updated_html = html.replace("<body/>", f"<body>{insert}</body>")
        else:
            updated_html = html + insert
        self._state = BrowserState(
            url=self._state.url,
            title=self._state.title,
            html_snapshot=updated_html,
            screenshot_path=self._state.screenshot_path,
            ready=True,
        )
        return self._state

    def get_state(self) -> BrowserState:
        """Return current browser state."""
        self._assert_open()
        return self._state

    def close(self) -> None:
        """Mark this driver as closed."""
        self._closed = True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BROWSER_DRIVER_REGISTRY: dict[str, type[BrowserDriver]] = {
    "stub": StubBrowserDriver,
}
