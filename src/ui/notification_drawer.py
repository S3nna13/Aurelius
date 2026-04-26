"""Notification drawer: severity levels, dismiss, history.

Only rich, stdlib, and project-local imports are used.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


class NotificationLevel(StrEnum):
    """Severity level for a :class:`Notification`."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


_LEVEL_COLOR: dict[NotificationLevel, str] = {
    NotificationLevel.INFO: "blue",
    NotificationLevel.WARNING: "yellow",
    NotificationLevel.ERROR: "red",
    NotificationLevel.SUCCESS: "green",
}


@dataclass
class Notification:
    """A single notification entry.

    Attributes:
        id: Unique identifier (auto-generated 8-char hex if not provided).
        level: Severity level of the notification.
        title: Short summary line.
        body: Optional longer description.
        dismissed: Whether the notification has been dismissed by the user.
    """

    level: NotificationLevel
    title: str
    body: str = ""
    dismissed: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


class NotificationDrawer:
    """Manages a capped history of :class:`Notification` objects.

    Provides push/dismiss semantics and Rich-based rendering of active
    (undismissed) notifications.
    """

    def __init__(self, max_history: int = 50) -> None:
        self._max_history = max_history
        self._notifications: list[Notification] = []

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def push(
        self,
        level: NotificationLevel,
        title: str,
        body: str = "",
    ) -> Notification:
        """Create a new :class:`Notification` and append it to history.

        If the history would exceed *max_history* after the append, the
        oldest notification is evicted first.

        Args:
            level: Severity level.
            title: Short summary line.
            body: Optional longer description.

        Returns:
            The newly created :class:`Notification`.
        """
        notification = Notification(level=level, title=title, body=body)
        if len(self._notifications) >= self._max_history:
            self._notifications.pop(0)
        self._notifications.append(notification)
        return notification

    def dismiss(self, notification_id: str) -> bool:
        """Mark the notification with *notification_id* as dismissed.

        Args:
            notification_id: The ``id`` of the notification to dismiss.

        Returns:
            ``True`` if found and dismissed, ``False`` if not found.
        """
        for n in self._notifications:
            if n.id == notification_id:
                n.dismissed = True
                return True
        return False

    def clear_dismissed(self) -> int:
        """Remove all dismissed notifications from history.

        Returns:
            The number of notifications removed.
        """
        before = len(self._notifications)
        self._notifications = [n for n in self._notifications if not n.dismissed]
        return before - len(self._notifications)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def active(self) -> list[Notification]:
        """Return undismissed notifications in insertion order."""
        return [n for n in self._notifications if not n.dismissed]

    def history(self) -> list[Notification]:
        """Return all notifications (dismissed and active) in insertion order."""
        return list(self._notifications)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_active(self) -> str:
        """Return a Rich-formatted string for all active notifications.

        Each notification is rendered as a bordered panel whose border
        colour reflects its severity level:
        - info → blue
        - warning → yellow
        - error → red
        - success → green

        Returns:
            A string ready for printing to the terminal.  Empty string
            when there are no active notifications.
        """
        active = self.active()
        if not active:
            return ""

        console = Console(highlight=False)
        with console.capture() as capture:
            for n in active:
                color = _LEVEL_COLOR[n.level]
                body_text = Text(n.body) if n.body else Text("")
                panel = Panel(
                    body_text,
                    title=f"[{color} bold]{n.title}[/{color} bold]",
                    subtitle=f"[dim]{n.level.value}[/dim]",
                    border_style=color,
                    expand=False,
                )
                console.print(panel)
        return capture.get()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

DEFAULT_NOTIFICATION_DRAWER = NotificationDrawer()
