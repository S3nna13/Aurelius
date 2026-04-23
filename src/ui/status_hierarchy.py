"""Task/agent/session status hierarchy for the Aurelius terminal UI surface.

Inspired by MoonshotAI/kimi-cli (MIT, terminal session lifecycle),
Anthropic Claude Code (MIT, command palette UX), clean-room
reimplementation with original Aurelius branding.

Provides a tree-structured status model that mirrors the shell's
Workstream/task hierarchy and renders via Rich.  Only rich, stdlib,
and project-local imports are used.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.tree import Tree as RichTree


class StatusLevel(enum.Enum):
    """Hierarchy level of a :class:`StatusNode`."""

    SESSION = "session"
    WORKSTREAM = "workstream"
    TASK = "task"
    SUBTASK = "subtask"


class StatusState(enum.Enum):
    """Execution state of a :class:`StatusNode`."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


_STATE_GLYPHS: dict[StatusState, str] = {
    StatusState.IDLE: "○",
    StatusState.RUNNING: "◎",
    StatusState.PAUSED: "⏸",
    StatusState.SUCCESS: "✓",
    StatusState.FAILED: "✗",
    StatusState.CANCELLED: "⊘",
}

_STATE_STYLES: dict[StatusState, str] = {
    StatusState.IDLE: "dim",
    StatusState.RUNNING: "bold cyan",
    StatusState.PAUSED: "yellow",
    StatusState.SUCCESS: "bold green",
    StatusState.FAILED: "bold red",
    StatusState.CANCELLED: "dim red",
}


@dataclass
class StatusNode:
    """A single node in the status tree.

    Attributes:
        id: Unique identifier for this node within a :class:`StatusTree`.
        level: Hierarchy level (SESSION, WORKSTREAM, TASK, SUBTASK).
        state: Current execution state.
        label: Human-readable label rendered in the tree.
        progress: Optional progress fraction in ``[0.0, 1.0]``.
        children: Ordered child nodes (mutated by :class:`StatusTree`).
        metadata: Arbitrary key/value pairs for persistence or tooling.
    """

    id: str
    level: StatusLevel
    state: StatusState
    label: str
    progress: float | None = None
    children: list[StatusNode] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class StatusTree:
    """Manages a tree of :class:`StatusNode` objects.

    All mutations go through methods; direct attribute assignment from
    outside the class is not the intended usage pattern.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, StatusNode] = {}
        self._roots: list[StatusNode] = []

    # ------------------------------------------------------------------
    # Mutation methods
    # ------------------------------------------------------------------

    def add_node(self, node: StatusNode, parent_id: str | None = None) -> None:
        """Add *node* to the tree, optionally as a child of *parent_id*.

        Args:
            node: The :class:`StatusNode` to add.
            parent_id: If provided the node is appended to the matching
                parent's ``children`` list.  If ``None`` the node
                becomes a root-level entry.

        Raises:
            ValueError: If *node.id* is already present in the tree.
            KeyError: If *parent_id* is provided but not found.
        """
        if node.id in self._nodes:
            raise ValueError(f"node id {node.id!r} is already in the tree")
        self._nodes[node.id] = node
        if parent_id is None:
            self._roots.append(node)
        else:
            parent = self._nodes[parent_id]  # raises KeyError if missing
            parent.children.append(node)

    def update_state(
        self,
        node_id: str,
        state: StatusState,
        progress: float | None = None,
    ) -> None:
        """Update the *state* (and optionally *progress*) of a node.

        Args:
            node_id: Id of the node to update.
            state: New :class:`StatusState`.
            progress: Optional new progress value (``0.0``–``1.0``).

        Raises:
            KeyError: If *node_id* is not in the tree.
        """
        node = self._nodes[node_id]  # raises KeyError if missing
        node.state = state
        if progress is not None:
            node.progress = progress

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> StatusNode:
        """Return the node with *node_id*.

        Raises:
            KeyError: If *node_id* is not in the tree.
        """
        try:
            return self._nodes[node_id]
        except KeyError:
            raise KeyError(f"no status node with id {node_id!r}") from None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, console: Console) -> None:
        """Render the status tree to *console* as a Rich Tree."""

        def _build(rich_node: RichTree, status_node: StatusNode) -> None:
            glyph = _STATE_GLYPHS.get(status_node.state, "?")
            style = _STATE_STYLES.get(status_node.state, "")
            label_parts = [f"[{style}]{glyph} {status_node.label}[/{style}]"]
            if status_node.progress is not None:
                pct = int(status_node.progress * 100)
                label_parts.append(f" [dim]{pct}%[/dim]")
            child_tree = rich_node.add("".join(label_parts))
            for child in status_node.children:
                _build(child_tree, child)

        if not self._roots:
            console.print("[dim](empty status tree)[/dim]")
            return

        root_rich = RichTree("[bold]Aurelius Status[/bold]")
        for root_node in self._roots:
            _build(root_rich, root_node)
        console.print(root_rich)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the tree."""

        def _node_to_dict(node: StatusNode) -> dict[str, Any]:
            return {
                "id": node.id,
                "level": node.level.value,
                "state": node.state.value,
                "label": node.label,
                "progress": node.progress,
                "metadata": dict(node.metadata),
                "children": [_node_to_dict(c) for c in node.children],
            }

        return {
            "roots": [_node_to_dict(r) for r in self._roots],
        }


STATUS_TREE_REGISTRY: dict[str, StatusTree] = {}

__all__ = [
    "StatusLevel",
    "StatusState",
    "StatusNode",
    "StatusTree",
    "STATUS_TREE_REGISTRY",
]
