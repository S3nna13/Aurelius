"""Model metadata display panel for the Aurelius terminal UI surface.

Inspired by kimi-cli (MIT, multi-tab), Claude Code (MIT, panel layouts),
clean-room Aurelius implementation with original branding.

Only rich, stdlib, and project-local imports are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text


class ModelInfoError(Exception):
    """Raised when a ModelInfoPanel operation fails."""


@dataclass
class ModelInfoEntry:
    """A single model metadata entry.

    Attributes:
        key: Unique identifier for this entry.
        value: Current value; may be str, int, float, or None.
        category: Logical grouping category for this entry.
        editable: Whether this entry can be edited interactively.
    """

    key: str
    value: str | int | float | None
    category: str = "general"
    editable: bool = False


def _format_value(value: str | int | float | None) -> str:
    """Format an entry value for display."""
    if value is None:
        return "—"
    return str(value)


def _default_entries() -> dict[str, ModelInfoEntry]:
    """Return the default pre-populated entries."""
    manifest_entries = [
        ModelInfoEntry(key="family_name", value=None, category="manifest"),
        ModelInfoEntry(key="variant_name", value=None, category="manifest"),
        ModelInfoEntry(key="backbone_class", value=None, category="manifest"),
        ModelInfoEntry(key="vocab_size", value=None, category="manifest"),
        ModelInfoEntry(key="max_seq_len", value=None, category="manifest"),
        ModelInfoEntry(key="n_parameters", value=None, category="manifest"),
        ModelInfoEntry(key="backend_name", value=None, category="manifest"),
    ]
    inference_entries = [
        ModelInfoEntry(key="tokens_per_sec", value=None, category="inference"),
        ModelInfoEntry(key="latency_p50_ms", value=None, category="inference"),
        ModelInfoEntry(key="context_utilization", value=None, category="inference"),
    ]
    result: dict[str, ModelInfoEntry] = {}
    for entry in manifest_entries + inference_entries:
        result[entry.key] = entry
    return result


class ModelInfoPanel:
    """Rich-rendered panel displaying model metadata grouped by category.

    Pre-populated with manifest entries (family_name, variant_name,
    backbone_class, vocab_size, max_seq_len, n_parameters, backend_name)
    and inference entries (tokens_per_sec, latency_p50_ms,
    context_utilization).
    """

    def __init__(self) -> None:
        self._entries: dict[str, ModelInfoEntry] = _default_entries()

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add_entry(self, entry: ModelInfoEntry) -> None:
        """Register *entry* by key; overwrites an existing entry.

        Args:
            entry: The :class:`ModelInfoEntry` to register.
        """
        self._entries[entry.key] = entry

    def update(self, key: str, value: str | int | float | None) -> None:
        """Update the value of an existing entry.

        Args:
            key: The entry key to update.
            value: New value.

        Raises:
            ModelInfoError: If *key* is not found.
        """
        if key not in self._entries:
            raise ModelInfoError(f"entry {key!r} not found; available: {list(self._entries)}")
        self._entries[key].value = value

    def get(self, key: str) -> ModelInfoEntry:
        """Return the entry for *key*.

        Args:
            key: The entry key to retrieve.

        Raises:
            ModelInfoError: If *key* is not found.
        """
        if key not in self._entries:
            raise ModelInfoError(f"entry {key!r} not found; available: {list(self._entries)}")
        return self._entries[key]

    def entries_by_category(self, category: str) -> list[ModelInfoEntry]:
        """Return all entries belonging to *category*.

        Args:
            category: Category name to filter by.

        Returns:
            List of matching entries; empty list if none found.
        """
        return [e for e in self._entries.values() if e.category == category]

    # ------------------------------------------------------------------
    # Render API
    # ------------------------------------------------------------------

    def render(self, console: Console) -> None:
        """Render all entries to *console* as a Rich Table grouped by category.

        Args:
            console: A :class:`~rich.console.Console` to print to.
        """
        if not self._entries:
            console.print("[dim](no model info entries)[/dim]")
            return

        # Collect unique categories in insertion order
        categories: list[str] = []
        seen: set[str] = set()
        for entry in self._entries.values():
            if entry.category not in seen:
                categories.append(entry.category)
                seen.add(entry.category)

        for category in categories:
            entries = self.entries_by_category(category)
            table = Table(
                title=Text(category.upper(), style="bold cyan"),
                show_header=True,
                header_style="bold",
                box=None,
                padding=(0, 1),
            )
            table.add_column("Key", style="dim", no_wrap=True)
            table.add_column("Value", justify="right")
            table.add_column("Editable", style="dim", no_wrap=True)

            for entry in entries:
                editable_marker = "[green]yes[/green]" if entry.editable else ""
                table.add_row(entry.key, _format_value(entry.value), editable_marker)

            console.print(table)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of all entries."""
        return {
            key: {
                "key": entry.key,
                "value": entry.value,
                "category": entry.category,
                "editable": entry.editable,
            }
            for key, entry in self._entries.items()
        }


# ---------------------------------------------------------------------------
# Registry and default singleton
# ---------------------------------------------------------------------------

#: Named pool of :class:`ModelInfoPanel` instances.
MODEL_INFO_PANEL_REGISTRY: dict[str, ModelInfoPanel] = {}

#: Pre-populated default panel singleton.
DEFAULT_MODEL_INFO_PANEL: ModelInfoPanel = ModelInfoPanel()

__all__ = [
    "ModelInfoEntry",
    "ModelInfoPanel",
    "ModelInfoError",
    "MODEL_INFO_PANEL_REGISTRY",
    "DEFAULT_MODEL_INFO_PANEL",
]
