"""Model debug information panel for the Aurelius terminal UI surface.

Inspired by MoonshotAI/kimi-cli debug surfaces (MIT), Anthropic Claude Code progress rendering (MIT),
clean-room reimplementation with original Aurelius design.

Only rich, stdlib, and project-local imports are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class DebugPanelError(Exception):
    """Raised when DebugPanel encounters a missing section or metric."""


@dataclass
class DebugMetric:
    """A single named metric with a value and formatting hint.

    Attributes:
        name: Metric identifier.
        value: Current value; may be float, int, str, or None.
        unit: Optional unit label appended to formatted output.
        fmt: Python format string applied when value is numeric.
    """

    name: str
    value: float | int | str | None
    unit: str = ""
    fmt: str = "{:.4f}"


@dataclass
class DebugSection:
    """A titled group of :class:`DebugMetric` objects.

    Attributes:
        title: Section heading displayed in the panel border.
        metrics: Ordered list of metrics for this section.
        collapsible: Whether the section can be collapsed in the UI.
    """

    title: str
    metrics: list[DebugMetric] = field(default_factory=list)
    collapsible: bool = True


def _format_value(metric: DebugMetric) -> str:
    """Format a metric value using its ``fmt`` hint where applicable."""
    if metric.value is None:
        return "—"
    if isinstance(metric.value, (int, float)):
        try:
            formatted = metric.fmt.format(metric.value)
        except (ValueError, KeyError):
            formatted = str(metric.value)
    else:
        formatted = str(metric.value)
    if metric.unit:
        return f"{formatted} {metric.unit}"
    return formatted


def _default_sections() -> dict[str, DebugSection]:
    return {
        "Model": DebugSection(
            title="Model",
            metrics=[
                DebugMetric(name="loss", value=None, unit="", fmt="{:.4f}"),
                DebugMetric(name="perplexity", value=None, unit="", fmt="{:.4f}"),
                DebugMetric(name="tokens_per_sec", value=None, unit="tok/s", fmt="{:.2f}"),
            ],
        ),
        "Memory": DebugSection(
            title="Memory",
            metrics=[
                DebugMetric(name="gpu_allocated_gb", value=None, unit="GB", fmt="{:.3f}"),
                DebugMetric(name="gpu_reserved_gb", value=None, unit="GB", fmt="{:.3f}"),
            ],
        ),
        "Attention": DebugSection(
            title="Attention",
            metrics=[
                DebugMetric(name="avg_attn_entropy", value=None, unit="", fmt="{:.4f}"),
                DebugMetric(name="kv_cache_size_mb", value=None, unit="MB", fmt="{:.2f}"),
            ],
        ),
    }


class DebugPanel:
    """Rich-rendered panel that displays model debug metrics grouped by section.

    Pre-populated with 3 default sections: ``"Model"``, ``"Memory"``,
    and ``"Attention"``.  All mutations go through explicit methods.
    """

    def __init__(self) -> None:
        self._sections: dict[str, DebugSection] = _default_sections()

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add_section(self, section: DebugSection) -> None:
        """Register *section* by its title; overwrites an existing entry."""
        self._sections[section.title] = section

    def update_metric(
        self,
        section_title: str,
        metric_name: str,
        value: float | int | str | None,
    ) -> None:
        """Update the value of a named metric within a section.

        Args:
            section_title: Title of the target section.
            metric_name: Name of the metric to update.
            value: New value.

        Raises:
            DebugPanelError: If the section or metric is not found.
        """
        if section_title not in self._sections:
            raise DebugPanelError(
                f"section {section_title!r} not found; "
                f"available: {list(self._sections)}"
            )
        section = self._sections[section_title]
        for metric in section.metrics:
            if metric.name == metric_name:
                metric.value = value
                return
        raise DebugPanelError(
            f"metric {metric_name!r} not found in section {section_title!r}; "
            f"available: {[m.name for m in section.metrics]}"
        )

    def add_metric(self, section_title: str, metric: DebugMetric) -> None:
        """Append *metric* to an existing section.

        Args:
            section_title: Title of the target section.
            metric: The :class:`DebugMetric` to append.

        Raises:
            DebugPanelError: If the section is not found.
        """
        if section_title not in self._sections:
            raise DebugPanelError(
                f"section {section_title!r} not found; "
                f"available: {list(self._sections)}"
            )
        self._sections[section_title].metrics.append(metric)

    # ------------------------------------------------------------------
    # Render API
    # ------------------------------------------------------------------

    def render(self, console: Console, collapsed: bool = False) -> None:
        """Render all sections to *console* as Rich panels.

        Args:
            console: A :class:`~rich.console.Console` to print to.
            collapsed: When ``True`` only the section title is shown.
        """
        if not self._sections:
            console.print("[dim](no debug sections)[/dim]")
            return

        for section in self._sections.values():
            title_text = Text(section.title, style="bold cyan")
            if collapsed:
                console.print(Panel(Text("…", style="dim"), title=title_text))
                continue

            table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
            table.add_column("Metric", style="dim", no_wrap=True)
            table.add_column("Value", justify="right")

            if not section.metrics:
                table.add_row("[dim](none)[/dim]", "")
            else:
                for metric in section.metrics:
                    table.add_row(metric.name, _format_value(metric))

            console.print(Panel(table, title=title_text))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of all sections and metrics."""
        return {
            title: {
                "title": section.title,
                "collapsible": section.collapsible,
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "unit": m.unit,
                        "fmt": m.fmt,
                    }
                    for m in section.metrics
                ],
            }
            for title, section in self._sections.items()
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Named pool of :class:`DebugPanel` instances.
DEBUG_PANEL_REGISTRY: dict[str, DebugPanel] = {}

__all__ = [
    "DebugMetric",
    "DebugSection",
    "DebugPanel",
    "DebugPanelError",
    "DEBUG_PANEL_REGISTRY",
]
