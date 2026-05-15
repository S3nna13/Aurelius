"""Aurelius CLI v2 — Status bar renderer for interactive modes."""

from __future__ import annotations

import shutil
from dataclasses import dataclass


@dataclass
class StatusBarState:
    """State required to render the status bar."""

    model_name: str = "forge"
    backend: str = "mlx"
    quantization: str = "q4"
    context_length: int = 32768
    ram_used_gb: float = 0.0
    ram_total_gb: float = 32.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    cua_mode: str = "off"
    skill_count: int = 0
    tool_count: int = 0
    profile_id: str = "default"


def _format_ram(used: float, total: float) -> str:
    """Format RAM usage string."""
    if total <= 0:
        return "?/??GB"
    pct = (used / total) * 100
    bar_width = 10
    filled = int(bar_width * used / total)
    bar_filled = "█" * filled
    bar_empty = "░" * (bar_width - filled)
    return f"{bar_filled}{bar_empty} {used:.1f}/{total:.1f}GB"


def _context_str(ctx: int) -> str:
    """Format context length."""
    if ctx >= 1_000_000:
        return f"{ctx // 1_000_000}M"
    if ctx >= 1_000:
        return f"{ctx // 1_000}K"
    return str(ctx)


def render_status_bar(state: StatusBarState, width: int | None = None) -> str:
    """Render a compact status bar line.

    Example output:
    Aurelius Forge | local mlx q4 | ctx 32K | RAM ██░░░░░░░░ 14.2/32GB | CUA local_full | skills 150 | tools 18 | profile mac_silicon_32gb
    """
    if width is None:
        try:
            width = shutil.get_terminal_size().columns
        except Exception:
            width = 120

    model_label = state.model_name.capitalize()
    backend_label = f"local {state.backend} {state.quantization}" if state.backend else "no backend"
    ctx_label = f"ctx {_context_str(state.context_length)}"
    ram_label = _format_ram(state.ram_used_gb, state.ram_total_gb)
    cua_label = f"CUA {state.cua_mode}" if state.cua_mode != "off" else "CUA off"
    skills_label = f"skills {state.skill_count}"
    tools_label = f"tools {state.tool_count}"
    profile_label = f"profile {state.profile_id}"

    segments = [
        f"Aurelius {model_label}",
        f"{backend_label}",
        ctx_label,
        f"RAM {ram_label}",
        cua_label,
        skills_label,
        tools_label,
        profile_label,
    ]

    line = " | ".join(segments)

    if len(line) > width:
        # Truncate intelligently — keep model/backend/ctx/ram always visible
        segments = segments[:5]
        line = " | ".join(segments)
        if len(line) > width:
            line = line[: width - 3] + "..."

        line = line.ljust(width, " ")
    else:
        line += " " * (width - len(line))

    return line


def render_status_bar_plain(state: StatusBarState) -> str:
    """Plain-text status bar without padding (for non-interactive output)."""
    segments = []
    segments.append(f"Aurelius {state.model_name.capitalize()}")
    segments.append(f"{'local' if state.backend else 'no'} {state.backend or 'backend'}")
    if state.quantization:
        segments[-1] += f" {state.quantization}"
    segments.append(f"ctx {_context_str(state.context_length)}")
    segments.append(
        f"RAM {state.ram_used_gb:.1f}/{state.ram_total_gb:.1f}GB"
        if state.ram_total_gb > 0
        else f"RAM {state.ram_used_gb:.1f}GB"
    )
    if state.cua_mode != "off":
        segments.append(f"CUA {state.cua_mode}")
    segments.append(f"skills {state.skill_count}")
    segments.append(f"tools {state.tool_count}")
    segments.append(f"profile {state.profile_id}")
    return " | ".join(segments)
