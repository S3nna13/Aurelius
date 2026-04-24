"""
workflow_visualizer.py — Text/DOT/Mermaid visualizations of workflow DAGs.
Stdlib-only. Exports WORKFLOW_VISUALIZER_REGISTRY.
"""

from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


class NodeStyle(enum.Enum):
    DEFAULT = "default"
    HIGHLIGHTED = "highlighted"
    FAILED = "failed"
    COMPLETED = "completed"


# Map NodeStyle -> DOT fillcolor
_FILLCOLOR: Dict[NodeStyle, str] = {
    NodeStyle.DEFAULT: "white",
    NodeStyle.HIGHLIGHTED: "yellow",
    NodeStyle.FAILED: "red",
    NodeStyle.COMPLETED: "green",
}


@dataclass(frozen=True)
class VisualizerConfig:
    indent: int = 2
    show_metadata: bool = False


class WorkflowVisualizer:
    """Generates text/DOT/Mermaid visualizations of workflow DAGs."""

    def __init__(self, config: Optional[VisualizerConfig] = None) -> None:
        self.config: VisualizerConfig = config if config is not None else VisualizerConfig()

    # ------------------------------------------------------------------
    # DOT output
    # ------------------------------------------------------------------

    def to_dot(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
        node_styles: Optional[Dict[str, NodeStyle]] = None,
    ) -> str:
        """
        Generate a DOT language digraph string.

        Each node: "node_id" [label="node_id", style=filled, fillcolor=<color>];
        Each edge: "src" -> "dst";
        """
        if node_styles is None:
            node_styles = {}

        lines: List[str] = ["digraph {"]

        for node in nodes:
            style = node_styles.get(node, NodeStyle.DEFAULT)
            color = _FILLCOLOR[style]
            lines.append(
                f'    "{node}" [label="{node}", style=filled, fillcolor={color}];'
            )

        for src, dst in edges:
            lines.append(f'    "{src}" -> "{dst}";')

        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # ASCII / text tree output
    # ------------------------------------------------------------------

    def to_ascii(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
    ) -> str:
        """
        Simple text tree. Roots (nodes with no incoming edges) are printed
        first; children are indented by config.indent spaces. Uses
        topological (BFS) order.
        """
        if not nodes:
            return ""

        # Build adjacency and in-degree structures
        children: Dict[str, List[str]] = {n: [] for n in nodes}
        in_degree: Dict[str, int] = {n: 0 for n in nodes}

        for src, dst in edges:
            if dst in in_degree:
                in_degree[dst] += 1
            if src in children:
                children[src].append(dst)

        # Roots = nodes with no incoming edges
        roots = [n for n in nodes if in_degree[n] == 0]

        lines: List[str] = []
        indent_str = " " * self.config.indent

        # BFS from roots, tracking depth
        queue: deque = deque()
        for root in roots:
            queue.append((root, 0))

        visited: set = set()
        while queue:
            node, depth = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            lines.append(indent_str * depth + node)
            for child in children[node]:
                queue.append((child, depth + 1))

        # Handle any nodes not reachable from roots (cycles / isolated)
        for node in nodes:
            if node not in visited:
                lines.append(node)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Mermaid output
    # ------------------------------------------------------------------

    def to_mermaid(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str]],
    ) -> str:
        """
        Generate a Mermaid flowchart LR diagram.

        Each node: id[id]
        Each edge: src --> dst
        """
        lines: List[str] = ["flowchart LR"]

        for node in nodes:
            lines.append(f"    {node}[{node}]")

        for src, dst in edges:
            lines.append(f"    {src} --> {dst}")

        return "\n".join(lines)


# Public registry
WORKFLOW_VISUALIZER_REGISTRY: dict = {"default": WorkflowVisualizer}
