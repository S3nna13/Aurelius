"""AST-aware fill-in-the-middle (FIM) processor for Aurelius.

Inspired by StarCoder2 PSM/SPM FIM (bigcode, Apache-2.0),
Kimi-Dev patch synthesis (MoonshotAI, Apache-2.0, 2025),
Aider unified-diff format (MIT), clean-room reimplementation.

Uses stdlib ``ast`` only (no tree-sitter, no libcst).
"""

from __future__ import annotations

import ast
import enum
import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FIMFormat(enum.Enum):
    """Fill-in-the-middle formatting strategy."""

    PSM = "prefix-suffix-middle"
    SPM = "suffix-prefix-middle"
    RANDOM = "random"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ASTNode:
    """Lightweight representation of a parsed AST node."""

    node_type: str
    start_line: int
    end_line: int
    name: str | None = None
    children: list[ASTNode] = field(default_factory=list)


@dataclass
class FIMSpan:
    """A fill-in-the-middle span consisting of prefix, suffix, and middle."""

    prefix: str
    suffix: str
    middle: str
    language: str = "python"
    cursor_line: int | None = None


# ---------------------------------------------------------------------------
# ASTAnalyzer
# ---------------------------------------------------------------------------


class ASTAnalyzer:
    """Python AST analysis utilities (stdlib ``ast`` only)."""

    def parse_python(self, source: str) -> list[ASTNode]:
        """Parse *source* and return top-level AST nodes.

        Returns an empty list on syntax error (never raises).

        Only functions, classes, and import statements are returned at the
        top level; all other top-level nodes are skipped.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        nodes: list[ASTNode] = []
        for node in ast.iter_child_nodes(tree):
            ast_node = self._convert_node(node)
            if ast_node is not None:
                nodes.append(ast_node)
        return nodes

    def find_enclosing_scope(self, source: str, line: int) -> ASTNode | None:
        """Return the innermost function/class node containing *line*.

        *line* is 1-based. Returns ``None`` if no such scope exists or on
        parse error.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        best: ASTNode | None = None
        best_size = float("inf")

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
                continue
            start = node.lineno
            end = node.end_lineno or node.lineno
            if start <= line <= end:
                size = end - start
                if size < best_size:
                    best_size = size
                    best = self._convert_node(node)

        return best

    def extract_context(
        self,
        source: str,
        cursor_line: int,
        context_lines: int = 20,
    ) -> tuple[str, str]:
        """Split *source* at *cursor_line* with a ±*context_lines* window.

        *cursor_line* is 1-based. Returns ``(prefix, suffix)`` where:
        - ``prefix`` = lines from ``max(1, cursor_line - context_lines)``
          up to (not including) ``cursor_line``.
        - ``suffix`` = lines from ``cursor_line`` up to
          ``cursor_line + context_lines``.

        Both are joined with newlines.
        """
        all_lines = source.splitlines(keepends=True)
        total = len(all_lines)
        # Convert to 0-based index
        idx = max(0, cursor_line - 1)

        prefix_start = max(0, idx - context_lines)
        prefix_lines = all_lines[prefix_start:idx]
        prefix = "".join(prefix_lines)

        suffix_end = min(total, idx + context_lines)
        suffix_lines = all_lines[idx:suffix_end]
        suffix = "".join(suffix_lines)

        return prefix, suffix

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _convert_node(self, node: ast.AST) -> ASTNode | None:
        """Convert an ``ast.AST`` node to an :class:`ASTNode`."""
        node_type: str
        name: str | None = None
        start_line: int = getattr(node, "lineno", 0)
        end_line: int = getattr(node, "end_lineno", start_line) or start_line

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node_type = "function"
            name = node.name
        elif isinstance(node, ast.ClassDef):
            node_type = "class"
            name = node.name
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            node_type = "import"
        elif isinstance(node, ast.Assign):
            node_type = "assignment"
        elif isinstance(node, ast.Expr):
            node_type = "expression"
        else:
            # Map by class name for other nodes
            node_type = type(node).__name__.lower()

        return ASTNode(
            node_type=node_type,
            start_line=start_line,
            end_line=end_line,
            name=name,
        )


# ---------------------------------------------------------------------------
# FIMTokenizer
# ---------------------------------------------------------------------------


class FIMTokenizer:
    """Format and parse fill-in-the-middle (FIM) prompts."""

    FIM_PREFIX_TOKEN: str = "<|fim_prefix|>"  # noqa: S105
    FIM_SUFFIX_TOKEN: str = "<|fim_suffix|>"  # noqa: S105
    FIM_MIDDLE_TOKEN: str = "<|fim_middle|>"  # noqa: S105
    FIM_PAD_TOKEN: str = "<|fim_pad|>"  # noqa: S105

    def format_psm(self, span: FIMSpan) -> str:
        """Format as prefix-suffix-middle (PSM) — StarCoder2 default."""
        return (
            f"{self.FIM_PREFIX_TOKEN}{span.prefix}"
            f"{self.FIM_SUFFIX_TOKEN}{span.suffix}"
            f"{self.FIM_MIDDLE_TOKEN}"
        )

    def format_spm(self, span: FIMSpan) -> str:
        """Format as suffix-prefix-middle (SPM) — Qwen2.5-Coder style."""
        return (
            f"{self.FIM_SUFFIX_TOKEN}{span.suffix}"
            f"{self.FIM_PREFIX_TOKEN}{span.prefix}"
            f"{self.FIM_MIDDLE_TOKEN}"
        )

    def format_span(self, span: FIMSpan, fmt: FIMFormat = FIMFormat.PSM) -> str:
        """Format *span* using the given :class:`FIMFormat`.

        ``FIMFormat.RANDOM`` randomly picks PSM or SPM each call.
        """
        if fmt == FIMFormat.PSM:
            return self.format_psm(span)
        if fmt == FIMFormat.SPM:
            return self.format_spm(span)
        # RANDOM
        chosen = random.choice([FIMFormat.PSM, FIMFormat.SPM])  # noqa: S311
        return self.format_span(span, chosen)

    def parse_completion(self, completion: str) -> str:
        """Extract the filled-in middle text from a raw model completion.

        Strips all FIM tokens if present; returns the middle content.
        If ``FIM_MIDDLE_TOKEN`` is present, returns everything after the
        last occurrence of it (up to the next FIM token or end-of-string).
        Otherwise returns the stripped completion unchanged.
        """
        tokens = [
            self.FIM_PREFIX_TOKEN,
            self.FIM_SUFFIX_TOKEN,
            self.FIM_MIDDLE_TOKEN,
            self.FIM_PAD_TOKEN,
        ]

        if self.FIM_MIDDLE_TOKEN in completion:
            # Extract the portion after the middle token
            after_middle = completion.split(self.FIM_MIDDLE_TOKEN, 1)[1]
            # Strip any subsequent FIM tokens
            for tok in tokens:
                after_middle = after_middle.split(tok)[0]
            return after_middle

        # No middle token — strip all FIM tokens and return
        result = completion
        for tok in tokens:
            result = result.replace(tok, "")
        return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Registry mapping format names to the :class:`FIMTokenizer` class.
AST_FIM_REGISTRY: dict[str, type] = {
    "psm": FIMTokenizer,
    "spm": FIMTokenizer,
}


__all__ = [
    "ASTAnalyzer",
    "ASTNode",
    "AST_FIM_REGISTRY",
    "FIMFormat",
    "FIMSpan",
    "FIMTokenizer",
]
